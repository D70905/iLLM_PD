# -*- coding: utf-8 -*-
"""
rl/env.py — PavementEnv with HARA integration (Phase 2BC, FINAL v3)
====================================================================

v3 fix: adapt to actual run_fea(base_dir=...) signature:
  - run_fea returns a single dict with 'responses', 'run_dir', 'inputs' keys
  - base_dir is the PROJECT ROOT; FEA scratch goes to <base_dir>/output/runs/<run_name>/
  - cleanup must rmtree the returned run_dir (or its parent's child)
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fea.runner import run_fea
from specs import get_protocol
from specs.protocol import DesignInputs
from rl.guards import NumericalGuard, GuardViolation
from rl.reward import CompositeReward, RewardResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    protocol_name: str = "JTG_D50_2017"

    # 6-layer initial design (top-down): upper_AC / mid_AC / lower_AC / base / subbase
    init_thickness_m: List[float] = field(
        default_factory=lambda: [0.04, 0.06, 0.08, 0.36, 0.18])
    init_modulus_MPa: List[float] = field(
        default_factory=lambda: [14000.0, 11000.0, 9000.0, 1500.0, 400.0])
    init_poisson: List[float] = field(
        default_factory=lambda: [0.25, 0.30, 0.30, 0.25, 0.35])

    E_subgrade: float = 60.0
    nu_subgrade: float = 0.40
    load_pressure_MPa: float = 0.7
    load_radius_m: float = 0.1065

    city: str = "beijing"
    climate_zone: str = ""               # JTG climate zone: cold|temperate|warm|hot|tropical
                                         #   If non-empty, passed to JTG protocol as fallback
                                         #   when city lookup fails (e.g. for non-Chinese sites).
    road_class: str = "expressway"
    traffic_level: str = "heavy"
    pavement_type: str = "semi_rigid"
    design_life_years: int = 15

    # Optional continuous spec inputs for OOD studies. When left as None,
    # protocols keep their original city/climate_zone and traffic_level paths.
    MAAT_C: Optional[float] = None
    annual_ESAL_BZZ100: Optional[float] = None
    total_ESAL_BZZ100: Optional[float] = None
    traffic_growth_rate: float = 0.0
    # Phase 2D: LCC + DSR/SCR post-evaluation (R1-2 + R2-2)
    enable_lcc_eval: bool = True
    design_life_years_lcc: float = 20.0   # FHWA-recommended for asphalt

    action_dh_max_m: float = 0.02
    action_dE_max_MPa: float = 100.0

    max_episode_steps: int = 20
    max_episodes: int = 200

    # FEA: project root for ABAQUS runs (run_fea spawns <base_dir>/output/runs/<run_name>/)
    fea_base_dir: Optional[str] = None     # default = os.getcwd()
    fea_num_cpus: int = 4
    fea_verbose: bool = False              # False = quiet ABAQUS output
    fea_keep_runs: bool = False            # False = rmtree the FEA scratch dir per step

    # Phase 2F ablation: reward-only constraint enforcement
    guard_enabled: bool = True              # False = NumericalGuard bypassed (Reward-only)

    # Surrogate data collection (Phase 2D prep)
    surrogate_data_path: Optional[str] = None  # JSONL file to log (design→FEA) pairs

    # LLM integration
    llm_enabled: bool = True
    evaluator: Optional[Any] = None
    evaluator_in_training_loop: bool = False   # R3-3: training Evaluator gated off by default;
                                                # its LLM capability moved to design_explainer output port
    generator: Optional[Any] = None
    audit_chain: Optional[Any] = None

    # Climate-coupled design temperature (R1-2 / R2-1)
    climate_enabled: bool = False                     # if True, AC moduli are temperature-adjusted
    design_temperature_C: Optional[float] = None       # hot-month pavement temp for design eval

    eval_alert_threshold: float = 2.0
    strict_mode_steps: int = 30
    log_every_n_steps: int = 10


# ─────────────────────────────────────────────────────────────
# Env
# ─────────────────────────────────────────────────────────────

class PavementEnv(gym.Env):
    """6-layer pavement design env with LLM HARA hooks."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(6,), dtype=np.float32)

        self.protocol = get_protocol(self.config.protocol_name)
        self.guard = NumericalGuard(base_type=getattr(self.config, "pavement_type", "semi_rigid"))
        self.reward_fn = CompositeReward()

        self._design: Dict[str, np.ndarray] = {}
        self._last_evaluation = None
        self._last_responses: Dict[str, float] = {}
        self._last_action = np.zeros(10, dtype=np.float32)
        self._last_action_norm = 0.0
        self.episode_step = 0
        self.total_episodes = 0
        self.global_step = 0

        # Cache Generator.blend static method (if generator provided)
        self._generator_blend = None
        if self.config.generator is not None:
            self._generator_blend = getattr(
                type(self.config.generator), "blend", None)

        logger.info(
            "PavementEnv ready (protocol={}, llm_enabled={}, eval={}, gen={}, audit={})".format(
                self.config.protocol_name,
                self.config.llm_enabled,
                "on" if self.config.evaluator else "off",
                "on" if self.config.generator else "off",
                "on" if self.config.audit_chain else "off",
            ))

        # ── Phase 2D: LCC + DSR/SCR post-evaluation (R1-2 + R2-2) ──
        self._compliance_history: list[bool] = []
        self._guard_violation_count: int = 0
        # R1-3 / R2-2: track best compliant design per episode
        self._best_design: Optional[Dict[str, np.ndarray]] = None
        self._best_cost: float = float('inf')
        self._best_evaluation = None
        self._best_responses = None
        self._lcc_evaluator = None
        self._metrics_module = None
        self._dsr_module = None          # weak-link DSR (R2-2); may be overridden below
        if getattr(self.config, "enable_lcc_eval", False):
            try:
                from rl.lifecycle_lcc_intl import lcc_npv_usd as _lcc_fn
                from rl import metrics as _metrics
                from rl import dsr_patch as _dsr_patch
                self._lcc_evaluator = _lcc_fn
                self._metrics_module = _metrics
                self._dsr_module = _dsr_patch
                logger.info("LCC + DSR/SCR post-eval ENABLED "
                            "(design_life_years_lcc={}, FHWA 4% discount)".format(
                                self.config.design_life_years_lcc))
            except ImportError as e:
                logger.warning("LCC modules unavailable, post-eval DISABLED: {}".format(e))

    # ─── Trainer hooks (kept for back-compat; not strictly needed by run_fea) ──
    def set_fea_output_dir(self, path: Path) -> None:
        """
        For run_fea, this is the PROJECT ROOT (base_dir), not a sub-output dir.
        run_fea will create <path>/output/runs/<timestamp>/ for each call.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.config.fea_base_dir = str(p)

    def set_episode_counter(self, total_episodes: int, global_step: int) -> None:
        self.total_episodes = total_episodes
        self.global_step = global_step

    # ─── Gym API ───────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._design = {
            "thickness": np.array(self.config.init_thickness_m, dtype=np.float64),
            "modulus":   np.array(self.config.init_modulus_MPa, dtype=np.float64),
            "poisson":   np.array(self.config.init_poisson,     dtype=np.float64),
        }
        self.episode_step = 0
        self._last_action = np.zeros(10, dtype=np.float32)
        self._last_action_norm = 0.0

        fea_responses, evaluation, _ = self._run_fea_and_evaluate(self._design)
        self._last_responses = fea_responses
        self._last_evaluation = evaluation

        if self.config.audit_chain is not None:
            try:
                self.config.audit_chain.record("reset", {
                    "episode": int(self.total_episodes),
                    "design":  self._design_to_dict(),
                    "margins": {k: float(v) for k, v in evaluation.margins.items()},
                    "feasible": bool(evaluation.feasible),
                    "critical": str(evaluation.critical_indicator),
                })
            except Exception as e:
                logger.warning("Audit record (reset) failed: {}".format(e))

        margins_str = {k: "{:.2f}".format(v) for k, v in evaluation.margins.items()}
        logger.info("[Env.reset] feasible={} critical={} margins={}".format(
            evaluation.feasible, evaluation.critical_indicator, margins_str))

        obs = self._make_observation(evaluation)
        info = self._make_info(evaluation, reward_result=None)
        # Phase 2D: LCC + DSR/SCR post-eval on initial design
        self._compliance_history = []  # reset per episode
        self._guard_violation_count = 0  # reset per episode
        self._best_design = None         # track cheapest compliant design
        self._best_cost = float('inf')
        self._best_evaluation = None
        self._best_responses = None
        # Track initial design
        if bool(evaluation.feasible):
            init_cost = self.reward_fn._material_cost(
                self._design["thickness"], self._design["modulus"],
                pavement_type=self.config.pavement_type)
            self._best_design = {
                "thickness": self._design["thickness"].copy(),
                "modulus": self._design["modulus"].copy(),
            }
            self._best_cost = float(init_cost)
            self._best_evaluation = evaluation
            self._best_responses = fea_responses
        post = self._compute_post_eval(
            margins={k: float(v) for k, v in evaluation.margins.items()},
            new_design={"thickness": self._design["thickness"],
                        "modulus":   self._design["modulus"]},
            feasible=bool(evaluation.feasible),
        )
        info.update(post)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        action_ppo = action.copy()

        # 1. Generator may propose; env does the blend
        action_used, gen_result = self._maybe_generator(action_ppo)

        # 2. Apply action → new design
        new_design = self._apply_action(self._design, action_used)

        # 3. Pre-FEA guard (bypassed if guard_enabled=False for Reward-only ablation)
        guard_blocked = False
        guard_code: Optional[str] = None
        if self.config.guard_enabled:
            try:
                self.guard.check_design(
                    thickness=new_design["thickness"],
                    modulus=new_design["modulus"],
                    E_subgrade=self.config.E_subgrade,
                )
            except GuardViolation as gv:
                guard_blocked = True
                guard_code = gv.code
                logger.warning("[Env.step {}.{}] Pre-FEA guard: {} — {}".format(
                    self.total_episodes, self.episode_step, gv.code, gv))

        if guard_blocked:
            return self._handle_guard_block(action_ppo, action_used, gen_result, guard_code)

        # 4. Submit Evaluator async BEFORE FEA
        critical = (self._last_evaluation.critical_indicator
                    if self._last_evaluation else None)
        last_margins = (self._last_evaluation.margins
                        if self._last_evaluation else {})
        evaluator_future = self._maybe_evaluator_submit(
            new_design, last_margins, action_used, critical)

        # 5. Run FEA
        fea_responses, evaluation, fea_time = self._run_fea_and_evaluate(new_design)

        # 6. Post-FEA guard
        try:
            self.guard.check_fea_result(fea_responses)
        except GuardViolation as gv:
            logger.warning("[Env.step {}.{}] Post-FEA guard: {} — fallback".format(
                self.total_episodes, self.episode_step, gv.code))
            evaluation = self._last_evaluation
            fea_responses = self._last_responses

        # 7. Reward
        last_design = self._design
        self._design = new_design
        self.episode_step += 1
        self.global_step += 1
        self._last_action = action_used
        self._last_action_norm = float(np.linalg.norm(action_used) / np.sqrt(10))

        reward_result: RewardResult = self.reward_fn.compute(
            margins={k: float(v) for k, v in evaluation.margins.items()},
            new_design={
                "thickness": new_design["thickness"],
                "modulus":   new_design["modulus"],
            },
            old_design={
                "thickness": last_design["thickness"],
                "modulus":   last_design["modulus"],
            },
            tau=self._tau(),
            critical_indicator=evaluation.critical_indicator,
            feasible=bool(evaluation.feasible),
            pavement_type=self.config.pavement_type,  # ← NEW: type-aware pricing
        )

        self._last_evaluation = evaluation
        self._last_responses = fea_responses

        # ── Track cheapest fully-compliant design in this episode ──
        if bool(evaluation.feasible) and not guard_blocked:
            cur_cost = self.reward_fn._material_cost(
                new_design["thickness"], new_design["modulus"],
                pavement_type=self.config.pavement_type)
            if cur_cost < self._best_cost:
                self._best_cost = float(cur_cost)
                self._best_design = {
                    "thickness": new_design["thickness"].copy(),
                    "modulus": new_design["modulus"].copy(),
                }
                self._best_evaluation = evaluation
                self._best_responses = fea_responses

        # 8. Collect Evaluator result
        eval_result = self._maybe_evaluator_collect(evaluator_future)

        # 9. Step summary
        if self.global_step % self.config.log_every_n_steps == 0:
            self._print_step_summary(
                evaluation, reward_result, eval_result, gen_result, fea_time)

        # 10. Env-level audit
        self._audit_step(action_ppo, action_used, gen_result, eval_result,
                         evaluation, reward_result, guard_blocked=False,
                         guard_code=None)

        # 11. Low-score alert
        if eval_result is not None and getattr(eval_result, "success", False):
            score = float(getattr(eval_result, "score", 5.0))
            if score <= self.config.eval_alert_threshold:
                logger.warning(
                    "[LOW EVAL {:.1f}/10] ep{} step{}: {}".format(
                        score, self.total_episodes, self.episode_step,
                        str(getattr(eval_result, "reasoning", ""))[:140]))

        done = self.episode_step >= self.config.max_episode_steps
        if done:
            self.total_episodes += 1
            # R1-3/R2-2: deliver the cheapest fully-compliant design,
            # not the last step which may have overshot.
            if self._best_design is not None:
                logger.info(
                    "[Env.deliver] best compliant design cost={:.1f} CNY/m2 "
                    "(vs last-step cost)".format(self._best_cost))

        # ── Post-eval uses REAL last-step margins (honest episode SCR) ──
        obs = self._make_observation(evaluation)
        info = self._make_info(evaluation, reward_result,
                               eval_result=eval_result, gen_result=gen_result)
        post = self._compute_post_eval(
            margins={k: float(v) for k, v in evaluation.margins.items()},
            new_design=new_design,
            feasible=bool(evaluation.feasible),
        )
        # ── Append delivered design fields (may differ from last step) ──
        if done and self._best_design is not None:
            delivered_design = {
                "thickness": self._best_design["thickness"],
                "modulus": self._best_design["modulus"],
            }
            delivered_cost = self.reward_fn._material_cost(
                delivered_design["thickness"], delivered_design["modulus"],
                pavement_type=self.config.pavement_type)
            info["delivered_design"] = delivered_design
            info["delivered_cost_cny"] = float(delivered_cost)
            if self._best_evaluation is not None:
                info["delivered_dsr"] = self._dsr_module.compute_dsr(
                    {k: float(v) for k, v in self._best_evaluation.margins.items()}
                ) if self._dsr_module else None
                info["delivered_margins"] = {
                    k: float(v) for k, v in self._best_evaluation.margins.items()}
        info.update(post)
        return obs, float(reward_result.total), done, False, info

    def close(self):
        if self.config.audit_chain is not None:
            try:
                self.config.audit_chain.close()
            except Exception:
                pass

    # ─── Phase 2D: LCC + DSR/SCR post-eval (R1-2 + R2-2) ─────
    def _compute_post_eval(self, margins, new_design, feasible):
        """Compute LCC NPV (USD/m2) + DSR + running SCR. Returns dict."""
        out = {}
        if self._metrics_module is not None:
            try:
                # DSR: weak-link min(1, min margin) — R2-2 locked; prefer dsr_patch
                out["dsr"] = (self._dsr_module.compute_dsr(margins)
                              if self._dsr_module is not None
                              else self._metrics_module.compute_dsr(margins))
                is_compliant = self._metrics_module.compute_compliance(margins)
                self._compliance_history.append(is_compliant)
                out["compliant"] = bool(is_compliant)
                out["scr_running"] = (
                    sum(1 for c in self._compliance_history if c)
                    / max(len(self._compliance_history), 1))
            except Exception as e:
                logger.debug("DSR/SCR compute failed: {}".format(e))
        if self._lcc_evaluator is not None:
            try:
                C_const_cny = self.reward_fn._material_cost(
                    np.asarray(new_design["thickness"]),
                    np.asarray(new_design["modulus"]),
                    pavement_type=self.config.pavement_type,
                )
                C_const_usd = float(C_const_cny) / 7.20
                margin_B1 = float(margins.get("B1_asphalt_fatigue", float("inf")))
                margin_B2 = float(margins.get("B2_semi_rigid_fatigue", float("inf")))
                lcc = self._lcc_evaluator(
                    C_construction_usd_per_m2=C_const_usd,
                    design_life_years=float(self.config.design_life_years_lcc),
                    margin_B1=margin_B1,
                    margin_B2=margin_B2,
                    discount_rate=0.04,
                )
                out["lcc"] = {
                    "C_construction_usd_per_m2": C_const_usd,
                    "C_construction_cny_per_m2": float(C_const_cny),
                    "NPV_total_usd_m2": lcc.get("NPV_total_usd_m2"),
                    "C_maint_NPV_usd_m2": lcc.get("C_maintenance_NPV_usd_m2"),
                    "n_events": lcc.get("n_events"),
                    "discount_rate": 0.04,
                    "analysis_years": float(self.config.design_life_years_lcc),
                }
            except Exception as e:
                logger.debug("LCC compute failed: {}".format(e))
        return out

    # ─── Guard-block handler ───────────────────────────────────
    def _handle_guard_block(self, action_ppo, action_used, gen_result, guard_code):
        self._guard_violation_count += 1
        self.episode_step += 1
        self.global_step += 1
        self._last_action = action_used
        self._last_action_norm = float(np.linalg.norm(action_used) / np.sqrt(10))

        ev = self._last_evaluation
        reward_result = self.reward_fn.compute(
            margins={k: float(v) for k, v in ev.margins.items()} if ev else {},
            new_design={"thickness": self._design["thickness"],
                        "modulus":   self._design["modulus"]},
            old_design={"thickness": self._design["thickness"],
                        "modulus":   self._design["modulus"]},
            tau=self._tau(),
            critical_indicator=(ev.critical_indicator if ev else None),
            feasible=False,
            pavement_type=self.config.pavement_type,  # NEW: type-aware pricing
        )
        penalized_total = -1.5

        self._audit_step(action_ppo, action_used, gen_result,
                         eval_result=None,
                         evaluation=ev, reward_result=reward_result,
                         guard_blocked=True, guard_code=guard_code)

        done = self.episode_step >= self.config.max_episode_steps
        if done:
            self.total_episodes += 1

        obs = self._make_observation(ev)
        info = self._make_info(ev, reward_result, guard_block=guard_code)
        post = self._compute_post_eval(
            margins={k: float(v) for k, v in ev.margins.items()} if ev else {},
            new_design={"thickness": self._design["thickness"],
                        "modulus":   self._design["modulus"]},
            feasible=False,
        )
        if done and self._best_design is not None:
            delivered_design = {
                "thickness": self._best_design["thickness"],
                "modulus": self._best_design["modulus"],
            }
            delivered_cost = self.reward_fn._material_cost(
                delivered_design["thickness"], delivered_design["modulus"],
                pavement_type=self.config.pavement_type)
            info["delivered_design"] = delivered_design
            info["delivered_cost_cny"] = float(delivered_cost)
            if self._best_evaluation is not None:
                info["delivered_dsr"] = self._dsr_module.compute_dsr(
                    {k: float(v) for k, v in self._best_evaluation.margins.items()}
                ) if self._dsr_module else None
                info["delivered_margins"] = {
                    k: float(v) for k, v in self._best_evaluation.margins.items()}
        info.update(post)
        return obs, float(penalized_total), done, False, info

    # ─── LLM hooks ─────────────────────────────────────────────
    def _maybe_generator(self, action_ppo: np.ndarray) -> Tuple[np.ndarray, Any]:
        if not self.config.llm_enabled or self.config.generator is None:
            return action_ppo, None

        ev = self._last_evaluation
        margins = {k: float(v) for k, v in ev.margins.items()} if ev else {}
        critical = ev.critical_indicator if ev else None
        last_infeas = bool(ev and not ev.feasible)

        try:
            result = self.config.generator.propose(
                thickness=self._design["thickness"].tolist(),
                modulus=self._design["modulus"].tolist(),
                margins=margins,
                action_PPO=np.asarray(action_ppo, dtype=np.float32),
                episode=int(self.total_episodes),
                step=int(self.episode_step),
                tau=float(self._tau()),
                critical_indicator=critical,
                last_step_was_infeasible=last_infeas,
                climate_zone=self.config.climate_zone,
                pavement_type=self.config.pavement_type,
            )
        except Exception as e:
            self._handle_llm_error("generator", e)
            return action_ppo, None

        if result is None:
            return action_ppo, None

        gen_action = getattr(result, "action", None)
        alpha = float(getattr(result, "alpha_used", 0.0))
        if gen_action is None or alpha <= 0.0:
            return action_ppo, result

        if self._generator_blend is not None:
            try:
                blended = self._generator_blend(
                    np.asarray(action_ppo, dtype=np.float32),
                    np.asarray(gen_action, dtype=np.float32),
                    alpha,
                )
                return np.asarray(blended, dtype=np.float32), result
            except Exception:
                pass

        action_ppo_arr = np.asarray(action_ppo, dtype=np.float32)
        gen_action_arr = np.asarray(gen_action, dtype=np.float32)
        blended = (1.0 - alpha) * action_ppo_arr + alpha * gen_action_arr
        return np.clip(blended, -1.0, 1.0).astype(np.float32), result

    def _maybe_evaluator_submit(self, new_design, margins, action_used, critical):
        # R3-3: training Evaluator gated out of the loop (score uncorrelated with
        # compliance, deterministic Guard already catches unsafe designs).
        # LLM capability moved to output port (design_explainer).
        if not getattr(self.config, "evaluator_in_training_loop", False):
            return None
        if not self.config.llm_enabled or self.config.evaluator is None:
            return None
        try:
            return self.config.evaluator.evaluate_async(
                thickness=new_design["thickness"].tolist(),
                modulus=new_design["modulus"].tolist(),
                margins={k: float(v) for k, v in margins.items()},
                action=np.asarray(action_used, dtype=np.float32),
                episode=int(self.total_episodes),
                step=int(self.episode_step),
                critical_indicator=critical,
            )
        except Exception as e:
            self._handle_llm_error("evaluator_submit", e)
            return None

    def _maybe_evaluator_collect(self, future):
        if future is None:
            return None
        try:
            return self.config.evaluator.collect(future)
        except Exception as e:
            self._handle_llm_error("evaluator_collect", e)
            return None

    def _handle_llm_error(self, where: str, err: Exception):
        if self.global_step < self.config.strict_mode_steps:
            logger.error("[STRICT step={}] LLM error in {}: {}".format(
                self.global_step, where, err))
            raise RuntimeError("LLM error in {} during strict mode: {}".format(where, err))
        else:
            logger.warning("LLM degradation in {}: {}".format(where, err))

    # ─── Env-level audit ───────────────────────────────────────
    def _audit_step(self, action_ppo, action_used, gen_result, eval_result,
                    evaluation, reward_result: RewardResult,
                    guard_blocked: bool, guard_code: Optional[str]):
        if self.config.audit_chain is None:
            return
        try:
            gen_was_called = bool(getattr(gen_result, "was_called", False)) if gen_result else False
            gen_success    = bool(getattr(gen_result, "success", False))    if gen_result else False
            eval_success   = bool(getattr(eval_result, "success", False))   if eval_result else False
            eval_score = None
            if eval_result and eval_success:
                eval_score = float(getattr(eval_result, "score", 5.0))

            entry = {
                "episode":      int(self.total_episodes),
                "step":         int(self.episode_step),
                "global_step":  int(self.global_step),
                "tau":          float(self._tau()),
                "design":       self._design_to_dict(),
                "action_ppo":   [float(x) for x in action_ppo],
                "action_used":  [float(x) for x in action_used],
                "guard_blocked": bool(guard_blocked),
                "guard_code":   guard_code,
                "reward_total": float(reward_result.total),
                "reward_components": {k: float(v) for k, v in reward_result.components.items()},
                "feasible":     bool(evaluation.feasible) if evaluation else False,
                "critical":     str(evaluation.critical_indicator) if evaluation else "unknown",
                "margins":      ({k: float(v) for k, v in evaluation.margins.items()}
                                 if evaluation else {}),
                "gen_was_called": gen_was_called,
                "gen_success":    gen_success,
                "eval_success":   eval_success,
                "eval_score":     eval_score,
            }
            self.config.audit_chain.record("step", entry)
        except Exception as e:
            logger.warning("Audit record (step) failed: {}".format(e))

    # ─── FEA + Spec — VERSION 3 ──────────────────────────────────
    def _fea_failure_fallback(self, design):
        """Return synthetic failed evaluation when FEA crashes (reward-only mode)."""
        from specs.protocol import DesignEvaluation
        empty_responses = {
            "epsilon_a_microstrain": 9999.0, "sigma_t_MPa": 9999.0,
            "epsilon_z_microstrain": 9999.0,
            "p_AC_upper_mid_MPa": 9999.0, "p_AC_mid_mid_MPa": 9999.0,
            "p_AC_lower_mid_MPa": 9999.0,
        }
        return empty_responses, DesignEvaluation(
            feasible=False,
            margins={"FEA_FAILURE": 0.0},
            responses=empty_responses,
            allowable_values={"FEA_FAILURE": 0.0},
            critical_indicator="FEA_FAILURE",
            spec_name=self.config.protocol_name,
            details={"_fea_crashed": True},
        ), 0.0

    def _design_extras(self) -> Dict[str, Any]:
        extras = {
            "city":         self.config.city,
            "climate_zone": self.config.climate_zone,
            "VFA_pct":      70.0,
            "R_s_MPa":      1.0,
            "R_0_mm":       1.5,
        }
        optional = {
            "MAAT_C":              getattr(self.config, "MAAT_C", None),
            "annual_ESAL_BZZ100":  getattr(self.config, "annual_ESAL_BZZ100", None),
            "total_ESAL_BZZ100":   getattr(self.config, "total_ESAL_BZZ100", None),
            "traffic_growth_rate": getattr(self.config, "traffic_growth_rate", None),
            "traffic_design_life_years": getattr(self.config, "design_life_years", None),
        }
        for key, value in optional.items():
            if value is not None:
                extras[key] = value
        return extras

    def _build_design_inputs(self, design) -> DesignInputs:
        return DesignInputs(
            pavement_type=self.config.pavement_type,
            road_class=self.config.road_class,
            traffic_level=self.config.traffic_level,
            thickness=design["thickness"].tolist(),
            modulus=design["modulus"].tolist(),
            poisson=design["poisson"].tolist(),
            E_subgrade=self.config.E_subgrade,
            nu_subgrade=self.config.nu_subgrade,
            design_life=self.config.design_life_years,
            extras=self._design_extras(),
        )

    def _run_fea_and_evaluate(self, design):
        """
        Calls run_fea() with its actual signature.
        On FEA failure (e.g. reward-only mode with extreme designs),
        returns a synthetic failed evaluation so training can continue.
        """
        t0 = time.time()
        run_dir = None

        # ── Climate-coupled temperature adjustment (R1-2 / R2-1) ──
        fea_modulus = list(design["modulus"])  # will be overwritten if climate enabled
        if self.config.climate_enabled and self.config.design_temperature_C is not None:
            from rl.dynamic_modulus import DynamicModulusMasterCurve
            T = float(self.config.design_temperature_C)
            ac_anchors = fea_modulus[:3]  # strategy-selected = 20°C reference
            ac_service = []
            for a in ac_anchors:
                mc = DynamicModulusMasterCurve(E_ref_MPa=float(a), T_ref_C=20.0)
                ac_service.append(mc.modulus_MPa(T))
            fea_modulus = ac_service + fea_modulus[3:]  # base/subbase unchanged

        try:
            full_result = run_fea(
                thickness=design["thickness"].tolist(),
                modulus=fea_modulus,
                poisson=design["poisson"].tolist(),
                E_subgrade=self.config.E_subgrade,
                nu_subgrade=self.config.nu_subgrade,
                load_pressure=self.config.load_pressure_MPa,
                load_radius=self.config.load_radius_m,
                base_dir=self.config.fea_base_dir,
                num_cpus=self.config.fea_num_cpus,
                verbose=self.config.fea_verbose,
            )

            # run_fea returns a single dict; extract 'responses' sub-dict
            fea_responses = full_result.get("responses", full_result)
            run_dir = full_result.get("run_dir")

            inputs = self._build_design_inputs(design)
            evaluation = self.protocol.evaluate(inputs, fea_responses)

            # --- Append surrogate training data (Phase 2D prep) ---
            if self.config.surrogate_data_path:
                self._log_surrogate_pair(design, full_result)

            return fea_responses, evaluation, time.time() - t0

        except Exception as e:
            logger.warning(f"[Env] FEA crashed (reward-only mode?): {e}")
            return self._fea_failure_fallback(design)

        finally:
            if run_dir is not None and not self.config.fea_keep_runs:
                self._cleanup_fea_run(run_dir)

    def _log_surrogate_pair(self, design, full_result):
        """Append one (input, output) pair for surrogate model training."""
        responses = full_result.get("responses", {})
        row = {
            "input": {
                "thickness_m":    design["thickness"].tolist(),
                "modulus_MPa":    design["modulus"].tolist(),
                "poisson":        design["poisson"].tolist(),
                "E_subgrade_MPa": float(self.config.E_subgrade),
                "nu_subgrade":    float(self.config.nu_subgrade),
            },
            "output": {
                "epsilon_a_microstrain":   responses.get("epsilon_a_microstrain"),
                "sigma_t_MPa":             responses.get("sigma_t_MPa"),
                "epsilon_z_microstrain":   responses.get("epsilon_z_microstrain"),
                "p_AC_upper_mid_MPa":      responses.get("p_AC_upper_mid_MPa"),
                "p_AC_mid_mid_MPa":        responses.get("p_AC_mid_mid_MPa"),
                "p_AC_lower_mid_MPa":      responses.get("p_AC_lower_mid_MPa"),
                "D_FEA_mm":                full_result.get("D_FEA_mm"),
            },
            "ts": int(self.global_step),
        }
        try:
            with open(self.config.surrogate_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _cleanup_fea_run(self, run_dir):
        try:
            p = Path(run_dir)
            if p.exists() and p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    # ─── Helpers ───────────────────────────────────────────────
    def _apply_action(self, design, action):
        dh = action[:5] * self.config.action_dh_max_m
        dE = action[5:] * self.config.action_dE_max_MPa
        new_h = design["thickness"] + dh
        new_E = design["modulus"] + dE
        new_h = np.clip(new_h, 0.02, 0.60)
        new_E = np.clip(new_E, 100.0, 30000.0)
        return {
            "thickness": new_h,
            "modulus":   new_E,
            "poisson":   design["poisson"].copy(),
        }

    def _make_observation(self, evaluation):
        margins = evaluation.margins if evaluation else {}
        keys = ["B1_asphalt_fatigue", "B2_semi_rigid_fatigue",
                "B3_ac_permanent_deformation", "B4_subgrade_strain"]
        logs = []
        for k in keys:
            m = max(float(margins.get(k, 1.0)), 1e-6)
            logs.append(float(np.clip(np.log10(m), -2.0, 2.0)))
        return np.array([
            logs[0], logs[1], logs[2], logs[3],
            self.episode_step / self.config.max_episode_steps,
            self._last_action_norm,
        ], dtype=np.float32)

    def _make_info(self, evaluation, reward_result=None, **extra):
        info = {
            "feasible":   bool(evaluation.feasible) if evaluation else False,
            "critical":   str(evaluation.critical_indicator) if evaluation else "unknown",
            "margins":    ({k: float(v) for k, v in evaluation.margins.items()}
                           if evaluation else {}),
            "design_h_cm":  (self._design["thickness"] * 100).round(2).tolist(),
            "design_E_MPa": self._design["modulus"].round(0).tolist(),
            "episode_num":  int(self.total_episodes),
            "episode_step": int(self.episode_step),
            "global_step":  int(self.global_step),
            "n_guard_violations": self._guard_violation_count,
        }
        if reward_result is not None:
            info["reward_total"] = float(reward_result.total)
            info["reward_components"] = {k: float(v) for k, v in reward_result.components.items()}
            info["evaluation"] = {
                "margins":  info["margins"],
                "feasible": info["feasible"],
            }
        info.update(extra)
        return info

    def _design_to_dict(self):
        return {
            "thickness_cm": (self._design["thickness"] * 100).round(2).tolist(),
            "modulus_MPa":  self._design["modulus"].round(0).tolist(),
        }

    def _tau(self) -> float:
        return min(1.0, self.total_episodes / max(1, self.config.max_episodes))

    def _print_step_summary(self, evaluation, reward_result, eval_result, gen_result, fea_time):
        margins_str = " ".join(
            "{}={:.2f}".format(k.split("_")[0], v)
            for k, v in evaluation.margins.items())
        gen_str = ""
        if gen_result is not None and getattr(gen_result, "was_called", False):
            gen_str = " gen[conf={:.2f},α={:.2f}]".format(
                float(getattr(gen_result, "confidence", 0.0)),
                float(getattr(gen_result, "alpha_used", 0.0)),
            )
        eval_str = ""
        if eval_result is not None and getattr(eval_result, "success", False):
            eval_str = " eval={:.1f}/10".format(float(getattr(eval_result, "score", 0)))
        logger.info(
            "[step {:4d} ep{}.{}] r={:+.3f} feas={} crit={} {}{}{} fea={:.1f}s".format(
                self.global_step, self.total_episodes, self.episode_step,
                reward_result.total, evaluation.feasible,
                evaluation.critical_indicator, margins_str, gen_str, eval_str, fea_time))
