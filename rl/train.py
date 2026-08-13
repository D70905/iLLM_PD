"""
rl/train.py 閳?Phase 2BC + 2D training entry (v6)
==================================================

v6: Adds Phase 2D surrogate routing support.
  - --use-surrogate flag (or surrogate.use_surrogate=true in yaml)
  - When enabled, builds SurrogateEnvConfig + PavementEnvWithSurrogate
  - When disabled, behavior is IDENTICAL to v5 (Phase 2BC). Backward-compatible.

v5 (kept): adapted to verified working interfaces:
  - Evaluator(audit, backend, ...)
  - Generator(config=GeneratorConfig(backend), rag, audit, fail_fast)
  - get_client() factory handles .env automatically
  - PPOHyperparams + build_ppo
  - SB3 monkey-patch to prevent ep_info_buffer crash
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import signal
import sys
import time
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from rl.env import PavementEnv, EnvConfig
from rl.policy import PPOHyperparams, build_ppo, SpecMarginsLoggingCallback
from rl.evaluator import Evaluator
from rl.generator import Generator, GeneratorConfig
from rl.rag import RAGStore
from rl.audit import AuditChain
from rl.reward import CompositeReward, RewardConfig

# 閳光偓閳光偓 PHASE 2D: surrogate routing 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig

logger = logging.getLogger("rl.train")


# ??????????????????????????????????????????????????????????????????????
# NC revision sensitivity knobs
# ??????????????????????????????????????????????????????????????????????

def build_reward_config(profile: str) -> RewardConfig:
    """Return a RewardConfig for reviewer-requested reward sensitivity tests."""
    cfg = RewardConfig()
    profile = (profile or "baseline").strip().lower().replace("_", "-")
    if profile == "baseline":
        return cfg
    if profile == "performance-up-12p5":
        # One-dimensional refinement along the performance-weight axis:
        # performance 0.40 -> 0.45; other components reduced proportionally.
        weights = {'performance': 0.45, 'economic': 0.275, 'guidance': 0.1375, 'smoothness': 0.0917, 'exploration': 0.0458}
    elif profile == "performance-up":
        # Existing +25% perturbation, retained for continuity with completed runs.
        weights = {'performance': 0.50, 'economic': 0.25, 'guidance': 0.125, 'smoothness': 0.083, 'exploration': 0.042}
    elif profile == "performance-up-37p5":
        weights = {'performance': 0.55, 'economic': 0.225, 'guidance': 0.1125, 'smoothness': 0.075, 'exploration': 0.0375}
    elif profile == "performance-up-50p0":
        weights = {'performance': 0.60, 'economic': 0.20, 'guidance': 0.10, 'smoothness': 0.0667, 'exploration': 0.0333}
    elif profile == "performance-down":
        weights = {'performance': 0.30, 'economic': 0.35, 'guidance': 0.175, 'smoothness': 0.117, 'exploration': 0.058}
    elif profile == "economy-up":
        weights = {'performance': 0.35, 'economic': 0.45, 'guidance': 0.10, 'smoothness': 0.067, 'exploration': 0.033}
    elif profile == "economy-down":
        weights = {'performance': 0.45, 'economic': 0.20, 'guidance': 0.175, 'smoothness': 0.117, 'exploration': 0.058}
    elif profile == "no-directional-reward":
        weights = {'performance': 0.47, 'economic': 0.35, 'guidance': 0.00, 'smoothness': 0.12, 'exploration': 0.06}
    else:
        raise ValueError(f"Unknown reward profile: {profile}")
    cfg.weights_early = dict(weights)
    cfg.weights_mid = dict(weights)
    cfg.weights_late = dict(weights)
    return cfg

# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# SB3 monkey-patch  (unchanged from v5)
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def _apply_sb3_monkey_patch():
    """Defense against ep_info_buffer poisoning (TypeError: object of type 'int' has no len())."""
    _orig_update_info_buffer = BaseAlgorithm._update_info_buffer

    def _safe_update_info_buffer(self, infos, dones=None):
        safe_infos = []
        for info in (infos or []):
            if not isinstance(info, dict):
                safe_infos.append({})
                continue
            info_copy = dict(info)
            ep = info_copy.get("episode")
            if ep is not None and not isinstance(ep, dict):
                info_copy.pop("episode", None)
            safe_infos.append(info_copy)
        return _orig_update_info_buffer(self, safe_infos, dones)

    BaseAlgorithm._update_info_buffer = _safe_update_info_buffer

    _orig_dump_logs = OnPolicyAlgorithm.dump_logs

    def _safe_dump_logs(self, iteration=0):
        if hasattr(self, "ep_info_buffer") and self.ep_info_buffer is not None:
            try:
                clean = [x for x in self.ep_info_buffer if isinstance(x, dict)]
                if len(clean) != len(self.ep_info_buffer):
                    logger.warning(
                        f"Sanitized ep_info_buffer: dropped "
                        f"{len(self.ep_info_buffer) - len(clean)} non-dict entries")
                self.ep_info_buffer.clear()
                self.ep_info_buffer.extend(clean)
            except Exception as e:
                logger.warning(f"ep_info_buffer sanitize failed ({e}); resetting")
                self.ep_info_buffer.clear()
        return _orig_dump_logs(self, iteration)

    OnPolicyAlgorithm.dump_logs = _safe_dump_logs
    logger.info("SB3 monkey-patch applied (ep_info_buffer protection).")


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Config loading
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# v6 dual-base defaults (engineering-feasible init per type)
_DEFAULT_INIT_BY_TYPE = {
    "semi_rigid": {
        "thickness": [0.04, 0.06, 0.08, 0.36, 0.18],
        "modulus":   [14000.0, 11000.0, 9000.0, 1500.0, 400.0],
        "poisson":   [0.25, 0.30, 0.30, 0.25, 0.35],
    },
    "flexible": {
        # Granular base + thicker subbase (GPS-1 / ACUB profile)
        "thickness": [0.04, 0.06, 0.08, 0.30, 0.25],
        "modulus":   [14000.0, 11000.0, 9000.0, 350.0, 250.0],
        "poisson":   [0.25, 0.30, 0.30, 0.40, 0.35],
    },
}


def _resolve_pavement_type(args, rl_cfg: dict) -> str:
    """CLI > yaml > 'semi_rigid'. Normalize aliases."""
    cli_val = getattr(args, "pavement_type", None)
    yaml_val = rl_cfg.get("env", {}).get("pavement_type", None)
    pt = (cli_val or yaml_val or "semi_rigid").lower().strip()
    if pt in ("semirigid", "semi-rigid"):
        pt = "semi_rigid"
    if pt in ("unbound", "granular", "unbound_granular"):
        pt = "flexible"
    if pt not in ("semi_rigid", "flexible"):
        raise ValueError(
            "Unknown pavement_type {!r}; expected 'semi_rigid' or 'flexible'"
            .format(pt))
    return pt


def _resolve_init_layers(rl_cfg: dict, pt: str):
    """
    Per-type override > (generic yaml IF yaml.pavement_type matches pt) > default.

    This prevents cross-contamination when CLI overrides yaml's
    pavement_type but yaml still has the OLD type's init_* fields.
    """
    e = rl_cfg.get("env", {})
    d = _DEFAULT_INIT_BY_TYPE[pt]

    # Normalize yaml's pavement_type for comparison
    yaml_pt_raw = (e.get("pavement_type") or "semi_rigid")
    yaml_pt = str(yaml_pt_raw).lower().strip()
    if yaml_pt in ("semirigid", "semi-rigid"):
        yaml_pt = "semi_rigid"
    if yaml_pt in ("unbound", "granular", "unbound_granular"):
        yaml_pt = "flexible"
    yaml_matches = (yaml_pt == pt)

    # Per-type override always wins
    t_override = e.get("init_thickness_m_{}".format(pt))
    m_override = e.get("init_modulus_MPa_{}".format(pt))
    p_override = e.get("init_poisson_{}".format(pt))

    # Generic fallback valid ONLY if yaml's pavement_type matches requested
    if t_override is not None:
        t = t_override
    elif yaml_matches and "init_thickness_m" in e:
        t = e["init_thickness_m"]
    else:
        t = d["thickness"]

    if m_override is not None:
        m = m_override
    elif yaml_matches and "init_modulus_MPa" in e:
        m = e["init_modulus_MPa"]
    else:
        m = d["modulus"]

    if p_override is not None:
        p = p_override
    elif yaml_matches and "init_poisson" in e:
        p = e["init_poisson"]
    else:
        p = d["poisson"]

    return list(t), list(m), list(p)


def _common_env_fields(rl_cfg: dict, llm_cfg: dict, llm_disabled: bool,
                         args) -> dict:                      # 閳?v6: + args
    """Field dict shared by EnvConfig and SurrogateEnvConfig.

    v6: pavement_type resolved (CLI > yaml > 'semi_rigid'). Init layer
    defaults picked per type so flexible training does NOT crash guards.
    """
    e = rl_cfg.get("env", {})
    pt = _resolve_pavement_type(args, rl_cfg)
    init_t, init_m, init_p = _resolve_init_layers(rl_cfg, pt)
    logger.info(
        "Pavement type: {}  |  init_h={}  init_E={}  init_nu={}".format(
            pt, init_t, init_m, init_p))

    return dict(
        protocol_name=e.get("protocol_name", "JTG_D50_2017"),
        init_thickness_m=init_t,
        init_modulus_MPa=init_m,
        init_poisson=init_p,
        E_subgrade=e.get("E_subgrade", 60.0),
        nu_subgrade=e.get("nu_subgrade", 0.40),
        load_pressure_MPa=e.get("load_pressure_MPa", 0.7),
        load_radius_m=e.get("load_radius_m", 0.1065),
        city=e.get("city", "beijing"),
        climate_zone=e.get("climate_zone", ""),
        road_class=e.get("road_class", "expressway"),
        traffic_level=e.get("traffic_level", "heavy"),
        pavement_type=pt,                                    # 閳?v6: resolved
        design_life_years=e.get("design_life_years", 15),
        action_dh_max_m=e.get("action_dh_max_m", 0.02),
        action_dE_max_MPa=e.get("action_dE_max_MPa", 100.0),
        max_episode_steps=e.get("max_episode_steps", 20),
        max_episodes=e.get("max_episodes", 200),
        fea_keep_runs=e.get("fea_keep_runs", False),
        llm_enabled=(not llm_disabled),
        eval_alert_threshold=llm_cfg.get("evaluator", {}).get("alert_threshold", 2.0),
        strict_mode_steps=llm_cfg.get("strict_mode_steps", 30),
        log_every_n_steps=rl_cfg.get("logging", {}).get("log_every_n_steps", 10),
    )


def build_env_config(rl_cfg: dict, llm_cfg: dict, llm_disabled: bool,
                       args) -> EnvConfig:
    """
    Returns an EnvConfig or SurrogateEnvConfig depending on flags/yaml.

    Precedence: CLI flag --use-surrogate > yaml surrogate.use_surrogate.
    Similarly for model_path / fea_validation_every / b3_threshold.
    """
    common = _common_env_fields(rl_cfg, llm_cfg, llm_disabled, args)
    s = rl_cfg.get("surrogate", {})

    cli_on = bool(getattr(args, "use_surrogate", False))
    yaml_on = bool(s.get("use_surrogate", False))
    use_surrogate = cli_on or yaml_on

    # 閳光偓閳光偓 Plain mode (Phase 2BC behavior) 閳光偓閳光偓
    if not use_surrogate:
        logger.info("Surrogate mode: DISABLED (pure FEA per step)")
        return EnvConfig(**common)

    # 閳光偓閳光偓 Surrogate mode (Phase 2D) 閳光偓閳光偓
    cli_path = (args.surrogate_model_path or "").strip()
    model_path = cli_path or s.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        logger.error(
            "--use-surrogate requested but model file not found at: "
            f"{model_path!r}")
        logger.error("Run scripts/train_surrogate_v2.py first, or pass "
                      "--surrogate-model-path <path>.")
        sys.exit(1)

    fva = (args.fea_validation_every if args.fea_validation_every > 0
           else int(s.get("fea_validation_every", 10)))
    b3t = (args.surrogate_b3_threshold if args.surrogate_b3_threshold > 0
           else float(s.get("b3_threshold", 1.2)))

    logger.info(
        "Surrogate mode: ENABLED  model={}  validate-every={}  B3-escalate-below={}".format(
            model_path, fva, b3t))

    return SurrogateEnvConfig(
        **common,
        use_surrogate=True,
        surrogate_model_path=str(model_path),
        fea_validation_every=int(fva),
        surrogate_b3_threshold=float(b3t),
    )


def build_ppo_hyperparams(rl_cfg: dict, seed: int) -> PPOHyperparams:
    p = rl_cfg.get("ppo", {})
    return PPOHyperparams(
        learning_rate=p.get("learning_rate", 2.0e-3),
        gamma=p.get("gamma", 0.99),
        clip_range=p.get("clip_range", 0.2),
        n_epochs=p.get("n_epochs", 4),
        ent_coef=p.get("ent_coef", 0.01),
        vf_coef=p.get("vf_coef", 0.5),
        gae_lambda=p.get("gae_lambda", 0.95),
        max_grad_norm=p.get("max_grad_norm", 0.5),
        n_steps=p.get("n_steps", 64),
        batch_size=p.get("batch_size", 32),
        target_kl=p.get("target_kl", 0.05),
        hidden_dims=p.get("hidden_dims", [64, 64]),
        total_timesteps=rl_cfg.get("total_timesteps", 4000),
        seed=seed,
        device=p.get("device", "auto"),
    )


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# LLM stack 閳?unchanged from v5
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def assemble_llm_stack(llm_cfg: dict, audit_dir: Path, enable: bool,
                       gen_alpha_initial: Optional[float] = None,
                       gen_alpha_decay: Optional[str] = None,
                       gen_alpha_fallback: Optional[float] = None,
                       gen_backend: str = "deepseek",
                       gen_model: Optional[str] = None,
                       gen_use_reranker: bool = True):
    if not enable:
        logger.info("LLM stack disabled (--no-llm); pure-PPO mode.")
        return None, None, None, None

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "audit_chain.jsonl"
    try:
        audit = AuditChain(path=str(audit_path))
        logger.info(f"AuditChain ready: {audit_path}")
    except Exception as e:
        logger.error(f"AuditChain init failed ({e}); aborting LLM stack")
        return None, None, None, None

    rag = None
    try:
        rag_dir = llm_cfg.get("rag", {}).get("persist_dir", "./output/rag_db")
        rag = RAGStore(persist_dir=rag_dir)
        n = rag.count()
        if n == 0:
            logger.warning(
                "RAG empty (0 chunks). Generator runs without regulation context.")
        else:
            logger.info(f"RAG ready ({n} chunks)")
    except Exception as e:
        logger.warning(f"RAG init failed ({e}); proceeding without it")
        rag = None

    evaluator = None
    try:
        evaluator = Evaluator(audit=audit, backend="deepseek", fail_fast=False)
        logger.info("Evaluator ready (DeepSeek)")
    except Exception as e:
        logger.warning(f"Evaluator init failed ({e}); proceeding without it")

    generator = None
    try:
        gen_cfg = GeneratorConfig(backend=gen_backend)
        gen_cfg.model = gen_model
        gen_cfg.use_reranker = bool(gen_use_reranker)
        if gen_alpha_initial is not None:
            gen_cfg.alpha_initial = float(gen_alpha_initial)
        if gen_alpha_decay is not None:
            gen_cfg.alpha_decay = str(gen_alpha_decay)
        if gen_alpha_fallback is not None:
            gen_cfg.alpha_fallback_infeasible = float(gen_alpha_fallback)
        generator = Generator(config=gen_cfg, rag=rag, audit=audit, fail_fast=False)
        logger.info("Generator ready (%s/%s, reranker=%s, alpha_initial=%.3f)",
                    gen_backend, gen_model or "backend-default", gen_use_reranker,
                    gen_cfg.alpha_initial)
    except Exception as e:
        logger.warning(f"Generator init failed ({e}); proceeding without it")

    return evaluator, generator, rag, audit


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Checkpoint manager 閳?unchanged from v5
# (works with both PavementEnv and PavementEnvWithSurrogate;
#  subclass inherits _design / total_episodes / global_step attrs)
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

class IPDCheckpoint:
    @staticmethod
    def save(ckpt_dir: Path, model: PPO, env, total_timesteps_done: int,
             meta_extra: Optional[dict] = None):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt_dir / "ppo_model.zip"))

        if env._design:
            if "thickness_m" in env._design:
                design = {
                    "thickness_m": env._design["thickness_m"].tolist(),
                    "modulus_MPa": env._design["modulus_MPa"].tolist(),
                    "poisson":     env._design["poisson"].tolist(),
                }
            elif "thickness" in env._design:
                design = {
                    "thickness_m": env._design["thickness"].tolist(),
                    "modulus_MPa": env._design["modulus"].tolist(),
                    "poisson":     env._design["poisson"].tolist(),
                }
            else:
                design = {}
        else:
            design = {}

        env_state = {
            "total_episodes": int(env.total_episodes),
            "global_step":    int(env.global_step),
            "episode_step":   int(env.episode_step),
            "design":         design,
        }
        with open(ckpt_dir / "env_state.json", "w", encoding="utf-8") as f:
            json.dump(env_state, f, indent=2, ensure_ascii=False)

        meta = {
            "saved_at": dt.datetime.now().isoformat(),
            "total_timesteps_done": int(total_timesteps_done),
        }
        if meta_extra:
            meta.update(meta_extra)
        with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Checkpoint saved: {ckpt_dir} (ts={total_timesteps_done})")

    @staticmethod
    def load_env_state(ckpt_dir: Path, env):
        env_state_path = ckpt_dir / "env_state.json"
        if not env_state_path.exists():
            logger.warning(f"No env_state.json in {ckpt_dir}; skipping env restore")
            return
        with open(env_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        env.set_episode_counter(
            total_episodes=int(state.get("total_episodes", 0)),
            global_step=int(state.get("global_step", 0)),
        )
        logger.info(f"Env state restored: ep={env.total_episodes} gs={env.global_step}")

    @staticmethod
    def find_latest(runs_root: Path) -> Optional[Path]:
        if not runs_root.exists():
            return None
        candidates = []
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            ckpt_root = run_dir / "checkpoints"
            if not ckpt_root.exists():
                continue
            for c in ckpt_root.iterdir():
                if c.is_dir() and (c / "ppo_model.zip").exists():
                    candidates.append(c)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]


class _SaveOnInterval(BaseCallback):
    def __init__(self, save_freq, ckpt_root, env_ref, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.ckpt_root = ckpt_root
        self.env_ref = env_ref

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            ckpt_dir = self.ckpt_root / f"ckpt_step_{self.num_timesteps:06d}"
            IPDCheckpoint.save(ckpt_dir, self.model, self.env_ref, self.num_timesteps)
        return True


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Main
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def main():
    parser = argparse.ArgumentParser(description="iLLM-PD Phase 2BC + 2D training")
    parser.add_argument("--config",     type=str, default="config/rl_default.yaml")
    parser.add_argument("--llm-config", type=str, default="config/llm_config.yaml")
    parser.add_argument("--timesteps",  type=int, default=None)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--gen-backend", type=str, default="deepseek",
                        choices=["deepseek", "chatfire", "siliconflow-qwen", "siliconflow-glm", "ollama", "ollama-llama"],
                        help="Generator LLM backend. Use deepseek for first-submission consistency.")
    parser.add_argument("--gen-model", type=str, default=None,
                        help="Optional Generator model override, e.g. deepseek-chat.")
    parser.add_argument("--gen-no-reranker", action="store_true",
                        help="Disable LLM reranking of RAG passages during training; use similarity-ranked RAG only.")
    parser.add_argument("--gen-alpha-initial", type=float, default=None,
                        help="Override Generator alpha_initial for alpha sensitivity.")
    parser.add_argument("--gen-alpha-decay", type=str, default=None,
                        choices=["linear_to_zero", "cosine", "constant"],
                        help="Override Generator alpha decay schedule.")
    parser.add_argument("--gen-alpha-fallback", type=float, default=None,
                        help="Override infeasible-state alpha fallback; canonical value is 0.0.")
    parser.add_argument("--reward-profile", type=str, default="baseline",
                        choices=["baseline", "performance-up-12p5", "performance-up", "performance-up-37p5", "performance-up-50p0", "performance-down", "economy-up", "economy-down", "no-directional-reward"],
                        help="Reward-weight profile for R3.4 sensitivity analysis.")
    parser.add_argument("--run-name",   type=str, default=None)
    parser.add_argument("--save-freq",  type=int, default=50)
    # 閳光偓閳光偓 PHASE 2D dual-base (v6) 閳光偓閳光偓
    parser.add_argument("--pavement-type", type=str, default=None,
                        choices=["semi_rigid", "flexible",
                                 "unbound", "granular", "semirigid", "semi-rigid"],
                        help="Override env.pavement_type. CLI > yaml > 'semi_rigid'. "
                             "'unbound'/'granular' aliased to 'flexible'.")
    # 閳光偓閳光偓 PHASE 2D 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    parser.add_argument("--use-surrogate", action="store_true",
                        help="Enable surrogate routing (Phase 2D).")
    parser.add_argument("--surrogate-model-path", type=str, default="",
                        help="Path to surrogate .pt file. Overrides yaml.")
    parser.add_argument("--fea-validation-every", type=int, default=-1,
                        help="Run real FEA every N global steps (default: read yaml, then 10).")
    parser.add_argument("--surrogate-b3-threshold", type=float, default=-1.0,
                        help="Surrogate-predicted B3 below this 閳?escalate to real FEA "
                             "(default: read yaml, then 1.2).")
    # 閳光偓閳光偓 PHASE 2F ablation 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    parser.add_argument("--ablation-variant", type=str, default="full",
                        choices=["full", "no-generator", "no-evaluator",
                                 "no-rag", "no-generator-no-rag",
                                 "no-language-no-guard",
                                 "no-guard", "reward-only"],
                        help="Ablation variant: full / no-generator / no-evaluator "
                             "/ no-rag / no-generator-no-rag / no-language-no-guard "
                             "/ no-guard / reward-only.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _apply_sb3_monkey_patch()
    load_dotenv()

    rl_cfg = load_yaml(Path(args.config))
    llm_cfg = load_yaml(Path(args.llm_config)) if Path(args.llm_config).exists() else {}

    runs_root = Path(rl_cfg.get("io", {}).get("runs_root", "./output/rl_runs"))

    resume_ckpt: Optional[Path] = None
    if args.resume:
        if args.resume == "latest":
            resume_ckpt = IPDCheckpoint.find_latest(runs_root)
            if resume_ckpt is None:
                logger.error(f"No previous checkpoint found in {runs_root}; aborting.")
                sys.exit(1)
        else:
            resume_ckpt = Path(args.resume)
            if not (resume_ckpt / "ppo_model.zip").exists():
                logger.error(f"Invalid checkpoint path: {resume_ckpt}")
                sys.exit(1)
        logger.info(f"Resuming from: {resume_ckpt}")
        run_dir = resume_ckpt.parent.parent
    else:
        if args.run_name:
            run_name = args.run_name
        else:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            protocol = rl_cfg.get("env", {}).get("protocol_name", "JTG").lower().replace("_", "")
            tag = "_surrogate" if args.use_surrogate else ""
            run_name = f"ppo_{protocol}_seed{args.seed}{tag}_{ts}"
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = run_dir / "audit"
    fea_output_dir = run_dir / "fea_runs"
    ckpt_root = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    for d in [audit_dir, fea_output_dir, ckpt_root, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 70)

    # LLM stack (with ablation override)
    llm_enable = (not args.no_llm)
    variant = args.ablation_variant
    logger.info(f"Ablation variant: {variant}")
    evaluator, generator, rag, audit = assemble_llm_stack(
        llm_cfg, audit_dir, enable=llm_enable,
        gen_alpha_initial=args.gen_alpha_initial,
        gen_alpha_decay=args.gen_alpha_decay,
        gen_alpha_fallback=args.gen_alpha_fallback,
        gen_backend=args.gen_backend,
        gen_model=args.gen_model,
        gen_use_reranker=(not args.gen_no_reranker))

    # Ablation overrides
    if variant in ("no-generator", "no-generator-no-rag", "no-language-no-guard"):
        generator = None
        rag = None          # RAG is Generator-internal; disable too
        if variant == "no-language-no-guard":
            logger.info("  -> Generator and RAG disabled; Guard disabled downstream")
        elif variant == "no-generator-no-rag":
            logger.info("  -> Generator and RAG disabled; Evaluator + Guard retained")
        else:
            logger.info("  -> Generator disabled; Evaluator + Guard retained")
    elif variant == "no-evaluator":
        evaluator = None
        logger.info("  -> Evaluator disabled; Generator + RAG + Guard retained")
    elif variant == "no-rag":
        rag = None          # Generator runs without spec context
        if generator is not None:
            generator.rag = None                # disconnect RAG on already-built Generator
            try:
                generator.config.use_rag = False  # double-lock via config flag
            except Exception:
                pass
        logger.info("  -> RAG disabled on Generator (rag=None + use_rag=False)")
    elif variant == "reward-only":
        audit = None        # No hard-constraint guard 閳?FEA penalty only
        logger.info("  -> Reward-only constraint; NumericalGuard bypassed")

    # Env config
    env_cfg = build_env_config(rl_cfg, llm_cfg, llm_disabled=args.no_llm, args=args)
    resolved_pavement_type = str(env_cfg.pavement_type)
    env_cfg.evaluator = evaluator
    env_cfg.generator = generator
    env_cfg.audit_chain = audit
    if variant in ("no-guard", "no-language-no-guard"):
        env_cfg.guard_enabled = False
        logger.info("  -> guard_enabled=False (R3-11 ablation cell)")
    if variant == "reward-only":
        env_cfg.guard_enabled = False

    # 閳光偓閳光偓 PHASE 2D: pick env class to instantiate 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    if isinstance(env_cfg, SurrogateEnvConfig) and env_cfg.use_surrogate:
        env = PavementEnvWithSurrogate(env_cfg)
        logger.info("Env class: PavementEnvWithSurrogate (Phase 2D routing active)")
    else:
        env = PavementEnv(env_cfg)
        logger.info("Env class: PavementEnv (Phase 2BC, pure FEA)")

    if args.reward_profile != "baseline":
        env.reward_fn = CompositeReward(build_reward_config(args.reward_profile))
        logger.info("Reward profile override: %s", args.reward_profile)
    else:
        logger.info("Reward profile: baseline")

    env.set_fea_output_dir(fea_output_dir)

    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "run_name": run_dir.name,
        "seed": args.seed,
        "target_timesteps": args.timesteps or rl_cfg.get("total_timesteps", 4000),
        "pavement_type": resolved_pavement_type,
        "generator": {
            "enabled": generator is not None,
            "backend": args.gen_backend if generator is not None else None,
            "model": (generator.config.model or getattr(generator._client, "default_model", None)) if generator is not None else None,
            "alpha_initial": generator.config.alpha_initial if generator is not None else 0.0,
            "alpha_decay": generator.config.alpha_decay if generator is not None else None,
            "alpha_fallback_infeasible": generator.config.alpha_fallback_infeasible if generator is not None else 0.0,
            "reranker": bool(generator.config.use_reranker) if generator is not None else False,
        },
        "reward_profile": args.reward_profile,
        "surrogate_b3_threshold": env_cfg.surrogate_b3_threshold if hasattr(env_cfg, "surrogate_b3_threshold") else None,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Run manifest written: %s", run_dir / "run_manifest.json")

    env_monitor = Monitor(env, filename=str(run_dir / "monitor.csv"))

    # PPO
    if resume_ckpt:
        logger.info("Loading PPO model from checkpoint...")
        model = PPO.load(
            str(resume_ckpt / "ppo_model.zip"),
            env=env_monitor,
            tensorboard_log=str(tb_dir))
        IPDCheckpoint.load_env_state(resume_ckpt, env)
    else:
        hp = build_ppo_hyperparams(rl_cfg, seed=args.seed)
        model = build_ppo(env_monitor, hp, tb_log_dir=str(tb_dir))
        logger.info("New PPO model created.")

    target_ts = args.timesteps or rl_cfg.get("total_timesteps", 4000)
    current_ts = model.num_timesteps
    increment = max(0, target_ts - current_ts)
    logger.info(
        f"Target total timesteps: {target_ts} (current: {current_ts}, will train +{increment})")

    if increment == 0:
        logger.info("Already reached target.")
        sys.exit(0)

    ckpt_cb = _SaveOnInterval(args.save_freq, ckpt_root, env)
    margins_cb = SpecMarginsLoggingCallback()

    def _emergency_save(sig=None, frame=None):
        logger.warning("Caught interrupt 閳?saving emergency checkpoint...")
        emergency_dir = ckpt_root / f"emergency_step_{model.num_timesteps:06d}"
        try:
            IPDCheckpoint.save(emergency_dir, model, env, model.num_timesteps,
                               meta_extra={"emergency": True})
            logger.warning(f"Emergency save complete: {emergency_dir}")
        except Exception as e:
            logger.error(f"Emergency save FAILED: {e}")
        finally:
            try: env.close()
            except: pass
            try:
                if evaluator: evaluator.close()
            except: pass
            sys.exit(130)
    signal.signal(signal.SIGINT, _emergency_save)

    # Train
    try:
        model.learn(
            total_timesteps=increment,
            callback=[ckpt_cb, margins_cb],
            reset_num_timesteps=False,
            progress_bar=True)
    except Exception as e:
        logger.error(f"Training crashed: {e}", exc_info=True)
        _emergency_save()
        raise

    # Final save
    final_dir = ckpt_root / f"ckpt_final_step_{model.num_timesteps:06d}"
    IPDCheckpoint.save(final_dir, model, env, model.num_timesteps,
                       meta_extra={"final": True})

    if audit is not None:
        try:
            audit.close()
        except Exception:
            pass

    # 閳光偓閳光偓 PHASE 2D: dump surrogate stats if backend was used 閳光偓閳光偓
    if hasattr(env, "backend_stats"):
        stats = env.backend_stats
        if stats is not None:
            logger.info("=" * 70)
            logger.info("Surrogate routing stats (this run):")
            for k, v in stats.items():
                logger.info(f"  {k:<40} = {v}")
            stats_path = run_dir / "surrogate_stats.json"
            try:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                logger.info(f"Surrogate stats saved: {stats_path}")
            except Exception as e:
                logger.warning(f"Failed to save surrogate stats: {e}")

    # 閳光偓閳光偓 Write completion flag (for ablation batch resume) 閳光偓閳光偓閳光偓閳光偓閳光偓
    try:
        flag_path = run_dir / "training_complete.flag"
        variant_str = getattr(args, "ablation_variant", "full") or "full"
        flag_content = (
            f"training_complete\n"
            f"timestamp: {dt.datetime.now().isoformat()}\n"
            f"timesteps: {model.num_timesteps}\n"
            f"seed: {args.seed}\n"
            f"variant: {variant_str}\n"
            f"pavement_type: {resolved_pavement_type}\n"
        )
        flag_path.write_text(flag_content)
        logger.info(f"Completion flag written: {flag_path}")
    except Exception as e:
        logger.warning(f"Failed to write completion flag: {e}")
        try:
            with open(str(run_dir / "training_complete.flag"), "w") as f:
                f.write("training_complete\n")
            logger.info("Minimal completion flag written via fallback")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")

    env.close()
    logger.info("=" * 70)
    logger.info(f"Training complete. Final timesteps: {model.num_timesteps}")
    logger.info(f"Final checkpoint: {final_dir}")
    if args.use_surrogate:
        logger.info(f"To continue: python -m rl.train --resume latest --timesteps <larger> --use-surrogate")
    else:
        logger.info(f"To continue: python -m rl.train --resume latest --timesteps <larger>")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()






