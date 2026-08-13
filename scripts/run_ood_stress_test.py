# -*- coding: utf-8 -*-
"""
scripts/run_ood_stress_test.py — R2-3 / R3-8 OOD Stress Test (v2, HONEST SCOPE)
===============================================================================

Tests HARA behaviour on out-of-distribution inputs relative to the surrogate-v3
LHS training prior (E_sub in [40, 200] MPa, semi-rigid/flexible base).

=========================  WHY v3 IS DIFFERENT  ============================
v1 declared 8 scenarios, but older EnvConfig/JTG paths only consumed
categorical city and traffic_level. Climate and traffic OOD rows could
therefore collapse to duplicate default runs.

v3 keeps the 4 historical core scenarios as the default Fig. 7 scope, and
adds opt-in climate/traffic scenarios through real continuous protocol inputs:
    - MAAT_C interpolates the verified JTG city temperature table
    - annual_ESAL_BZZ100 / total_ESAL_BZZ100 override cumulative N_e
Use --include-pending to run the full 8-scenario matrix.
============================================================================

Modes:
    surrogate_only      escalation disabled (B3 threshold = -1.0). Safe during ablation.
    escalation_enabled  B3 < threshold triggers ABAQUS. Use after ablation finishes.

Usage:
    python scripts/run_ood_stress_test.py --mode surrogate_only
    python scripts/run_ood_stress_test.py --mode escalation_enabled --seeds 3
    python scripts/run_ood_stress_test.py --mode both          # comparative
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("ood_stress")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

POLICY_FLEX = Path("output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/checkpoints/ckpt_final_step_002048/ppo_model.zip")
POLICY_SEMI = Path("output/rl_runs/ppo_semi_rigid_v3_1000ts_seed0/checkpoints/ckpt_final_step_001024/ppo_model.zip")
SURROGATE_MODEL = "output/surrogate_model/surrogate_v3.pt"

MAX_STEPS = 20
SEEDS = [0, 1, 2]

# Standard in-distribution initial designs
_INIT_5L_FLEX  = [0.04, 0.06, 0.08, 0.30, 0.25]
_INIT_5L_SEMI  = [0.04, 0.06, 0.08, 0.36, 0.18]
_INIT_MOD_FLEX = [14000.0, 11000.0, 9000.0, 350.0,  250.0]
_INIT_MOD_SEMI = [14000.0, 11000.0, 9000.0, 1500.0, 400.0]
_INIT_POISSON  = [0.25, 0.30, 0.30, 0.35, 0.35]


@dataclass
class OODScenario:
    case_id: str
    category: str               # subgrade / material  (env-consumed)
    description: str
    pavement_type: str
    init_thickness_m: List[float]
    init_modulus_MPa: List[float]
    init_poisson: List[float]
    E_subgrade: float
    MAAT_C: Optional[float] = None
    annual_ESAL_BZZ100: Optional[float] = None
    total_ESAL_BZZ100: Optional[float] = None
    traffic_growth_rate: float = 0.0
    nu_subgrade: float = 0.40
    ood_axes: Dict[str, str] = field(default_factory=dict)
    expected_failure_mode: str = ""


def build_real_scenarios() -> List[OODScenario]:
    """The 4 scenarios that vary a parameter the current env actually consumes."""
    s: List[OODScenario] = []

    s.append(OODScenario(
        case_id="sg_very_soft", category="subgrade",
        description="Very soft subgrade (E_sub=25 MPa, below LHS [40,200])",
        pavement_type="semi_rigid",
        init_thickness_m=list(_INIT_5L_SEMI), init_modulus_MPa=list(_INIT_MOD_SEMI),
        init_poisson=list(_INIT_POISSON), E_subgrade=25.0,
        ood_axes={"E_subgrade": "below_training_range"},
        expected_failure_mode="B4 subgrade strain critical"))

    s.append(OODScenario(
        case_id="sg_very_stiff", category="subgrade",
        description="Very stiff subgrade (E_sub=600 MPa, above LHS)",
        pavement_type="semi_rigid",
        init_thickness_m=list(_INIT_5L_SEMI), init_modulus_MPa=list(_INIT_MOD_SEMI),
        init_poisson=list(_INIT_POISSON), E_subgrade=600.0,
        ood_axes={"E_subgrade": "above_training_range"},
        expected_failure_mode="safe_but_suboptimal (low reward)"))

    s.append(OODScenario(
        case_id="sg_ltpp_48_0001", category="subgrade",
        description="LTPP 48_0001 natural OOD (E_sub=700 MPa, FWD backcalc anomaly)",
        pavement_type="flexible",
        init_thickness_m=list(_INIT_5L_FLEX), init_modulus_MPa=list(_INIT_MOD_FLEX),
        init_poisson=list(_INIT_POISSON), E_subgrade=700.0,
        ood_axes={"E_subgrade": "extreme_above_training"},
        expected_failure_mode="safe_but_suboptimal (prior mean reward ~0.12)"))

    s.append(OODScenario(
        case_id="mat_soft_base", category="material",
        description="Soft base modulus (E_base=130 MPa, below NumericalGuard 150)",
        pavement_type="flexible",
        init_thickness_m=list(_INIT_5L_FLEX),
        init_modulus_MPa=[14000.0, 11000.0, 9000.0, 130.0, 250.0],   # L4 below guard
        init_poisson=list(_INIT_POISSON), E_subgrade=80.0,
        ood_axes={"E_base": "below_guard_threshold"},
        expected_failure_mode="GUARD_CLAMP (NumericalGuard intercepts)"))

    return s



def build_extended_scenarios() -> List[OODScenario]:
    """Opt-in climate/traffic OOD axes now wired through protocol extras."""
    s: List[OODScenario] = []

    s.append(OODScenario(
        case_id="tr_super_heavy", category="traffic",
        description="Super-heavy traffic (annual BZZ-100 ESAL = 1.0e7)",
        pavement_type="semi_rigid",
        init_thickness_m=list(_INIT_5L_SEMI), init_modulus_MPa=list(_INIT_MOD_SEMI),
        init_poisson=list(_INIT_POISSON), E_subgrade=80.0,
        annual_ESAL_BZZ100=1.0e7,
        ood_axes={"annual_ESAL_BZZ100": "above_heavy_baseline"},
        expected_failure_mode="traffic demand increases N_e in JTG checks"))

    s.append(OODScenario(
        case_id="tr_ultra_light", category="traffic",
        description="Ultra-light traffic (annual BZZ-100 ESAL = 1.0e5)",
        pavement_type="flexible",
        init_thickness_m=list(_INIT_5L_FLEX), init_modulus_MPa=list(_INIT_MOD_FLEX),
        init_poisson=list(_INIT_POISSON), E_subgrade=80.0,
        annual_ESAL_BZZ100=1.0e5,
        ood_axes={"annual_ESAL_BZZ100": "below_heavy_baseline"},
        expected_failure_mode="low traffic demand; reward mainly cost/safety balance"))

    s.append(OODScenario(
        case_id="cl_extreme_hot", category="climate",
        description="Extreme-hot climate (MAAT=28 C)",
        pavement_type="semi_rigid",
        init_thickness_m=list(_INIT_5L_SEMI), init_modulus_MPa=list(_INIT_MOD_SEMI),
        init_poisson=list(_INIT_POISSON), E_subgrade=80.0,
        MAAT_C=28.0,
        ood_axes={"MAAT_C": "above_city_table_range"},
        expected_failure_mode="hot climate raises kT/T_pef and rutting demand"))

    s.append(OODScenario(
        case_id="cl_extreme_cold", category="climate",
        description="Extreme-cold climate (MAAT=-5 C)",
        pavement_type="semi_rigid",
        init_thickness_m=list(_INIT_5L_SEMI), init_modulus_MPa=list(_INIT_MOD_SEMI),
        init_poisson=list(_INIT_POISSON), E_subgrade=80.0,
        MAAT_C=-5.0,
        ood_axes={"MAAT_C": "below_city_table_range"},
        expected_failure_mode="cold climate lowers temperature factors"))

    return s
# Documented but NOT run by default — require spec-layer extension to be real.
PENDING_AXES_NOTE = (
    "Climate-OOD and traffic-OOD are now protocol-wired through MAAT_C and "
    "annual_ESAL_BZZ100. They remain opt-in so legacy Fig. 7 runs keep the "
    "4-scenario env-consumed scope unless --include-pending is specified."
)

# ============================================================================
# Env construction
# ============================================================================

def make_env(sc: OODScenario, mode: str):
    from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig

    b3 = -1.0 if mode == "surrogate_only" else 1.0
    cfg = SurrogateEnvConfig(
        pavement_type=sc.pavement_type,
        init_thickness_m=sc.init_thickness_m,
        init_modulus_MPa=sc.init_modulus_MPa,
        init_poisson=sc.init_poisson,
        E_subgrade=sc.E_subgrade, nu_subgrade=sc.nu_subgrade,
        MAAT_C=sc.MAAT_C, annual_ESAL_BZZ100=sc.annual_ESAL_BZZ100,
        total_ESAL_BZZ100=sc.total_ESAL_BZZ100,
        traffic_growth_rate=sc.traffic_growth_rate,
        city="", climate_zone="temperate",
        road_class="expressway", traffic_level="heavy",
        design_life_years=15, max_episode_steps=MAX_STEPS, max_episodes=1,
        llm_enabled=False, fea_keep_runs=False, fea_verbose=False,
        enable_lcc_eval=True, design_life_years_lcc=20.0,
        use_surrogate=True, surrogate_model_path=SURROGATE_MODEL,
        surrogate_b3_threshold=b3,
    )
    env = PavementEnvWithSurrogate(cfg)
    try:
        env.set_fea_output_dir(Path(PROJECT_ROOT))
    except AttributeError:
        pass
    return env


# ============================================================================
# Per-scenario run
# ============================================================================

def run_one(sc: OODScenario, policy, mode: str, seed: int) -> dict:
    from rl import metrics as M

    env = make_env(sc, mode)
    t0 = time.time()
    obs, info = env.reset(seed=seed)

    margins_hist: List[Dict[str, float]] = []
    rewards: List[float] = []
    guard_violations = 0
    trajectory = []

    for step in range(MAX_STEPS):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        m = {k: float(v) for k, v in info.get("margins", {}).items()}
        margins_hist.append(m)
        rewards.append(float(reward))

        # guard violations: best-effort (depends on env.step exposing a flag)
        if info.get("guard_violation") or info.get("guard_clamped"):
            guard_violations += 1
        gv = info.get("n_guard_violations")
        if isinstance(gv, (int, float)):
            guard_violations = int(gv)

        thk_cm = info.get("design_h_cm")
        trajectory.append({
            "step": step, "reward": float(reward),
            "B1": m.get("B1_asphalt_fatigue"),
            "B3": m.get("B3_ac_permanent_deformation"),
            "B4": m.get("B4_subgrade_strain"),
            "feasible": info.get("feasible"),
            "thickness_cm": thk_cm,
        })
        if terminated or truncated:
            break

    final_margins = margins_hist[-1] if margins_hist else {}
    dsr = M.compute_dsr(final_margins)
    final_compliant = M.compute_compliance(final_margins)
    scr_final = (1.0 if final_compliant
                 else (sum(1 for v in final_margins.values() if v >= 1.0)
                       / max(len(final_margins), 1)))
    scr_traj = (sum(1 for mm in margins_hist if M.compute_compliance(mm))
                / max(len(margins_hist), 1))

    # Escalation / surrogate routing — read from backend (reliable)
    n_escal = n_surr = 0
    surr_frac = None
    try:
        bs = env.backend_stats
        if bs:
            n_escal = int(bs.get("n_fea_escalation", 0))
            n_surr = int(bs.get("n_surrogate_calls", 0))
            surr_frac = bs.get("surrogate_fraction")
    except Exception:
        pass
    n_steps = len(rewards)
    escal_rate = (n_escal / max(n_steps, 1)) if mode == "escalation_enabled" else 0.0

    try:
        env.close()
    except Exception:
        pass

    return {
        "case_id": sc.case_id, "category": sc.category,
        "pavement_type": sc.pavement_type, "mode": mode, "seed": seed,
        "E_subgrade": sc.E_subgrade,
        "MAAT_C": sc.MAAT_C,
        "annual_ESAL_BZZ100": sc.annual_ESAL_BZZ100,
        "total_ESAL_BZZ100": sc.total_ESAL_BZZ100,
        "n_steps": n_steps,
        "mean_reward": round(float(np.mean(rewards)) if rewards else 0.0, 4),
        "min_reward": round(float(np.min(rewards)) if rewards else 0.0, 4),
        "final_dsr": round(dsr, 4),
        "final_scr": round(scr_final, 4),
        "scr_trajectory": round(scr_traj, 4),
        "B1": round(final_margins.get("B1_asphalt_fatigue", float("nan")), 3),
        "B3": round(final_margins.get("B3_ac_permanent_deformation", float("nan")), 3),
        "B4": round(final_margins.get("B4_subgrade_strain", float("nan")), 3),
        "n_guard_violations": guard_violations,
        "n_fea_escalations": n_escal,
        "n_surrogate_calls": n_surr,
        "surrogate_fraction": (round(surr_frac, 3) if surr_frac is not None else None),
        "escalation_rate": round(escal_rate, 3),
        "wall_clock_sec": round(time.time() - t0, 1),
        "trajectory": trajectory,
        "status": "ok",
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["surrogate_only", "escalation_enabled", "both"],
                        default="surrogate_only")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--include-pending", action="store_true",
                        help="Also run protocol-wired climate/traffic scenarios "
                             "using MAAT_C and annual_ESAL_BZZ100.")
    args = parser.parse_args()

    global SEEDS
    SEEDS = list(range(args.seeds))
    modes = (["surrogate_only", "escalation_enabled"] if args.mode == "both"
             else [args.mode])
    if "escalation_enabled" in modes:
        logger.warning("escalation_enabled uses ABAQUS — do NOT run during the ablation campaign.")

    scenarios = build_real_scenarios()
    if args.include_pending:
        scenarios.extend(build_extended_scenarios())
    logger.info(f"Running {len(scenarios)} OOD scenarios "
                f"({'4 core + 4 climate/traffic' if args.include_pending else '4 core'}).")
    if not args.include_pending:
        logger.info(f"Opt-in axes not run: {PENDING_AXES_NOTE}")

    for p in (POLICY_FLEX, POLICY_SEMI):
        if not p.exists():
            logger.error(f"Missing policy: {p}")
            return
    from stable_baselines3 import PPO
    logger.info("Loading PPO policies...")
    policy_flex = PPO.load(POLICY_FLEX)
    policy_semi = PPO.load(POLICY_SEMI)

    results: List[dict] = []
    total = len(scenarios) * len(SEEDS) * len(modes)
    idx = 0
    for sc in scenarios:
        policy = policy_flex if sc.pavement_type == "flexible" else policy_semi
        for mode in modes:
            for seed in SEEDS:
                idx += 1
                logger.info(f"[{idx}/{total}] {sc.case_id} | {mode} | seed={seed} "
                            f"| E_sub={sc.E_subgrade:.0f}")
                try:
                    r = run_one(sc, policy, mode, seed)
                    results.append(r)
                    logger.info(f"   DSR={r['final_dsr']:.2f} SCR={r['final_scr']:.2f} "
                                f"reward={r['mean_reward']:.3f} "
                                f"escal={r['n_fea_escalations']} "
                                f"guards={r['n_guard_violations']} "
                                f"({r['wall_clock_sec']:.0f}s)")
                except Exception as e:
                    logger.exception(f"   FAILED: {e}")
                    results.append({"case_id": sc.case_id, "mode": mode,
                                    "seed": seed, "status": f"error: {e}"})

    out_dir = Path("experiments/ltpp_data/deliverables/ood_stress")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    cols = ["case_id", "category", "pavement_type", "mode", "seed", "E_subgrade",
            "MAAT_C", "annual_ESAL_BZZ100", "total_ESAL_BZZ100",
            "n_steps", "mean_reward", "min_reward",
            "final_dsr", "final_scr", "scr_trajectory", "B1", "B3", "B4",
            "n_guard_violations", "n_fea_escalations", "n_surrogate_calls",
            "surrogate_fraction", "escalation_rate", "wall_clock_sec", "status"]
    per_run = out_dir / f"ood_per_run_{ts}.csv"
    with open(per_run, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    logger.info(f"Per-run CSV: {per_run}")

    # Aggregate (pandas optional)
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        try:
            import pandas as pd
            df = pd.DataFrame(ok)
            agg = df.groupby(["case_id", "category", "pavement_type", "mode"]).agg(
                n=("seed", "count"),
                MAAT_C=("MAAT_C", "first"),
                annual_ESAL_BZZ100=("annual_ESAL_BZZ100", "first"),
                total_ESAL_BZZ100=("total_ESAL_BZZ100", "first"),
                mean_reward=("mean_reward", "mean"),
                final_dsr=("final_dsr", "mean"),
                final_scr=("final_scr", "mean"),
                scr_traj=("scr_trajectory", "mean"),
                B3_min=("B3", "min"),
                escalation_rate=("escalation_rate", "mean"),
                guards_total=("n_guard_violations", "sum"),
            ).round(3).reset_index()
            agg.to_csv(out_dir / f"ood_aggregate_{ts}.csv", index=False)
            print("\n" + "=" * 110)
            print(f"OOD STRESS TEST SUMMARY ({len(scenarios)} scenarios)")
            print("=" * 110)
            print(f"{'case':<18}{'type':<11}{'mode':<20}{'DSR':>5}{'SCR':>6}"
                  f"{'reward':>8}{'B3min':>7}{'escal%':>8}{'guards':>8}")
            print("-" * 110)
            for _, r in agg.iterrows():
                print(f"{r['case_id']:<18}{r['pavement_type']:<11}{r['mode']:<20}"
                      f"{r['final_dsr']:>5.2f}{r['final_scr']:>6.2f}"
                      f"{r['mean_reward']:>8.3f}{r['B3_min']:>7.2f}"
                      f"{r['escalation_rate']*100:>7.1f}%{int(r['guards_total']):>8}")
            print("=" * 110)
        except ImportError:
            logger.info("pandas not available; per-run CSV written, skipping aggregate table.")

    json_path = out_dir / f"ood_trajectories_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Trajectory JSON: {json_path}")


if __name__ == "__main__":
    main()
