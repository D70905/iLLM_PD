from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NCAT as-built versus frozen-policy iLLM-PD design benchmark"
    )
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--cases",
        type=Path,
        default=Path("experiments/ncat_data/ncat_cases.json"),
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=Path(
            "output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/"
            "checkpoints/ckpt_final_step_002048"
        ),
    )
    p.add_argument("--sections", default="N1,N8,S13")
    p.add_argument("--seeds", default="0")
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/ncat_design_benchmark"))
    p.add_argument("--protocol", default="JTG_D50_2017")
    p.add_argument("--design-temperature-C", type=float, default=20.0)
    p.add_argument("--base-price-cny-m3", type=float, default=100.0)
    p.add_argument("--use-surrogate", action="store_true")
    p.add_argument(
        "--surrogate-model",
        type=Path,
        default=Path("output/surrogate_model/surrogate_v3.pt"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_sections(raw: str) -> set[str]:
    return {x.strip().upper().replace("NCAT_CG_", "") for x in raw.split(",") if x.strip()}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def split_equivalent_base(case: Dict[str, Any]) -> Dict[str, List[float]]:
    """Represent NCAT's one aggregate base as two identical contiguous FE sublayers."""
    ec = case["envconfig"]
    h = list(map(float, ec["init_thickness_m"]))
    e = list(map(float, ec["init_modulus_MPa"]))
    nu = list(map(float, ec["init_poisson"]))
    if len(h) != 5 or len(e) != 5 or len(nu) != 5:
        raise ValueError(f"{case['section_id']}: expected five-element source vectors")
    aggregate_h = h[3]
    aggregate_e = e[3]
    aggregate_nu = nu[3]
    if aggregate_h <= 0:
        raise ValueError(f"{case['section_id']}: aggregate-base thickness must be positive")
    half = aggregate_h / 2.0
    return {
        "thickness": [h[0], h[1], h[2], half, half],
        "modulus": [e[0], e[1], e[2], aggregate_e, aggregate_e],
        "poisson": [nu[0], nu[1], nu[2], aggregate_nu, aggregate_nu],
    }


def project_action(action: Iterable[float]) -> np.ndarray:
    """Freeze material moduli and tie the two equivalent aggregate sublayers."""
    a = np.asarray(list(action), dtype=np.float32).reshape(-1)
    if a.shape != (10,):
        raise ValueError(f"expected 10-dimensional action, got {a.shape}")
    a = np.clip(a, -1.0, 1.0)
    # One physical aggregate-base variable represented by two equal sublayers.
    aggregate = float((a[3] + a[4]) / 2.0)
    a[3] = aggregate
    a[4] = aggregate
    # The NCAT benchmark compares thickness alternatives using the measured materials.
    a[5:10] = 0.0
    return a


def design_cost(env, design: Dict[str, Iterable[float]]) -> float:
    return float(
        env.reward_fn._material_cost(
            np.asarray(design["thickness"], dtype=float),
            np.asarray(design["modulus"], dtype=float),
            pavement_type="flexible",
        )
    )


def design_from_info(info: Dict[str, Any], fallback: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    delivered = info.get("delivered_design")
    if delivered is None:
        delivered = fallback
    return {
        "thickness": [float(x) for x in delivered["thickness"]],
        "modulus": [float(x) for x in delivered["modulus"]],
    }


def jtg_result(evaluation, dsr_fn) -> Dict[str, Any]:
    margins = {k: float(v) for k, v in evaluation.margins.items()}
    return {
        "feasible": bool(evaluation.feasible),
        "dsr": float(dsr_fn(margins)),
        "critical": str(evaluation.critical_indicator),
        "margins": margins,
    }


def mepdg_rut_result(
    responses: Dict[str, Any],
    design: Dict[str, List[float]],
    n_esal: float,
    temp_c: float,
    mepdg_rutting_mm,
) -> Dict[str, float]:
    h = design["thickness"]
    e = design["modulus"]
    h_ac = sum(h[:3])
    e_ac = sum(x * y for x, y in zip(h[:3], e[:3])) / h_ac
    eps_z = float(responses["epsilon_z_microstrain"])
    eps_hma = responses.get("eps_AC_mid_mid_microstrain")
    if eps_hma is None:
        eps_hma = abs(float(responses["p_AC_mid_mid_MPa"])) / e_ac * 1.0e6
    rh, rb, rs, _ = mepdg_rutting_mm(
        float(eps_hma),
        eps_z,
        h_ac * 1000.0,
        (h[3] + h[4]) * 1000.0,
        float(n_esal),
        temp_c * 9.0 / 5.0 + 32.0,
        beta_r1=1.0,
        beta_s1_gran=1.0,
        beta_s1_subg=1.0,
    )
    return {
        "eps_hma_microstrain": float(eps_hma),
        "eps_z_microstrain": eps_z,
        "rut_hma_mm": float(rh),
        "rut_base_mm": float(rb),
        "rut_subgrade_mm": float(rs),
        "rut_total_mm": float(rh + rb + rs),
    }


def build_env_class():
    from rl.env_surrogate import PavementEnvWithSurrogate
    from rl.guards import GuardConfig, NumericalGuard

    class NcatEquivalentBaseEnv(PavementEnvWithSurrogate):
        def __init__(self, config, base_price: float):
            super().__init__(config)
            self.guard = NumericalGuard(
                GuardConfig(
                    h_min=[0.02, 0.03, 0.04, 0.04, 0.04],
                    h_max=[0.10, 0.15, 0.25, 0.25, 0.25],
                    E_min=[4000.0, 3000.0, 2000.0, 100.0, 100.0],
                    E_max=[25000.0, 18000.0, 15000.0, 500.0, 500.0],
                    base_type="flexible",
                )
            )
            rcfg = self.reward_fn.config
            rcfg.material_prices_flexible[3] = float(base_price)
            rcfg.material_prices_flexible[4] = float(base_price)
            rcfg.modulus_price_coeffs_flexible[3] = 0.0
            rcfg.modulus_price_coeffs_flexible[4] = 0.0

        def _maybe_generator(self, action_ppo):
            used, result = super()._maybe_generator(action_ppo)
            return project_action(used), result

    return NcatEquivalentBaseEnv


def build_config(args: argparse.Namespace, case: Dict[str, Any], design: Dict[str, List[float]]):
    from rl.env_surrogate import SurrogateEnvConfig

    root = args.project_root
    return SurrogateEnvConfig(
        protocol_name=args.protocol,
        init_thickness_m=list(design["thickness"]),
        init_modulus_MPa=list(design["modulus"]),
        init_poisson=list(design["poisson"]),
        E_subgrade=float(case["envconfig"]["E_subgrade"]),
        nu_subgrade=float(case["envconfig"]["nu_subgrade"]),
        load_pressure_MPa=0.7,
        load_radius_m=0.1065,
        city="",
        climate_zone="warm",
        road_class="expressway",
        traffic_level="heavy",
        pavement_type="flexible",
        design_life_years=15,
        max_episode_steps=args.max_steps,
        max_episodes=1,
        fea_base_dir=str(root),
        fea_num_cpus=4,
        fea_verbose=False,
        fea_keep_runs=False,
        llm_enabled=False,
        enable_lcc_eval=True,
        design_life_years_lcc=20.0,
        climate_enabled=False,
        design_temperature_C=None,
        use_surrogate=bool(args.use_surrogate),
        surrogate_model_path=str(resolve(root, args.surrogate_model)),
        fea_validation_every=10,
        surrogate_b3_threshold=1.0,
    )


def run_case(args, case, seed, policy, env_class, dsr_fn, mepdg_rutting_mm) -> Dict[str, Any]:
    eq = split_equivalent_base(case)
    cfg = build_config(args, case, eq)
    env = env_class(cfg, args.base_price_cny_m3)
    obs, initial_info = env.reset(seed=seed)
    asbuilt_design = {
        "thickness": [float(x) for x in env._design["thickness"]],
        "modulus": [float(x) for x in env._design["modulus"]],
    }
    asbuilt_eval = env._last_evaluation
    asbuilt_responses = dict(env._last_responses)
    last_info = initial_info
    for _ in range(args.max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, last_info = env.step(project_action(action))
        if terminated or truncated:
            break
    alt_design = design_from_info(last_info, env._design)
    alt_eval = env._best_evaluation or env._last_evaluation
    alt_responses = dict(env._best_responses or env._last_responses)
    n_esal = float(case["meta"]["measured_final"]["ESAL_at_final"])
    measured_rut = float(case["meta"]["measured_final"]["rut_mm"])
    asbuilt_jtg = jtg_result(asbuilt_eval, dsr_fn)
    alt_jtg = jtg_result(alt_eval, dsr_fn)
    asbuilt_rut = mepdg_rut_result(
        asbuilt_responses, asbuilt_design, n_esal, args.design_temperature_C, mepdg_rutting_mm
    )
    alt_rut = mepdg_rut_result(
        alt_responses, alt_design, n_esal, args.design_temperature_C, mepdg_rutting_mm
    )
    row = {
        "section": case["section_id"],
        "seed": seed,
        "representation": "3AC + aggregate base split into two tied equivalent sublayers",
        "policy_mode": "frozen PPO, deterministic, modulus actions masked",
        "temperature_C": args.design_temperature_C,
        "ESAL": n_esal,
        "measured_asbuilt_rut_mm": measured_rut,
        "asbuilt_h_cm": [round(x * 100.0, 3) for x in asbuilt_design["thickness"]],
        "alternative_h_cm": [round(x * 100.0, 3) for x in alt_design["thickness"]],
        "asbuilt_cost_cny_m2": design_cost(env, asbuilt_design),
        "alternative_cost_cny_m2": design_cost(env, alt_design),
        "asbuilt_jtg": asbuilt_jtg,
        "alternative_jtg": alt_jtg,
        "asbuilt_mepdg_rut": asbuilt_rut,
        "alternative_mepdg_rut": alt_rut,
        "guard_rejections": int(last_info.get("n_guard_violations", 0) or 0),
        "claim_boundary": (
            "retrospective comparative benchmark; the alternative design was not constructed "
            "and has no observed field performance"
        ),
    }
    env.close()
    return row


def flat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ac = row["asbuilt_cost_cny_m2"]
    ic = row["alternative_cost_cny_m2"]
    ar = row["asbuilt_mepdg_rut"]["rut_total_mm"]
    ir = row["alternative_mepdg_rut"]["rut_total_mm"]
    return {
        "section": row["section"],
        "seed": row["seed"],
        "temperature_C": row["temperature_C"],
        "ESAL": row["ESAL"],
        "measured_asbuilt_rut_mm": row["measured_asbuilt_rut_mm"],
        "asbuilt_h_cm": json.dumps(row["asbuilt_h_cm"]),
        "alternative_h_cm": json.dumps(row["alternative_h_cm"]),
        "asbuilt_cost_cny_m2": ac,
        "alternative_cost_cny_m2": ic,
        "cost_change_pct": (ic / ac - 1.0) * 100.0 if ac else math.nan,
        "asbuilt_jtg_dsr": row["asbuilt_jtg"]["dsr"],
        "alternative_jtg_dsr": row["alternative_jtg"]["dsr"],
        "asbuilt_jtg_feasible": row["asbuilt_jtg"]["feasible"],
        "alternative_jtg_feasible": row["alternative_jtg"]["feasible"],
        "asbuilt_pred_rut_mm": ar,
        "alternative_pred_rut_mm": ir,
        "pred_rut_change_pct": (ir / ar - 1.0) * 100.0 if ar else math.nan,
        "guard_rejections": row["guard_rejections"],
        "claim_boundary": row["claim_boundary"],
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    args.project_root = root
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    cases_path = resolve(root, args.cases)
    policy_path = resolve(root, args.policy)
    out_dir = resolve(root, args.out_dir)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    wanted = parse_sections(args.sections)
    cases = [
        c for c in payload["cases"]
        if c["section_id"].replace("NCAT_CG_", "").upper() in wanted
    ]
    seeds = parse_ints(args.seeds)
    if not cases:
        raise SystemExit("no selected NCAT sections found")
    if not seeds:
        raise SystemExit("no seeds selected")
    plan = {
        "project_root": str(root),
        "cases": [c["section_id"] for c in cases],
        "seeds": seeds,
        "max_steps": args.max_steps,
        "policy": str(policy_path),
        "protocol": args.protocol,
        "use_surrogate": args.use_surrogate,
        "representation": "NCAT aggregate base split into two identical tied sublayers",
        "moduli": "fixed to measured/reference NCAT values; all modulus actions masked",
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    from scripts.ltpp_inference import load_policy
    from scripts.run_mepdg_cross_spec_check import mepdg_rutting_mm
    from rl.dsr_patch import compute_dsr

    env_class = build_env_class()
    policy = load_policy(policy_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        for seed in seeds:
            print(f"Running {case['section_id']} seed={seed}", flush=True)
            rows.append(
                run_case(args, case, seed, policy, env_class, compute_dsr, mepdg_rutting_mm)
            )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"ncat_design_benchmark_{stamp}.json"
    csv_path = out_dir / f"ncat_design_benchmark_{stamp}.csv"
    json_path.write_text(json.dumps(rows, indent=2, default=jsonable), encoding="utf-8")
    flat = [flat_row(r) for r in rows]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
