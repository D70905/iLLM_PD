from __future__ import annotations

"""Diagnose NCAT rutting validation without starting ABAQUS.

The script reuses cached elastic responses, corrects the HMA sublayer
integration, evaluates the uncalibrated national-coefficient proxy, and runs a
strictly separated early-period calibration and later-period forecast.  It
never labels the reduced-order ESAL calculation as a full MEPDG validation.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SECTIONS = ["N1", "N2", "N5", "N8", "S5", "S6", "S13"]
FIELD_COLUMNS = {s: s[0] + s[1:].zfill(2) for s in SECTIONS}
K1, K2, K3 = -3.35412, 1.5606, 0.4791


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit and diagnose the NCAT rut model")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--cases", type=Path,
                   default=Path("experiments/ncat_data/ncat_cases.json"))
    p.add_argument("--field-data", type=Path,
                   default=Path("experiments/ncat_data/Field Data.xlsx"))
    p.add_argument("--mastercurves", type=Path,
                   default=Path("experiments/ncat_data/Dynamic Modulus_Mastercurves.xlsx"))
    p.add_argument("--temperature-C", type=float, default=22.8)
    p.add_argument("--frequency-Hz", type=float, default=10.0)
    p.add_argument("--calibration-ESAL", type=float, default=10_000_000.0)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--out-dir", type=Path,
                   default=Path("experiments/ncat_external_validation/diagnosis"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def hma_depth_factor(total_hma_in: float, depth_in: float) -> float:
    c1 = -0.1039 * total_hma_in ** 2 + 2.4868 * total_hma_in - 17.342
    c2 = 0.0172 * total_hma_in ** 2 - 1.7331 * total_hma_in + 27.428
    return (c1 + c2 * depth_in) * 0.328196 ** depth_in


def hma_rut_sublayers(thickness_mm: Iterable[float], strain_micro: Iterable[float],
                       n_proxy: float, temperature_f: float) -> Tuple[float, List[float]]:
    hs = np.asarray(list(thickness_mm), dtype=float)
    strains = np.asarray(list(strain_micro), dtype=float)
    if hs.shape != (3,) or strains.shape != (3,):
        raise ValueError("three HMA thicknesses and three HMA strains are required")
    total_in = float(np.sum(hs) / 25.4)
    traffic_temperature = (10.0 ** K1) * temperature_f ** K2 * n_proxy ** K3
    depth_mm = 0.0
    layers = []
    for h, strain in zip(hs, strains):
        depth_in = (depth_mm + h / 2.0) / 25.4
        kz = hma_depth_factor(total_in, depth_in)
        layers.append(float(h * strain * 1.0e-6 * kz * traffic_temperature))
        depth_mm += h
    return float(sum(layers)), layers


def unbound_rut(eps_z_micro: float, h_base_mm: float, n_proxy: float) -> float:
    # The cache does not contain a separate base-midpoint strain.  This remains
    # an explicit proxy using subgrade-top strain for both unbound components.
    eps = max(float(eps_z_micro), 0.0) * 1.0e-6
    granular = 1.673 * (2.03 / 0.5) * math.exp(-((650.0 / n_proxy) ** 0.92))
    subgrade = 1.350 * (1.62 / 0.5) * math.exp(-((367.0 / n_proxy) ** 1.04))
    return float(h_base_mm * eps * granular + 152.4 * eps * subgrade)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    dx = x - float(np.mean(x))
    dy = y - float(np.mean(y))
    denom = math.sqrt(float(np.sum(dx * dx)) * float(np.sum(dy * dy)))
    return float(np.sum(dx * dy) / denom) if denom > 0 else math.nan


def metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    error = predicted - observed
    ss_tot = float(np.sum((observed - float(np.mean(observed))) ** 2))
    return {
        "n": int(len(observed)),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": math.sqrt(float(np.mean(error ** 2))),
        "mean_bias_mm": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error ** 2) / ss_tot) if ss_tot > 0 else math.nan,
        "pearson_r": safe_corr(observed, predicted),
    }


def bootstrap_error_intervals(observed: np.ndarray, predicted: np.ndarray,
                              n_boot: int, seed: int) -> Dict[str, List[float]]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(observed), size=(n_boot, len(observed)))
    e = (predicted - observed)[idx]
    values = {
        "mae_mm": np.mean(np.abs(e), axis=1),
        "rmse_mm": np.sqrt(np.mean(e ** 2, axis=1)),
        "mean_bias_mm": np.mean(e, axis=1),
    }
    return {f"{k}_95pct_ci": [float(v) for v in np.percentile(a, [2.5, 97.5])]
            for k, a in values.items()}


def load_field_points(path: Path, target_esal: float) -> Dict[str, Dict[str, float]]:
    frame = pd.read_excel(path, sheet_name="Rutting", header=1)
    esal = pd.to_numeric(frame["ESAL"], errors="coerce")
    out: Dict[str, Dict[str, float]] = {}
    for section in SECTIONS:
        values = pd.to_numeric(frame[FIELD_COLUMNS[section]], errors="coerce")
        valid = esal.notna() & values.notna()
        if not valid.any():
            raise ValueError(f"{section}: no rutting observations")
        first_idx = values[valid].index[0]
        cal_idx = (esal[valid] - target_esal).abs().idxmin()
        final_idx = esal[valid].idxmax()
        out[section] = {
            "baseline_rut_mm": float(values[first_idx]) * 25.4,
            "calibration_ESAL": float(esal[cal_idx]),
            "calibration_rut_mm": float(values[cal_idx]) * 25.4,
            "final_ESAL": float(esal[final_idx]),
            "final_rut_mm": float(values[final_idx]) * 25.4,
            "complete_to_20M": bool(float(esal[final_idx]) >= 19_500_000.0),
        }
    return out


def expected_moduli(xlsx: Path, temperature_c: float, frequency_hz: float) -> Dict[str, Tuple[float, float]]:
    import mc_estar
    return {
        s: (
            float(mc_estar.estar_MPa(str(xlsx), s, "RH", "surface",
                                     temperature_c, frequency_hz)),
            float(mc_estar.estar_MPa(str(xlsx), s, "RH", "base",
                                     temperature_c, frequency_hz)),
        )
        for s in SECTIONS
    }


def cached_candidates(root: Path) -> List[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    candidates = []
    for result_path in (root / "output/runs").glob("run_*/pavement_result.json"):
        input_path = result_path.with_name("pavement_input.json")
        if not input_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            inp = json.loads(input_path.read_text(encoding="utf-8"))
            responses = result.get("responses", {})
            required = [
                "eps_AC_upper_mid_microstrain", "eps_AC_mid_mid_microstrain",
                "eps_AC_lower_mid_microstrain", "epsilon_z_microstrain",
            ]
            if all(responses.get(k) is not None for k in required):
                candidates.append((result_path.parent, inp, responses))
        except (OSError, ValueError, TypeError):
            continue
    return candidates


def match_cached_runs(root: Path, cases: Dict[str, Any], expected: Dict[str, Tuple[float, float]]
                      ) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    candidates = cached_candidates(root)
    matched = {}
    for section in SECTIONS:
        case = cases[f"NCAT_CG_{section}"]
        target_h = np.asarray(case["envconfig"]["init_thickness_m"], dtype=float)
        target_surface, target_base = expected[section]
        ranked = []
        for run_dir, inp, response in candidates:
            h = np.asarray(inp.get("thickness", []), dtype=float)
            mod = np.asarray(inp.get("modulus", []), dtype=float)
            if h.shape != (5,) or mod.shape != (5,):
                continue
            if float(np.max(np.abs(h - target_h))) > 5.0e-4:
                continue
            score = abs(mod[0] / target_surface - 1.0) + abs(mod[1] / target_base - 1.0)
            ranked.append((score, run_dir.stat().st_mtime, run_dir, response))
        if not ranked:
            raise FileNotFoundError(f"{section}: no compatible cached FEA result")
        score, _, run_dir, response = min(ranked, key=lambda x: (x[0], -x[1]))
        if score > 0.03:
            raise ValueError(f"{section}: closest cached FEA modulus mismatch is {score:.3f}")
        matched[section] = (run_dir, response)
    return matched


def fit_beta(observed_change: np.ndarray, ac_proxy: np.ndarray,
             unbound_proxy: np.ndarray) -> float:
    denominator = float(np.sum(ac_proxy ** 2))
    if denominator <= 0:
        raise ValueError("cannot fit beta with zero HMA proxy")
    return float(np.sum(ac_proxy * (observed_change - unbound_proxy)) / denominator)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    case_payload = json.loads(resolve(root, args.cases).read_text(encoding="utf-8"))
    cases = {c["section_id"]: c for c in case_payload["cases"]}
    field = load_field_points(resolve(root, args.field_data), args.calibration_ESAL)
    expected = expected_moduli(resolve(root, args.mastercurves),
                               args.temperature_C, args.frequency_Hz)
    matched = match_cached_runs(root, cases, expected)

    plan = {
        "sections": SECTIONS,
        "temperature_C": args.temperature_C,
        "frequency_Hz": args.frequency_Hz,
        "calibration_target_ESAL": args.calibration_ESAL,
        "cached_runs": {s: str(matched[s][0]) for s in SECTIONS},
        "starts_ABAQUS": False,
        "model_class": "reduced-order ESAL proxy, not full MEPDG",
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    temperature_f = args.temperature_C * 9.0 / 5.0 + 32.0
    rows: List[Dict[str, Any]] = []
    for section in SECTIONS:
        case = cases[f"NCAT_CG_{section}"]
        h_mm = np.asarray(case["envconfig"]["init_thickness_m"][:3]) * 1000.0
        base_h_mm = float(case["envconfig"]["init_thickness_m"][3]) * 1000.0
        run_dir, response = matched[section]
        strains = [response["eps_AC_upper_mid_microstrain"],
                   response["eps_AC_mid_mid_microstrain"],
                   response["eps_AC_lower_mid_microstrain"]]
        fp = field[section]
        ac_cal, ac_layers_cal = hma_rut_sublayers(
            h_mm, strains, fp["calibration_ESAL"], temperature_f)
        ac_final, ac_layers_final = hma_rut_sublayers(
            h_mm, strains, fp["final_ESAL"], temperature_f)
        unbound_cal = unbound_rut(response["epsilon_z_microstrain"], base_h_mm,
                                  fp["calibration_ESAL"])
        unbound_final = unbound_rut(response["epsilon_z_microstrain"], base_h_mm,
                                    fp["final_ESAL"])
        rows.append({
            "section": section,
            "cached_run": str(run_dir),
            **fp,
            "observed_calibration_change_mm": fp["calibration_rut_mm"] - fp["baseline_rut_mm"],
            "observed_final_change_mm": fp["final_rut_mm"] - fp["baseline_rut_mm"],
            "ac_proxy_calibration_mm": ac_cal,
            "unbound_proxy_calibration_mm": unbound_cal,
            "ac_proxy_final_mm": ac_final,
            "unbound_proxy_final_mm": unbound_final,
            "national_proxy_final_mm": ac_final + unbound_final,
            "ac_layer1_final_mm": ac_layers_final[0],
            "ac_layer2_final_mm": ac_layers_final[1],
            "ac_layer3_final_mm": ac_layers_final[2],
        })

    observed_cal = np.asarray([r["observed_calibration_change_mm"] for r in rows])
    ac_cal = np.asarray([r["ac_proxy_calibration_mm"] for r in rows])
    unbound_cal = np.asarray([r["unbound_proxy_calibration_mm"] for r in rows])
    observed_final = np.asarray([r["observed_final_change_mm"] for r in rows])
    ac_final = np.asarray([r["ac_proxy_final_mm"] for r in rows])
    unbound_final = np.asarray([r["unbound_proxy_final_mm"] for r in rows])
    national_final = ac_final + unbound_final
    beta = fit_beta(observed_cal, ac_cal, unbound_cal)
    temporal_final = beta * ac_final + unbound_final

    loso_predictions = []
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        beta_i = fit_beta(observed_cal[keep], ac_cal[keep], unbound_cal[keep])
        loso_predictions.append(beta_i * ac_final[i] + unbound_final[i])
        rows[i]["loso_beta_from_other_sections"] = beta_i
    loso_predictions = np.asarray(loso_predictions)
    for i, row in enumerate(rows):
        row["temporal_prediction_global_beta_mm"] = temporal_final[i]
        row["loso_temporal_prediction_mm"] = loso_predictions[i]

    nominal_metrics = metrics(observed_final, national_final)
    nominal_metrics.update(bootstrap_error_intervals(
        observed_final, national_final, args.bootstrap, args.seed))
    temporal_metrics = metrics(observed_final, temporal_final)
    temporal_metrics.update(bootstrap_error_intervals(
        observed_final, temporal_final, args.bootstrap, args.seed))
    loso_metrics = metrics(observed_final, loso_predictions)
    loso_metrics.update(bootstrap_error_intervals(
        observed_final, loso_predictions, args.bootstrap, args.seed))

    blockers = [
        "MEPDG requires axle-load spectra; cumulative ESAL is not a load-repetition input.",
        "Damage is evaluated at one temperature instead of incrementally by traffic and climate period.",
        "Cached FEA lacks an independent base-midpoint strain; the unbound term uses a proxy.",
        "Foundation moduli are provisional rather than NCAT backcalculated values.",
        "The seven sections share one nominal geometry and do not test cross-structure transfer.",
        "N5 and N8 terminate before 20 million ESAL and must not be marked complete to 20M.",
        "Dynamic modulus alone does not identify mixture-specific permanent-deformation susceptibility.",
    ]
    summary = {
        "status": "not_ready_for_quantitative_validation_claim",
        "analysis_plan": plan,
        "corrected_implementation": "three HMA sublayers integrated separately",
        "national_proxy_metrics": nominal_metrics,
        "early_period_global_beta": beta,
        "temporal_forecast_metrics": temporal_metrics,
        "leave_one_section_out_temporal_metrics": loso_metrics,
        "blockers": blockers,
        "recommended_primary_target": (
            "validate cached FEA against NCAT earth-pressure-cell measurements at the "
            "AC/base and base/subgrade interfaces"
        ),
        "claim_boundary": (
            "The temporal analysis is a locally calibrated reduced-order forecast. "
            "It is not an independent full-MEPDG field validation."
        ),
    }
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ncat_model_diagnosis_{stamp}.csv"
    json_path = out_dir / f"ncat_model_diagnosis_{stamp}.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path),
                      "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
