from __future__ import annotations

"""Audit whether the fixed-material NCAT thickness space contains a JTG-compliant design.

This workflow deliberately bypasses PPO and the LLM Generator.  It is a design-space
feasibility audit, not an iLLM-PD validation experiment.  Every accepted conclusion
is based on real FEA evaluations; the dry-run path only prepares and validates a plan.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


VARIABLE_NAMES = ("upper_ac_m", "mid_ac_m", "lower_ac_m", "aggregate_total_m")
LOWER = np.array([0.02, 0.03, 0.04, 0.08], dtype=float)
UPPER = np.array([0.10, 0.15, 0.25, 0.50], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--cases", type=Path,
                   default=Path("experiments/ncat_data/ncat_cases.json"))
    p.add_argument("--sections", default="N1")
    p.add_argument("--protocol", default="JTG_D50_2017")
    p.add_argument("--n-candidates", type=int, default=48)
    p.add_argument("--max-evals", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--num-cpus", type=int, default=4)
    p.add_argument("--base-price-cny-m3", type=float, default=100.0)
    p.add_argument("--out-dir", type=Path,
                   default=Path("experiments/ncat_feasibility_audit"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-first-compliant", action="store_true")
    p.add_argument("--keep-runs", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def section_tokens(raw: str) -> set[str]:
    return {
        token.strip().upper().replace("NCAT_CG_", "")
        for token in raw.split(",") if token.strip()
    }


def validate_case(case: Dict[str, Any]) -> None:
    if not str(case.get("section_id", "")).startswith("NCAT_CG_"):
        raise ValueError("section_id must start with NCAT_CG_")
    ec = case.get("envconfig", {})
    for key in ("init_thickness_m", "init_modulus_MPa", "init_poisson"):
        values = np.asarray(ec.get(key, []), dtype=float)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{case['section_id']}: invalid {key}")
    if np.any(np.asarray(ec["init_thickness_m"], dtype=float) <= 0):
        raise ValueError(f"{case['section_id']}: thicknesses must be positive")
    if np.any(np.asarray(ec["init_modulus_MPa"], dtype=float) <= 0):
        raise ValueError(f"{case['section_id']}: moduli must be positive")
    for key in ("E_subgrade", "nu_subgrade"):
        value = float(ec.get(key, math.nan))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{case['section_id']}: invalid {key}")


def equivalent_materials(case: Dict[str, Any]) -> Dict[str, List[float]]:
    """Map 3 AC + one aggregate base + subgrade to the five-layer FE interface."""
    validate_case(case)
    ec = case["envconfig"]
    h = list(map(float, ec["init_thickness_m"]))
    e = list(map(float, ec["init_modulus_MPa"]))
    nu = list(map(float, ec["init_poisson"]))
    return {
        "asbuilt_variables": [h[0], h[1], h[2], h[3]],
        "modulus": [e[0], e[1], e[2], e[3], e[3]],
        "poisson": [nu[0], nu[1], nu[2], nu[3], nu[3]],
    }


def variables_to_thickness(values: Sequence[float]) -> List[float]:
    x = np.asarray(values, dtype=float)
    if x.shape != (4,) or np.any(~np.isfinite(x)):
        raise ValueError("candidate variables must be four finite values")
    if np.any(x < LOWER - 1e-12) or np.any(x > UPPER + 1e-12):
        raise ValueError("candidate lies outside the audited bounds")
    half = float(x[3] / 2.0)
    return [float(x[0]), float(x[1]), float(x[2]), half, half]


def _candidate_id(section_id: str, values: Sequence[float]) -> str:
    body = section_id + "|" + ",".join(f"{float(x):.6f}" for x in values)
    return hashlib.sha256(body.encode("ascii")).hexdigest()[:12]


def _anchors(asbuilt: np.ndarray) -> List[np.ndarray]:
    clipped = np.clip(asbuilt, LOWER, UPPER)
    anchors = [
        clipped,
        np.array([0.10, 0.15, 0.25, 0.50]),
        np.array([0.02, 0.15, 0.25, 0.50]),
        np.array([0.04, 0.12, 0.20, 0.50]),
        np.array([0.06, 0.10, 0.16, 0.50]),
        np.array([0.08, 0.12, 0.18, 0.30]),
        np.array([0.04, 0.10, 0.16, 0.30]),
        np.array([0.03, 0.08, 0.14, 0.40]),
    ]
    for factor in (1.25, 1.50, 2.00):
        anchors.append(np.clip(asbuilt * factor, LOWER, UPPER))
    return anchors


def generate_candidates(case: Dict[str, Any], n_candidates: int,
                        seed: int) -> List[Dict[str, Any]]:
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    materials = equivalent_materials(case)
    asbuilt = np.asarray(materials["asbuilt_variables"], dtype=float)
    section_id = str(case["section_id"])
    section_seed = int(hashlib.sha256(section_id.encode("ascii")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + section_seed)

    points = list(_anchors(asbuilt))
    n_lhs = max(n_candidates, 1)
    lhs = np.empty((n_lhs, 4), dtype=float)
    for column in range(4):
        lhs[:, column] = (rng.permutation(n_lhs) + rng.random(n_lhs)) / n_lhs
    points.extend(LOWER + lhs * (UPPER - LOWER))

    candidates: List[Dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for point in points:
        point = np.clip(np.asarray(point, dtype=float), LOWER, UPPER)
        key = tuple(round(float(x), 6) for x in point)
        if key in seen:
            continue
        seen.add(key)
        thickness = variables_to_thickness(point)
        candidates.append({
            "candidate_id": _candidate_id(section_id, point),
            "section_id": section_id,
            "variables": dict(zip(VARIABLE_NAMES, map(float, point))),
            "thickness_m": thickness,
            "thickness_cm": [round(x * 100.0, 4) for x in thickness],
            "aggregate_tied": bool(abs(thickness[3] - thickness[4]) < 1e-12),
        })
        if len(candidates) >= n_candidates:
            break
    return candidates


def material_cost(design_h: Iterable[float], design_e: Iterable[float],
                  base_price: float) -> float:
    h = np.asarray(list(design_h), dtype=float)
    e = np.asarray(list(design_e), dtype=float)
    prices = np.array([1800.0, 1100.0, 900.0, base_price, base_price])
    coeffs = np.array([2.0e-5, 1.8e-5, 1.5e-5, 0.0, 0.0])
    return float(np.sum(prices * h * (1.0 + coeffs * e)))


def evaluate_candidate(root: Path, case: Dict[str, Any], candidate: Dict[str, Any],
                       protocol_name: str, num_cpus: int, base_price: float,
                       keep_runs: bool) -> Dict[str, Any]:
    from fea.runner import run_fea
    from rl.dsr_patch import compute_dsr
    from specs import get_protocol
    from specs.protocol import DesignInputs

    materials = equivalent_materials(case)
    ec = case["envconfig"]
    h = list(map(float, candidate["thickness_m"]))
    e = list(map(float, materials["modulus"]))
    nu = list(map(float, materials["poisson"]))
    started = time.time()
    run_dir = None
    try:
        full = run_fea(
            thickness=h, modulus=e, poisson=nu,
            E_subgrade=float(ec["E_subgrade"]),
            nu_subgrade=float(ec["nu_subgrade"]),
            load_pressure=0.7, load_radius=0.1065,
            base_dir=str(root), num_cpus=num_cpus, verbose=False,
        )
        run_dir = full.get("run_dir")
        responses = full.get("responses", full)
        inputs = DesignInputs(
            pavement_type="flexible", road_class="expressway",
            traffic_level="heavy", thickness=h, modulus=e, poisson=nu,
            E_subgrade=float(ec["E_subgrade"]),
            nu_subgrade=float(ec["nu_subgrade"]), design_life=15,
            extras={"city": "", "climate_zone": "warm", "VFA_pct": 70.0,
                    "R_s_MPa": 1.0, "R_0_mm": 1.5},
        )
        evaluation = get_protocol(protocol_name).evaluate(inputs, responses)
        margins = {k: float(v) for k, v in evaluation.margins.items()}
        return {
            **candidate,
            "status": "ok",
            "feasible": bool(evaluation.feasible),
            "dsr": float(compute_dsr(margins)),
            "critical": str(evaluation.critical_indicator),
            "margins": margins,
            "cost_cny_m2": material_cost(h, e, base_price),
            "elapsed_s": time.time() - started,
            "responses": {k: float(v) for k, v in responses.items()
                          if isinstance(v, (int, float, np.number))},
            "claim_boundary": "real-FEA design-space feasibility audit; not an iLLM-PD output",
        }
    except Exception as exc:
        return {
            **candidate, "status": "error", "feasible": False,
            "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_s": time.time() - started,
        }
    finally:
        if run_dir and not keep_runs:
            shutil.rmtree(Path(run_dir), ignore_errors=True)


def load_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["candidate_id"])] = row
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("status") == "ok"]
    feasible = [r for r in valid if r.get("feasible")]
    best_dsr = max(valid, key=lambda r: float(r["dsr"])) if valid else None
    cheapest = min(feasible, key=lambda r: float(r["cost_cny_m2"])) if feasible else None
    return {
        "n_records": len(rows),
        "n_successful_fea": len(valid),
        "n_failed_fea": len(rows) - len(valid),
        "n_compliant": len(feasible),
        "compliant_design_found": bool(feasible),
        "best_dsr_candidate": best_dsr,
        "cheapest_compliant_candidate": cheapest,
        "interpretation": (
            "At least one compliant design exists in the audited fixed-material thickness space."
            if feasible else
            "No compliant design has yet been found among the evaluated candidates; this is not proof that the full bounded space is infeasible."
        ),
        "manuscript_status": "pending; do not report as NCAT validation",
    }


def write_plan(path: Path, plan: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        flat.append({
            "candidate_id": row.get("candidate_id"),
            "section_id": row.get("section_id"),
            "status": row.get("status"),
            "feasible": row.get("feasible"),
            "dsr": row.get("dsr"),
            "critical": row.get("critical"),
            "cost_cny_m2": row.get("cost_cny_m2"),
            "thickness_cm": json.dumps(row.get("thickness_cm")),
            "margins": json.dumps(row.get("margins"), sort_keys=True),
            "elapsed_s": row.get("elapsed_s"),
            "error": row.get("error"),
        })
    if not flat:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    args = parse_args()
    if args.n_candidates <= 0 or args.max_evals <= 0:
        raise SystemExit("n-candidates and max-evals must be positive")
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    cases_path = resolve(root, args.cases)
    out_dir = resolve(root, args.out_dir)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    wanted = section_tokens(args.sections)
    cases = [c for c in payload["cases"]
             if c["section_id"].replace("NCAT_CG_", "").upper() in wanted]
    if not cases:
        raise SystemExit("no selected NCAT sections found")

    plans = []
    for case in cases:
        validate_case(case)
        candidates = generate_candidates(case, args.n_candidates, args.seed)
        plans.append({
            "section_id": case["section_id"],
            "fixed_modulus_MPa": equivalent_materials(case)["modulus"],
            "fixed_poisson": equivalent_materials(case)["poisson"],
            "E_subgrade_MPa": float(case["envconfig"]["E_subgrade"]),
            "bounds": {name: [float(lo), float(hi)] for name, lo, hi
                       in zip(VARIABLE_NAMES, LOWER, UPPER)},
            "n_candidates": len(candidates),
            "max_evals_this_invocation": args.max_evals,
            "candidates": candidates,
        })
    plan = {
        "workflow": "NCAT fixed-material thickness-space feasibility audit",
        "protocol": args.protocol,
        "seed": args.seed,
        "real_fea_required_for_conclusion": True,
        "claim_boundary": "independent feasibility audit; PPO and Generator are bypassed",
        "sections": plans,
    }
    write_plan(out_dir / "candidate_plan.json", plan)
    print(json.dumps({"plan": str(out_dir / "candidate_plan.json"),
                      "sections": [p["section_id"] for p in plans],
                      "dry_run": bool(args.dry_run)}, indent=2))
    if args.dry_run:
        return

    for case, section_plan in zip(cases, plans):
        section = case["section_id"].replace("NCAT_CG_", "").lower()
        section_dir = out_dir / section
        section_dir.mkdir(parents=True, exist_ok=True)
        jsonl = section_dir / "evaluations.jsonl"
        completed = {} if args.no_resume else load_completed(jsonl)
        attempted = 0
        for candidate in section_plan["candidates"]:
            if candidate["candidate_id"] in completed:
                continue
            if attempted >= args.max_evals:
                break
            row = evaluate_candidate(
                root, case, candidate, args.protocol, args.num_cpus,
                args.base_price_cny_m3, args.keep_runs,
            )
            with jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            completed[candidate["candidate_id"]] = row
            attempted += 1
            print(json.dumps({"section": case["section_id"],
                              "candidate": candidate["candidate_id"],
                              "status": row["status"], "dsr": row.get("dsr"),
                              "feasible": row.get("feasible")}), flush=True)
            if args.stop_on_first_compliant and row.get("feasible"):
                break
        rows = list(completed.values())
        summary = summarize(rows)
        (section_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        write_csv(section_dir / "evaluations.csv", rows)
        print(json.dumps({"section": case["section_id"], **summary},
                         default=str, indent=2))


if __name__ == "__main__":
    main()
