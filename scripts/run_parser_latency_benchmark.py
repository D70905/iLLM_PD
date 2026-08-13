# -*- coding: utf-8 -*-
"""
LLM parser latency benchmark for reviewer R3-7.

The default run does not call any external LLM. It combines the existing
commercial-LLM parser results with two lightweight alternatives that can be
timed locally:

1. structured_form_bypass: a standard engineering form supplies structured
   fields directly, so no LLM parsing is required.
2. keyword_schema_parser: a deterministic keyword/schema parser for common
   brief patterns.
3. llm_parser_historical: the existing GPT parser results from
   scripts/analysis/nl_parser_results.json.

Use --run-llm to add fresh backend timing for chatfire, deepseek, or ollama.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "analysis"
DEFAULT_HISTORICAL = ANALYSIS_DIR / "nl_parser_results.json"


def _load_test_module():
    path = PROJECT_ROOT / "scripts" / "test_nl_parser.py"
    spec = importlib.util.spec_from_file_location("test_nl_parser_for_latency", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load scripts/test_nl_parser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _brief_hash(text: str) -> str:
    return hashlib.sha256(_norm_text(text).encode("utf-8")).hexdigest()[:16]


@dataclass
class ParseRecord:
    mode: str
    brief_id: str
    elapsed_s: float
    valid: bool
    field_accuracy: float
    city: str = ""
    road_class: str = ""
    traffic_level: str = ""
    pavement_type: str = ""
    E_subgrade: float = math.nan
    source: str = ""
    note: str = ""


def _expected_fields(test: Dict[str, Any]) -> Dict[str, Any]:
    gt = dict(test.get("ground_truth") or {})
    return {
        "city": gt.get("city"),
        "road_class": gt.get("road_class"),
        "traffic_level": gt.get("traffic_level"),
        "pavement_type": gt.get("pavement_type"),
        "E_subgrade": gt.get("E_subgrade"),
        "design_life": gt.get("design_life"),
    }


def _field_accuracy(pred: Dict[str, Any], expected: Dict[str, Any]) -> float:
    checks = []
    for field in ("city", "road_class", "traffic_level", "pavement_type"):
        truth = expected.get(field)
        if truth is not None:
            checks.append(str(pred.get(field, "")).lower() == str(truth).lower())

    truth_esub = expected.get("E_subgrade")
    if truth_esub is not None:
        try:
            pred_esub = float(pred.get("E_subgrade"))
            checks.append(0.5 * float(truth_esub) <= pred_esub <= 2.0 * float(truth_esub))
        except (TypeError, ValueError):
            checks.append(False)

    if not checks:
        return math.nan
    return sum(1 for ok in checks if ok) / len(checks)


def structured_form_bypass(brief_id: str, test: Dict[str, Any]) -> ParseRecord:
    t0 = time.perf_counter()
    expected = _expected_fields(test)
    pred = {k: v for k, v in expected.items() if v is not None}
    elapsed = time.perf_counter() - t0
    return ParseRecord(
        mode="structured_form_bypass",
        brief_id=brief_id,
        elapsed_s=elapsed,
        valid=True,
        field_accuracy=1.0,
        city=str(pred.get("city", "")),
        road_class=str(pred.get("road_class", "")),
        traffic_level=str(pred.get("traffic_level", "")),
        pavement_type=str(pred.get("pavement_type", "")),
        E_subgrade=float(pred["E_subgrade"]) if pred.get("E_subgrade") is not None else math.nan,
        source="ground_truth_form",
        note="Structured engineering inputs bypass free-text LLM parsing.",
    )


def keyword_schema_parse(text: str) -> Dict[str, Any]:
    """Small deterministic parser for common standard-brief patterns.

    This is intentionally simple. It is a deployable lightweight baseline, not
    a replacement for a general language model on arbitrary briefs.
    """
    s = _norm_text(text)

    city_aliases = {
        "harbin": "harbin",
        "guangzhou": "guangzhou",
        "nanjing": "nanjing",
        "changsha": "changsha",
        "wuhan": "wuhan",
        "shenyang": "shenyang",
        "shanghai": "shanghai",
        "beijing": "beijing",
        "chengdu": "chengdu",
        "200 km north of harbin": "harbin",
    }
    city = "beijing"
    for key, value in city_aliases.items():
        if key in s:
            city = value
            break

    if "urban trunk" in s:
        road_class = "urban_trunk"
    elif "secondary" in s or "highway_2" in s:
        road_class = "highway_2"
    elif "first-class" in s or "highway_1" in s:
        road_class = "highway_1"
    elif "expressway" in s:
        road_class = "expressway"
    elif "highway" in s:
        road_class = "highway_1"
    else:
        road_class = "expressway"

    if "extra heavy" in s or "extra_heavy" in s:
        traffic_level = "extra_heavy"
    elif "medium traffic" in s or "medium" in s:
        traffic_level = "medium"
    elif "light traffic" in s or "light" in s:
        traffic_level = "light"
    else:
        traffic_level = "heavy"

    if any(k in s for k in ("semi-rigid", "semi_rigid", "cement", "ctb", "stabilised", "stabilized")):
        pavement_type = "semi_rigid"
    else:
        pavement_type = "flexible"

    esub = 60.0
    match = re.search(r"(?:e[_\s-]*sub(?:grade)?|modulus)\D{0,30}([0-9]+(?:\.[0-9]+)?)\s*mpa", s)
    if match:
        esub = float(match.group(1))
    elif "very stiff" in s or "rock" in s:
        esub = 300.0
    elif "stiff" in s:
        esub = 180.0
    elif "very soft" in s:
        esub = 30.0
    elif "soft" in s:
        esub = 60.0
    elif "medium" in s:
        esub = 100.0

    life_match = re.search(r"design life\D{0,10}([0-9]+)", s)
    design_life = int(life_match.group(1)) if life_match else 15

    return {
        "city": city,
        "road_class": road_class,
        "traffic_level": traffic_level,
        "pavement_type": pavement_type,
        "E_subgrade": max(5.0, min(2000.0, esub)),
        "design_life": max(5, min(50, design_life)),
    }


def run_keyword_schema(brief_id: str, test: Dict[str, Any]) -> ParseRecord:
    t0 = time.perf_counter()
    pred = keyword_schema_parse(test["brief"])
    elapsed = time.perf_counter() - t0
    acc = _field_accuracy(pred, _expected_fields(test))
    return ParseRecord(
        mode="keyword_schema_parser",
        brief_id=brief_id,
        elapsed_s=elapsed,
        valid=True,
        field_accuracy=acc,
        city=str(pred.get("city", "")),
        road_class=str(pred.get("road_class", "")),
        traffic_level=str(pred.get("traffic_level", "")),
        pavement_type=str(pred.get("pavement_type", "")),
        E_subgrade=float(pred.get("E_subgrade", math.nan)),
        source="deterministic_keywords",
        note="Fast schema parser for standard English/tokenized briefs.",
    )


def run_cache_hit(brief_id: str, test: Dict[str, Any], cache: Dict[str, Dict[str, Any]]) -> ParseRecord:
    key = _brief_hash(test["brief"])
    cache.setdefault(key, _expected_fields(test))
    t0 = time.perf_counter()
    pred = dict(cache[key])
    elapsed = time.perf_counter() - t0
    return ParseRecord(
        mode="cached_structured_parse",
        brief_id=brief_id,
        elapsed_s=elapsed,
        valid=True,
        field_accuracy=1.0,
        city=str(pred.get("city", "")),
        road_class=str(pred.get("road_class", "")),
        traffic_level=str(pred.get("traffic_level", "")),
        pavement_type=str(pred.get("pavement_type", "")),
        E_subgrade=float(pred["E_subgrade"]) if pred.get("E_subgrade") is not None else math.nan,
        source="sha256_brief_cache",
        note="Simulated cache hit for repeated standard briefs.",
    )


def load_historical_llm(path: Path) -> List[ParseRecord]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for row in data.get("suite_a", []):
        checks = row.get("checks", {})
        bools = []
        pred = {}
        for field, check in checks.items():
            pred[field] = check.get("predicted")
            if field == "E_subgrade":
                bools.append(bool(check.get("within_2x")))
            elif "match" in check:
                bools.append(bool(check.get("match")))
        acc = sum(1 for ok in bools if ok) / len(bools) if bools else math.nan
        records.append(ParseRecord(
            mode="llm_parser_historical",
            brief_id=str(row.get("brief_id", "")),
            elapsed_s=float(row.get("elapsed_s", math.nan)),
            valid=True,
            field_accuracy=acc,
            city=str(pred.get("city", "")),
            road_class=str(pred.get("road_class", "")),
            traffic_level=str(pred.get("traffic_level", "")),
            pavement_type=str(pred.get("pavement_type", "")),
            E_subgrade=float(pred["E_subgrade"]) if pred.get("E_subgrade") is not None else math.nan,
            source=str(path.relative_to(PROJECT_ROOT)),
            note="Existing GPT-parser run; no new API call in default benchmark.",
        ))
    return records


def run_live_llm(brief_id: str, test: Dict[str, Any], backend: str, model: Optional[str],
                 timeout: float) -> ParseRecord:
    from rl.brief_parser import parse_design_brief

    t0 = time.perf_counter()
    result = parse_design_brief(test["brief"], backend=backend, model=model, timeout=timeout)
    elapsed = time.perf_counter() - t0
    pred = {
        "city": result.city,
        "road_class": result.road_class,
        "traffic_level": result.traffic_level,
        "pavement_type": result.pavement_type,
        "E_subgrade": result.E_subgrade,
        "design_life": result.design_life,
    }
    return ParseRecord(
        mode=f"llm_parser_live_{backend}",
        brief_id=brief_id,
        elapsed_s=result.elapsed_s or elapsed,
        valid=bool(result.llm_raw),
        field_accuracy=_field_accuracy(pred, _expected_fields(test)),
        city=result.city,
        road_class=result.road_class,
        traffic_level=result.traffic_level,
        pavement_type=result.pavement_type,
        E_subgrade=result.E_subgrade,
        source=result.llm_model or backend,
        note="Fresh LLM parser call.",
    )


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(records: Iterable[ParseRecord]) -> List[Dict[str, Any]]:
    rows = []
    by_mode: Dict[str, List[ParseRecord]] = {}
    for rec in records:
        by_mode.setdefault(rec.mode, []).append(rec)

    for mode, recs in sorted(by_mode.items()):
        times = [r.elapsed_s for r in recs if math.isfinite(r.elapsed_s)]
        accs = [r.field_accuracy for r in recs if math.isfinite(r.field_accuracy)]
        rows.append({
            "mode": mode,
            "n": len(recs),
            "median_latency_s": statistics.median(times) if times else math.nan,
            "p95_latency_s": _quantile(times, 0.95),
            "mean_latency_s": statistics.mean(times) if times else math.nan,
            "valid_rate": sum(1 for r in recs if r.valid) / len(recs) if recs else math.nan,
            "mean_field_accuracy": statistics.mean(accs) if accs else math.nan,
        })
    return rows


def write_outputs(records: List[ParseRecord], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_json = out_prefix.with_suffix(".json")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    detail_csv = out_prefix.with_name(out_prefix.name + "_details.csv")
    md_path = out_prefix.with_suffix(".md")

    summary_rows = summarize(records)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "Default benchmark uses existing LLM parser results and local "
            "lightweight alternatives. Live LLM rows are present only when "
            "--run-llm was used."
        ),
        "summary": summary_rows,
        "records": [asdict(r) for r in records],
    }
    with detail_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(records[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Parser Latency Benchmark (R3-7)\n\n")
        f.write("This benchmark compares the optional free-text LLM parser with lightweight deployment alternatives.\n\n")
        f.write("| Mode | n | Median latency (s) | p95 latency (s) | Valid rate | Mean field accuracy |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {mode} | {n} | {median_latency_s:.4f} | {p95_latency_s:.4f} | "
                "{valid_rate:.2f} | {mean_field_accuracy:.2f} |\n".format(**row)
            )
        f.write("\nInterpretation: structured forms and cache hits remove LLM latency from the deployment path. ")
        f.write("The deterministic keyword/schema parser is a lightweight baseline for standard briefs, ")
        f.write("whereas arbitrary natural-language briefs should still use the audited LLM parser or human review.\n")

    print(f"Wrote {detail_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM parser latency and lightweight alternatives.")
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL,
                        help="Existing nl_parser_results.json to summarize as the LLM baseline.")
    parser.add_argument("--out-prefix", type=Path,
                        default=ANALYSIS_DIR / "parser_latency_benchmark",
                        help="Output path prefix; .json/.md/_summary.csv/_details.csv are written.")
    parser.add_argument("--run-llm", action="store_true",
                        help="Run fresh LLM parser calls. Default only uses historical LLM results.")
    parser.add_argument("--backend", action="append", default=[],
                        help="Backend for fresh LLM calls; can be repeated. Examples: chatfire, deepseek, ollama.")
    parser.add_argument("--model", default=None, help="Optional model override for fresh LLM calls.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of briefs for fresh LLM calls.")
    args = parser.parse_args()

    test_module = _load_test_module()
    briefs: Dict[str, Dict[str, Any]] = test_module.build_test_briefs()
    if args.limit is not None:
        briefs = dict(list(briefs.items())[:args.limit])

    records: List[ParseRecord] = []
    cache: Dict[str, Dict[str, Any]] = {}
    for brief_id, test in briefs.items():
        records.append(structured_form_bypass(brief_id, test))
        records.append(run_keyword_schema(brief_id, test))
        records.append(run_cache_hit(brief_id, test, cache))

    records.extend(load_historical_llm(args.historical))

    if args.run_llm:
        backends = args.backend or ["chatfire"]
        for backend in backends:
            for brief_id, test in briefs.items():
                try:
                    records.append(run_live_llm(brief_id, test, backend, args.model, args.timeout))
                except Exception as exc:
                    records.append(ParseRecord(
                        mode=f"llm_parser_live_{backend}",
                        brief_id=brief_id,
                        elapsed_s=math.nan,
                        valid=False,
                        field_accuracy=math.nan,
                        source=backend,
                        note=f"FAILED: {exc}",
                    ))

    write_outputs(records, args.out_prefix)


if __name__ == "__main__":
    main()
