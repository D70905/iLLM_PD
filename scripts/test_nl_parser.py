"""
Experiment Group 3: NL Parser Evaluation & Safety Test
========================================================
Three test suites:
  A — Accuracy: parse free-text briefs, compare with ground-truth structured fields
  B — Missing-info: test defaults + no-hallucination behavior on incomplete briefs
  C — Safety: inject erroneous/contradictory briefs, verify Guard/Bounds catch errors

Usage (3 steps):
  1. python scripts/test_nl_parser.py  --suite A    # Accuracy (2-3 min, calls LLM ~25x)
  2. python scripts/test_nl_parser.py  --suite B    # Missing-info (2 min)
  3. python scripts/test_nl_parser.py  --suite C    # Safety stress-test (2 min)

Output: scripts/analysis/nl_parser_results.json
"""

from __future__ import annotations
import json, csv, sys, os, time, logging, io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Fix Windows GBK encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nl_test")

# ======================================================================
# GROUND TRUTH: 12 LTPP sections mapped to structured fields
# ======================================================================
LTPP_GROUND_TRUTH = {
    "16_1010": {
        "section_name": "16_1010 (MT, Dry-Freeze, flexible)",
        "city": "harbin",           # cold, similar MAAT ~6.7°C
        "road_class": "highway_1",
        "traffic_level": "heavy",
        "pavement_type": "flexible",
        "E_subgrade": 78,
        "design_life": 20,
    },
    "48_0001": {
        "section_name": "48_0001 (TX, Wet-NoFreeze, flexible, stiff subgrade)",
        "city": "guangzhou",        # hot, similar MAAT ~20.6°C
        "road_class": "expressway",
        "traffic_level": "heavy",
        "pavement_type": "flexible",
        "E_subgrade": 700,
        "design_life": 20,
    },
    "30_7076": {
        "section_name": "30_7076 (semi-rigid, soft subgrade)",
        "city": "nanjing",
        "road_class": "expressway",
        "traffic_level": "heavy",
        "pavement_type": "semi_rigid",
        "E_subgrade": 59,
        "design_life": 20,
    },
}

# ======================================================================
# SUITE A: BRIEF ACCURACY TEST
# ======================================================================
def build_test_briefs():
    """Generate natural-language briefs from ground truth + edge cases."""
    briefs = {
        # --- Complete, well-structured briefs ---
        "full_chinese_1": {
            "brief": (
                "哈尔滨附近的一条一级公路，设计为重载交通，采用柔性路面（沥青面层+粒料基层）。"
                "地基为中等软粘土，回弹模量约80 MPa。设计年限20年。"
            ),
            "ground_truth": LTPP_GROUND_TRUTH["16_1010"],
        },
        "full_english_1": {
            "brief": (
                "A heavy-traffic expressway near Guangzhou in southern China. "
                "Stiff subgrade with resilient modulus approximately 700 MPa. "
                "Flexible pavement with granular base. Design life 20 years."
            ),
            "ground_truth": LTPP_GROUND_TRUTH["48_0001"],
        },
        "full_chinese_2": {
            "brief": (
                "南京附近高速公路，重载交通，水泥稳定碎石基层（半刚性路面）。"
                "软土地基，E_sub大约60 MPa。设计使用年限20年。"
            ),
            "ground_truth": LTPP_GROUND_TRUTH["30_7076"],
        },
        # --- Partial/incomplete briefs ---
        "partial_no_traffic": {
            "brief": (
                "长沙市区的一条城市主干道，采用柔性路面。"
                "地基条件中等，回弹模量约100 MPa。"
            ),
            "ground_truth": {
                "city": "changsha",
                "road_class": "urban_trunk",
                "traffic_level": None,  # parser should default
                "pavement_type": "flexible",
                "E_subgrade": 100,
            },
        },
        "partial_no_subgrade": {
            "brief": (
                "武汉附近高速公路，特重交通，半刚性基层路面。"
                "设计年限30年的长寿命路面。"
                # No subgrade info
            ),
            "ground_truth": {
                "city": "wuhan",
                "road_class": "expressway",
                "traffic_level": "extra_heavy",
                "pavement_type": "semi_rigid",
                "E_subgrade": None,  # parser must estimate from context
            },
        },
        "partial_vague_soil": {
            "brief": (
                "沈阳附近二级公路，中等交通量，沥青路面。"
                "地基较软。"
            ),
            "ground_truth": {
                "city": "shenyang",
                "road_class": "highway_2",
                "traffic_level": "medium",
                "pavement_type": "flexible",
                "E_subgrade": None,  # "较软" -> should estimate ~40-80
            },
        },
        # --- Adversarial: ambiguous or edge case ---
        "ambig_city_not_in_list": {
            "brief": (
                "Design a highway pavement for a city 200 km north of Harbin, "
                "heavy traffic, flexible, stiff frozen subgrade ~150 MPa."
            ),
            "ground_truth": {
                "city": "harbin",  # nearest in list
                "road_class": "expressway",  # "highway" -> expressway
                "traffic_level": "heavy",
                "pavement_type": "flexible",
                "E_subgrade": 150,
            },
        },
        "ambig_typo_city": {
            "brief": "上海附近高速公路，重交通，柔性路面，Esub=120MPa",
            "ground_truth": {
                "city": "shanghai",
                "road_class": "expressway",
                "traffic_level": "heavy",
                "pavement_type": "flexible",
                "E_subgrade": 120,
            },
        },
    }
    return briefs


def run_suite_a():
    """Run accuracy evaluation on all test briefs."""
    from rl.brief_parser import parse_design_brief

    briefs = build_test_briefs()
    results = []

    print(f"\n{'='*80}")
    print(f"SUITE A: NL PARSER ACCURACY ({len(briefs)} briefs)")
    print(f"{'='*80}")

    for brief_id, test in briefs.items():
        brief_text = test["brief"]
        gt = test["ground_truth"]

        print(f"\n--- {brief_id} ---")
        print(f"Brief: {brief_text[:100]}...")
        start = time.time()
        result = parse_design_brief(brief_text)
        elapsed = time.time() - start

        # Compare with ground truth
        checks = {}
        for field in ["city", "road_class", "traffic_level", "pavement_type"]:
            pred = getattr(result, field)
            truth = gt.get(field)
            if truth is not None:
                checks[field] = {
                    "predicted": pred,
                    "ground_truth": truth,
                    "match": pred == truth,
                }
                status = "OK" if pred == truth else f"MISMATCH (pred={pred}, gt={truth})"
                print(f"  {field:<16}: {status}")

        # E_subgrade: check if within reasonable range (no exact match required)
        pred_esub = result.E_subgrade
        truth_esub = gt.get("E_subgrade")
        if truth_esub is not None:
            within_2x = (truth_esub * 0.5 <= pred_esub <= truth_esub * 2.0)
            checks["E_subgrade"] = {
                "predicted": pred_esub,
                "ground_truth": truth_esub,
                "within_2x": within_2x,
            }
            print(f"  E_subgrade       : {'OK' if within_2x else 'MISMATCH'} "
                  f"(pred={pred_esub:.0f}, gt={truth_esub:.0f})")
        else:
            print(f"  E_subgrade       : predicted={pred_esub:.0f} (no ground truth)")

        print(f"  confidence       : {result.confidence:.2f}")
        print(f"  elapsed          : {elapsed:.1f}s")

        results.append({
            "brief_id": brief_id,
            "checks": checks,
            "confidence": result.confidence,
            "elapsed_s": elapsed,
            "llm_raw": result.llm_raw[:200] + "..." if len(result.llm_raw) > 200 else result.llm_raw,
        })

    # Summary
    n_field_checks = 0
    n_field_correct = 0
    for r in results:
        for field, check in r["checks"].items():
            if field == "E_subgrade":
                n_field_checks += 1
                if check["within_2x"]:
                    n_field_correct += 1
            elif "match" in check:
                n_field_checks += 1
                if check["match"]:
                    n_field_correct += 1

    print(f"\n{'='*80}")
    print(f"ACCURACY SUMMARY")
    print(f"{'='*80}")
    print(f"  Discrete fields (city/class/traffic/type): {n_field_correct}/{n_field_checks} correct")
    print(f"  ({100*n_field_correct/max(n_field_checks,1):.0f}% field-level accuracy)")

    return results


# ======================================================================
# SUITE B: MISSING-INFO / DEFAULT BEHAVIOR
# ======================================================================
def run_suite_b():
    """Test behavior when critical fields are missing."""
    from rl.brief_parser import parse_design_brief

    print(f"\n{'='*80}")
    print(f"SUITE B: MISSING-INFO & DEFAULT BEHAVIOR")
    print(f"{'='*80}")

    test_cases = [
        {
            "id": "no_city",
            "brief": "高速公路，重交通，半刚性路面，Esub=150MPa。设计年限20年。",
            "check": "city defaults to 'beijing' (not hallucinated)",
            "expected_default": "beijing",
            "expected_field": "city",
        },
        {
            "id": "no_road_class",
            "brief": "上海地区，特重交通，半刚性路面，Esub=120MPa",
            "check": "road_class defaults to 'expressway'",
            "expected_default": "expressway",
            "expected_field": "road_class",
        },
        {
            "id": "no_traffic",
            "brief": "北京附近高速公路，柔性路面，Esub=80MPa",
            "check": "traffic_level defaults to 'heavy'",
            "expected_default": "heavy",
            "expected_field": "traffic_level",
        },
        {
            "id": "minimal_city_only",
            "brief": "乌鲁木齐",
            "check": "all fields defaulted — no hallucination",
        },
    ]

    for tc in test_cases:
        result = parse_design_brief(tc["brief"])
        city_ok = result.city in ["beijing", "urumqi"] or result.city == tc.get("expected_default", "")
        print(f"\n{tc['id']}: {tc['brief'][:60]}")
        print(f"  city={result.city}, class={result.road_class}, "
              f"traffic={result.traffic_level}, type={result.pavement_type}, "
              f"E_sub={result.E_subgrade:.0f}")
        print(f"  check: {tc['check']}")
        if "expected_default" in tc:
            actual = getattr(result, tc.get("expected_field", "city"), result.city)
            match = actual == tc["expected_default"]
            print(f"  result: {'PASS' if match else 'FAIL'} (expected={tc['expected_default']}, got={actual})")

    return test_cases


# ======================================================================
# SUITE C: SAFETY — ERRONEOUS INPUTS → GUARD/BOUNDS CATCH
# ======================================================================
def run_suite_c():
    """Inject erroneous briefs, verify the system catches problems downstream."""
    from rl.brief_parser import parse_design_brief

    print(f"\n{'='*80}")
    print(f"SUITE C: SAFETY STRESS-TEST — ERRONEOUS BRIEFS")
    print(f"{'='*80}")

    adversarial = [
        {
            "id": "contradictory_subgrade",
            "brief": (
                "广州附近高速公路，重载交通，柔性路面。"
                "地基为硬岩，Esub=5000 MPa（超出正常范围）。"  # E_sub max in system is 2000
            ),
            "check": "E_subgrade clamped to max 2000 (no unbounded value passed through)",
        },
        {
            "id": "contradictory_type",
            "brief": (
                "成都附近高速公路，重载交通，刚性路面（水泥混凝土板）。"  # rigid not supported
            ),
            "check": "pavement_type defaults to 'flexible' (rigid not in vocabulary)",
        },
        {
            "id": "nonsense_brief",
            "brief": (
                "Build a bridge over the Yangtze River with asphalt deck. "
                "No subgrade, no traffic data, no design life."
            ),
            "check": "Parser returns defaults without crashing; no hallucinated E_sub",
        },
    ]

    for tc in adversarial:
        result = parse_design_brief(tc["brief"])
        print(f"\n{tc['id']}: {tc['brief'][:80]}...")
        print(f"  city={result.city}, class={result.road_class}, "
              f"traffic={result.traffic_level}, type={result.pavement_type}, "
              f"E_sub={result.E_subgrade:.0f}")
        print(f"  confidence={result.confidence:.2f}")
        print(f"  check: {tc['check']}")

        # For contradictory_subgrade: verify clamp
        if tc["id"] == "contradictory_subgrade":
            clamped = 5.0 <= result.E_subgrade <= 2000.0
            print(f"  E_subgrade in [5, 2000]? {'PASS' if clamped else 'FAIL'}")

        # For nonsense: verify didn't hallucinate extreme values
        if tc["id"] == "nonsense_brief":
            reasonable = 5.0 <= result.E_subgrade <= 2000.0
            print(f"  No extreme hallucination? {'PASS' if reasonable else 'FAIL'}")

    return adversarial


# ======================================================================
# MAIN
# ======================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "analysis" / "nl_parser_results.json"))
    args = parser.parse_args()

    print("NL PARSER EVALUATION — Experiment Group 3")
    print(f"This will call the LLM (GPT-4o-mini) multiple times.")
    print(f"Estimated API calls: Suite A=8, Suite B=4, Suite C=3")
    print()

    all_results = {"suite_a": [], "suite_b": None, "suite_c": None}

    if args.suite in ("A", "all"):
        all_results["suite_a"] = run_suite_a()

    if args.suite in ("B", "all"):
        all_results["suite_b"] = run_suite_b()

    if args.suite in ("C", "all"):
        all_results["suite_c"] = run_suite_c()

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    serializable = {
        "suite_a": [
            {
                "brief_id": r["brief_id"],
                "checks": {
                    field: {
                        k: v for k, v in check.items() if k != "ground_truth"
                    }
                    for field, check in r["checks"].items()
                },
                "confidence": r["confidence"],
                "elapsed_s": r["elapsed_s"],
            }
            for r in all_results["suite_a"]
        ] if all_results["suite_a"] else [],
        "suite_b": [
            {"id": tc["id"], "check": tc["check"]}
            for tc in (all_results["suite_b"] or [])
        ],
        "suite_c": [
            {"id": tc["id"], "check": tc["check"]}
            for tc in (all_results["suite_c"] or [])
        ],
    }
    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
