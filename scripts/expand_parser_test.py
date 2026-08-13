"""
Expand NL Parser test set from 15 to ~50 briefs.
Adds more diverse briefs: varied wording, partial info, city-not-in-text.
Appends to existing results.

Usage (in illm_pd conda env):
  cd d:\iLLM_PD_new
  set PYTHONPATH=.
  python scripts/expand_parser_test.py
"""
import sys, os, time, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.brief_parser import parse_design_brief

# ---- Additional test briefs (~35 more, bringing total to ~50) ----
NEW_BRIEFS = [
    # === VARIED WORDING (same ground truth as existing, different phrasing) ===
    {"id": "flex_word_1", "brief": "一条在东北寒冷地区的高速公路，特重交通，沥青路面加粒料基层，地基大约80MPa", "gt_city": "harbin", "gt_type": "flexible", "gt_traffic": "heavy"},
    {"id": "flex_word_2", "brief": "南方湿热地区城市主干道，中等交通量，半刚性基层，软基约60MPa", "gt_city": "guangzhou", "gt_type": "semi_rigid", "gt_traffic": "medium"},
    {"id": "flex_word_3", "brief": "Build an expressway in a cold region with heavy trucks, flexible pavement, subgrade around 100 MPa", "gt_city": "harbin", "gt_type": "flexible", "gt_traffic": "heavy"},
    {"id": "flex_word_4", "brief": "西北干旱区一级公路，轻交通，柔性路面，Esub大约120MPa，设计使用15年", "gt_city": "urumqi", "gt_type": "flexible", "gt_traffic": "light"},
    {"id": "flex_word_5", "brief": "Design a semi-rigid pavement for a medium-traffic highway in central China, subgrade ~90 MPa", "gt_city": "wuhan", "gt_type": "semi_rigid", "gt_traffic": "medium"},
    {"id": "flex_word_6", "brief": "重庆山区二级公路，重交通，半刚性路面，Esub约200MPa（岩石地基）", "gt_city": "chongqing", "gt_type": "semi_rigid", "gt_traffic": "heavy"},
    {"id": "flex_word_7", "brief": "东部沿海平原高速公路，特重交通，柔性路面，软土路基Esub=40MPa", "gt_city": "shanghai", "gt_type": "flexible", "gt_traffic": "extra_heavy"},

    # === NO CITY IN TEXT (parser must infer from other clues or default) ===
    {"id": "no_city_1", "brief": "Expressway in a cold northern region with permafrost concerns, heavy traffic, flexible pavement, subgrade 120 MPa", "gt_city": "harbin", "gt_type": "flexible"},
    {"id": "no_city_2", "brief": "Urban arterial in a hot humid southern climate, medium traffic, semi-rigid base, soft subgrade ~50 MPa", "gt_city": "guangzhou", "gt_type": "semi_rigid"},
    {"id": "no_city_3", "brief": "Highway in a high-altitude region with extreme temperature swings, light traffic, flexible, Esub=200 MPa", "gt_city": "lhasa", "gt_type": "flexible"},
    {"id": "no_city_4", "brief": "Coastal highway with typhoon exposure, heavy traffic, semi-rigid, subgrade ~70 MPa", "gt_city": "fuzhou", "gt_type": "semi_rigid"},

    # === PARTIAL INFO (missing fields, ambiguous descriptions) ===
    {"id": "partial_1", "brief": "城市主干道，沥青路面。地基条件一般。", "gt_type": "flexible"},
    {"id": "partial_2", "brief": "Heavy traffic highway. Semi-rigid base. That's all I know.", "gt_type": "semi_rigid"},
    {"id": "partial_3", "brief": "一条路，在北方，交通量比较大，其他信息暂时没有。", "gt_city": "beijing"},
    {"id": "partial_4", "brief": "Rural road, very light traffic, flexible, local gravel subgrade", "gt_type": "flexible", "gt_traffic": "light"},
    {"id": "partial_5", "brief": "Expressway near a major port city. Container truck traffic (very heavy).", "gt_city": "shanghai", "gt_traffic": "extra_heavy"},

    # === EDGE CASES ===
    {"id": "edge_1", "brief": "Design for 50-year life. Climate zone 6. Full-depth asphalt on stabilized subgrade 300MPa.", "gt_type": "flexible"},
    {"id": "edge_2", "brief": "Temporary construction access road, 2 year design life, flexible, any subgrade, minimum cost",
     "gt_type": "flexible"},
    {"id": "edge_3", "brief": "Airport taxiway pavement. Heavy aircraft loading. Semi-rigid base over treated subgrade 150MPa.",
     "gt_type": "semi_rigid"},
    {"id": "edge_4", "brief": "Renovate an existing highway. Mill 5cm and overlay. Current structure: 15cm AC + 30cm granular base + semi-infinite subgrade ~80MPa.",
     "gt_type": "flexible"},
    {"id": "edge_5", "brief": "Build a parking lot for heavy trucks. Asphalt surface over aggregate base. Subgrade is compacted fill ~60MPa.",
     "gt_type": "flexible"},
    {"id": "edge_6", "brief": "Pavement for a bus rapid transit (BRT) lane. Frequent stops, channelised loading, 1000 buses/day.",
     "gt_type": "semi_rigid"},

    # === REALISTIC PROJECT DESCRIPTIONS (mixed completeness) ===
    {"id": "real_1", "brief": "项目位于华北平原，设计为双向六车道高速公路，设计年限20年，累计当量轴次约2.5×10^7次。初步钻探显示路基为粉质粘土，回弹模量约70MPa。拟采用半刚性基层沥青路面。",
     "gt_city": "beijing", "gt_type": "semi_rigid", "gt_traffic": "extra_heavy"},
    {"id": "real_2", "brief": "Southern China, 4-lane expressway, 15-yr design, ~15M ESAL. Silty-clay subgrade. Consider both flexible (granular base) and semi-rigid (CTB) options.",
     "gt_city": "guangzhou", "gt_type": "flexible", "gt_traffic": "heavy"},
    {"id": "real_3", "brief": "西北某省会城市外环路，一级公路标准，设计车速80km/h。路面结构拟采用4cm AC-13 + 6cm AC-20 + 8cm AC-25 + 20cm水泥稳定碎石基层 + 20cm级配碎石底基层。路基为黄土，Esub=50MPa。",
     "gt_city": "xian", "gt_type": "semi_rigid", "gt_traffic": "heavy"},
    {"id": "real_4", "brief": "Industrial park internal road. Occasional heavy trucks (5%). Mostly passenger cars. Design for 10 years. Local practice is flexible pavement. Subgrade CBR=5 (~50MPa).",
     "gt_city": "nanjing", "gt_type": "flexible", "gt_traffic": "medium"},
    {"id": "real_5", "brief": "川西高原国道改建项目，海拔3500m，年均气温5°C，极端低温-25°C。设计年限15年，交通量为中等（约5000 AADT，15%货车）。路基为碎石土，Esub约120MPa。",
     "gt_city": "chengdu", "gt_type": "flexible", "gt_traffic": "medium"},

    # === DELIBERATELY AMBIGUOUS ===
    {"id": "ambig_1", "brief": "Grade-separated interchange on a major highway. Design the approach pavement.", "gt_type": "flexible"},
    {"id": "ambig_2", "brief": "Pavement renewal project. Existing road has severe rutting and fatigue cracking after 8 years. Fix it.",
     "gt_type": "flexible"},
    {"id": "ambig_3", "brief": "Low-volume road in a developing region. Budget is very limited. Use locally available materials.",
     "gt_type": "flexible"},
]


def main():
    out_dir = Path(__file__).parent / "analysis"
    results = []

    print(f"Expanding parser test: {len(NEW_BRIEFS)} new briefs")
    print(f"Each brief calls GPT-4o-mini once (~6-8 sec)")
    print(f"Estimated time: {len(NEW_BRIEFS) * 8 // 60} min")
    print()

    for i, tc in enumerate(NEW_BRIEFS):
        print(f"[{i+1}/{len(NEW_BRIEFS)}] {tc['id']}: {tc['brief'][:60]}...")
        start = time.time()
        result = parse_design_brief(tc["brief"])
        elapsed = time.time() - start

        gt = {k: v for k, v in tc.items() if k.startswith("gt_")}
        checks = {}
        for field in ["city", "road_class", "traffic_level", "pavement_type"]:
            pred = getattr(result, field)
            truth = tc.get(f"gt_{field}" if field != "pavement_type" else "gt_type")
            if truth:
                checks[field] = {"predicted": pred, "ground_truth": truth, "match": pred == truth}
                status = "OK" if pred == truth else f"MISMATCH({pred} vs {truth})"
                print(f"  {field}: {status}")

        print(f"  E_sub={result.E_subgrade:.0f} conf={result.confidence:.2f} ({elapsed:.1f}s)")

        results.append({
            "id": tc["id"],
            "brief": tc["brief"],
            "city": result.city,
            "road_class": result.road_class,
            "traffic_level": result.traffic_level,
            "pavement_type": result.pavement_type,
            "E_subgrade": result.E_subgrade,
            "confidence": result.confidence,
            "elapsed_s": round(elapsed, 1),
            "checks": checks,
        })

    # Save
    out_path = out_dir / "parser_expanded_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Summary stats
    n_fields = sum(len(r["checks"]) for r in results)
    n_correct = sum(sum(1 for c in r["checks"].values() if c.get("match", False)) for r in results)

    print()
    print(f"=== RESULTS ===")
    print(f"Briefs: {len(results)}")
    print(f"Field checks: {n_fields}")
    print(f"Correct: {n_correct} ({100*n_correct/max(n_fields,1):.0f}%)")
    print(f"Saved: {out_path}")

    # Merge with existing 15-brief results
    old_path = out_dir / "nl_parser_results.json"
    if old_path.exists():
        with open(old_path) as f:
            old = json.load(f)
        print(f"Existing results: {len(old.get('suite_a', []))} briefs from Suite A")

    print()
    print("Next: combine old (15) + new ({}) = {} total briefs".format(
        len(results), len(results) + (len(old.get("suite_a", [])) if old_path.exists() else 15)))
    print("Update Supplementary S11 with combined numbers.")


if __name__ == "__main__":
    main()
