"""
calibrate_evaluator_threshold.py
=================================
从历史审计链 JSONL 中提取 (eval_score, margins, feasible) 对，
分析 eval_score 与实际合规性的相关性，标定 Evaluator 否决阈值。

数据来源: output/rl_runs/ablation_full_*_seed*/audit/audit_chain.jsonl
只使用解析成功的 evaluator 记录（reasoning != '[unparseable]'）。

输出:
  1. eval_score 分布 (parseable vs unparseable)
  2. 各分数段的实际不合规率 (feasible=False 或 最小 margin < 1)
  3. 不同阈值下的召回/误伤率 (ROC-like)
  4. 推荐阈值
"""

import json, os, sys
import numpy as np
from collections import defaultdict

BASE = "output/rl_runs"

# ── 收集所有可用的 ablation_full 运行 ──
RUNS = []
for d in sorted(os.listdir(BASE)):
    dpath = os.path.join(BASE, d)
    if not os.path.isdir(dpath):
        continue
    # 只用有 Evaluator 且训练完成的运行
    audit_path = os.path.join(dpath, "audit", "audit_chain.jsonl")
    flag_path = os.path.join(dpath, "training_complete.flag")
    if os.path.exists(audit_path) and os.path.exists(flag_path):
        # 排除 no-evaluator 运行
        if "no-evaluator" not in d:
            RUNS.append((d, audit_path))

print(f"Found {len(RUNS)} completed runs with Evaluator enabled")
print()

# ── 提取数据 ──
# 从审计链中提取 evaluator 记录（有 parseable score）和对应的 step 记录
all_pairs = []  # (eval_score, is_feasible, min_margin, section_name)

for run_name, audit_path in RUNS:
    # 第一遍: 收集所有 step 记录，按 (episode, step) 索引
    step_map = {}  # (episode, step) -> {feasible, margins, ...}

    with open(audit_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "step":
                d = r.get("data", {})
                ep = d.get("episode")
                st = d.get("step")
                if ep is not None and st is not None:
                    step_map[(int(ep), int(st))] = {
                        "feasible": d.get("feasible"),
                        "margins": d.get("margins", {}),
                        "critical": d.get("critical"),
                    }

    # 第二遍: 收集 parseable evaluator 记录
    eval_count = 0
    parseable_count = 0
    with open(audit_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "evaluator":
                continue
            eval_count += 1

            d = r.get("data", {})
            reason = d.get("reasoning", "")
            score = d.get("score")
            ep = d.get("episode")
            st = d.get("step")

            # 跳过 unparseable
            if reason == "[unparseable]" or score is None:
                continue

            parseable_count += 1

            # 关联到对应 step
            key = (int(ep), int(st)) if ep is not None and st is not None else None
            step_info = step_map.get(key) if key else None

            if step_info is None:
                continue

            # 判断是否真的不合规
            feasible = step_info.get("feasible", True)
            margins = step_info.get("margins", {})

            # 计算最小 margin
            min_margin = float("inf")
            for k, v in margins.items():
                if v is not None and v > 0:
                    min_margin = min(min_margin, float(v))
            if min_margin == float("inf"):
                min_margin = 1.0  # 无 margin 数据，假设合规

            is_compliant = bool(feasible) and min_margin >= 1.0

            all_pairs.append({
                "run": run_name[:40],
                "score": float(score),
                "feasible": bool(feasible),
                "min_margin": min_margin,
                "is_compliant": is_compliant,
                "margins": {k: round(float(v), 2) for k, v in margins.items() if v is not None},
            })

    pct = 100 * parseable_count / max(eval_count, 1)
    print(f"  {run_name:<50s}: {eval_count:>5d} evaluator, {parseable_count:>5d} parseable ({pct:.0f}%)")

print(f"\nTotal parseable (score, margins) pairs: {len(all_pairs)}")

if len(all_pairs) == 0:
    print("\nERROR: No parseable evaluator records found!")
    sys.exit(1)

# ── 分析 ──
scores = np.array([p["score"] for p in all_pairs])
compliant = np.array([p["is_compliant"] for p in all_pairs])

print(f"\n{'='*70}")
print(f"SCORE DISTRIBUTION (parseable records only)")
print(f"{'='*70}")
print(f"  N = {len(scores)}")
print(f"  Mean score = {scores.mean():.2f}")
print(f"  Std = {scores.std():.2f}")
print(f"  Compliant rate overall = {compliant.mean()*100:.1f}%")

# 各分数段的实际不合规率
print(f"\n{'='*70}")
print(f"COMPLIANCE RATE BY SCORE BIN")
print(f"{'='*70}")
print(f"  {'Score':>8s}  {'N':>6s}  {'Compliant':>10s}  {'Non-compliant':>14s}")
print(f"  {'-'*50}")

for lo in range(0, 10):
    hi = lo + 1
    mask = (scores >= lo) & (scores < hi)
    n = mask.sum()
    if n == 0:
        continue
    comp = compliant[mask].mean()
    noncomp = 1 - comp
    bar = "#" * int(noncomp * 50)
    print(f"  [{lo}-{hi})  {n:>6d}  {comp*100:>9.1f}%  {noncomp*100:>13.1f}%  {bar}")

# ── 阈值分析 (ROC-like) ──
print(f"\n{'='*70}")
print(f"THRESHOLD CALIBRATION: at each threshold T,")
print(f"  'Reject' = score <= T")
print(f"  'Recall' = % of actually-noncompliant that are rejected")
print(f"  'Precision' = % of rejected that are actually-noncompliant")
print(f"  'False-positive rate' = % of actually-compliant that are rejected")
print(f"{'='*70}")
print(f"  {'Thr':>5s}  {'Reject%':>8s}  {'Recall':>8s}  {'Precision':>10s}  {'FPR':>8s}  {'F1':>8s}")
print(f"  {'-'*60}")

best_threshold = None
best_f1 = 0

for thr in np.arange(0.5, 9.5, 0.5):
    rejected = scores <= thr
    n_reject = rejected.sum()
    reject_rate = n_reject / len(scores) * 100

    # 真正不合规的
    n_actual_bad = (~compliant).sum()
    n_reject_bad = ((~compliant) & rejected).sum()
    recall = n_reject_bad / max(n_actual_bad, 1) * 100

    # 拒绝中的不合规比例
    precision = n_reject_bad / max(n_reject, 1) * 100

    # 误伤: 合规但被拒绝
    n_actual_good = compliant.sum()
    n_reject_good = (compliant & rejected).sum()
    fpr = n_reject_good / max(n_actual_good, 1) * 100

    # F1 score
    f1 = 2 * recall * precision / max(recall + precision, 1)

    if n_reject > 0 and f1 > best_f1:
        best_f1 = f1
        best_threshold = thr

    marker = " <--" if thr == best_threshold else ""
    print(f"  {thr:>5.1f}  {reject_rate:>7.1f}%  {recall:>7.1f}%  {precision:>9.1f}%  {fpr:>7.1f}%  {f1:>7.1f}%{marker}")

# ── 汇总 ──
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

if best_threshold:
    rejected_best = scores <= best_threshold
    n_reject_best = rejected_best.sum()
    n_bad_reject = ((~compliant) & rejected_best).sum()
    n_actual_bad = (~compliant).sum()

    print(f"  Recommended threshold: score <= {best_threshold:.1f}")
    print(f"  Rejection rate: {n_reject_best}/{len(scores)} ({100*n_reject_best/len(scores):.1f}%)")
    print(f"  Recall: {n_bad_reject}/{n_actual_bad} non-compliant caught ({100*n_bad_reject/max(n_actual_bad,1):.1f}%)")
    print(f"  Best F1: {best_f1:.1f}%")

    # 判断 Evaluator 是否有判别力
    # 简单检验: eval_score 与 is_compliant 的 Spearman 相关
    from scipy.stats import spearmanr
    rho, pval = spearmanr(scores, compliant)
    print(f"\n  Spearman rho(eval_score, compliant) = {rho:.3f} (p={pval:.4f})")

    if abs(rho) < 0.1:
        print(f"  WARNING: eval_score and compliance are essentially UNCORRELATED.")
        print(f"  The Evaluator's score has NO meaningful relationship with actual design quality.")
        print(f"  A veto gate would be ARBITRARY — fix the Evaluator's scoring first.")
    elif abs(rho) < 0.2:
        print(f"  CAUTION: eval_score and compliance are WEAKLY correlated.")
        print(f"  The veto gate may have marginal utility; proceed with caution.")
    else:
        print(f"  OK: eval_score and compliance show meaningful correlation.")
        print(f"  A veto gate at threshold {best_threshold:.1f} is justified.")

    # 额外: 对比 default-5.0 的被拒绝率
    default_mask = np.isclose(scores, 5.0, atol=0.01)
    n_default = default_mask.sum()
    n_default_bad = ((~compliant) & default_mask).sum()
    print(f"\n  For reference: score=5.0 (default/unparseable):")
    print(f"    N={n_default}, non-compliant rate={100*n_default_bad/max(n_default,1):.1f}%")
else:
    print("  No evaluator records with sufficient data to calibrate threshold.")
    print("  Fix the JSON parsing first, then re-run with a fresh evaluator-enabled training batch.")