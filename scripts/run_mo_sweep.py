"""
scripts/run_mo_sweep.py  —  加权 PPO 的 λ 扫描 -> RL Pareto 前沿
================================================================
对每个 λ: 用 reward_mo(λ) 配置环境 -> 训练 PPO -> 取所得设计 x=[h(米),E] -> 评估(LCC,GWP)
       -> 收集成点 -> 取非支配集。
保留你的 RL 叙事 (策略设定数值); 报整条前沿 (不报单一权重)。

放置位置: scripts/run_mo_sweep.py
运行: python scripts/run_mo_sweep.py --type flexible
"""
import os, csv, argparse
import numpy as np
from rl.mo_objectives import DesignCase, evaluate_design

OUT = "experiments/ltpp_data/deliverables/mo_pareto/"


def train_one(lam: float, case: DesignCase, P_barrier: float = 10.0,
              timesteps: int = 8000):
    """
    用 reward_mo(λ) 训一个策略, 返回其确定性 rollout 选出的设计 x=[h1..h5(米), E1..E5(MPa)]。
    TODO(接 train.py/env.py):
      1) 在 rl/env.py 的 step() 里, 把奖励改成:
           from rl.reward_mo import mo_reward, MORewardConfig
           from rl.mo_objectives import ObjResult, Normalizer
           obj = ObjResult(dsr=post['dsr'], feasible=feasible,
                           lcc=post['lcc']['NPV_total_usd_m2'], gwp=<碳后评估>)
           reward = mo_reward(obj, self._normalizer, MORewardConfig(self.lam, P_barrier))
         (碳后评估: 在 _compute_post_eval 里加 lca_gwp_for_design, 与 LCC 并列)
      2) 复用 rl/train.py 的 PPO(MlpPolicy) 训练; 训完做确定性 rollout 取 best design。
      3) self.lam 由本函数传入 (EnvConfig 增加 lam 字段, 或 set_lambda())。
    """
    raise NotImplementedError("接 train.py/env.py: 用 reward_mo(λ) 训练并返回所选设计 x(10维)")


def non_dominated(points):
    """points: list of (lcc, gwp, ...); 返回非支配(都最小化)下标。"""
    pts = np.array([[p[0], p[1]] for p in points], dtype=float)
    keep = []
    for i, a in enumerate(pts):
        dominated = np.any(np.all(pts <= a, axis=1) & np.any(pts < a, axis=1))
        if not dominated:
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="flexible", choices=["flexible", "semi_rigid"])
    ap.add_argument("--lams", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--E_subgrade", type=float, default=60.0)
    a = ap.parse_args()
    lams = [float(s) for s in a.lams.split(",")]
    os.makedirs(OUT, exist_ok=True)
    case = DesignCase(pavement_type=a.type, E_subgrade=a.E_subgrade)

    rows = []
    for lam in lams:
        x = train_one(lam, case)
        obj = evaluate_design(x, case=case)
        rows.append((obj.lcc, obj.gwp, lam, obj.dsr,
                     list(np.round(np.asarray(x[:5]), 4)),
                     list(np.round(np.asarray(x[5:10]), 1))))
    nd = set(non_dominated(rows))
    path = os.path.join(OUT, f"mo_pareto_rl_{a.type}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lcc_usd_m2", "gwp_kgco2e_m2", "lambda", "dsr",
                    "on_front", "thickness_m", "modulus_MPa"])
        for i, (lcc, gwp, lam, dsr, h, E) in enumerate(rows):
            w.writerow([lcc, gwp, lam, dsr, int(i in nd), h, E])
    print("saved", path)


if __name__ == "__main__":
    main()
