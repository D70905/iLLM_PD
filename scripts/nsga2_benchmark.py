"""
scripts/nsga2_benchmark.py  —  pymoo NSGA-II Pareto 参考解集 (无标量化权重)
============================================================================
设计变量 x = [h1..h5(米), E1..E5(MPa)] (10 维), 与 env 动作空间一致。
上下界直接取自 rl.guards.GuardConfig.from_base_type (与 Guard 完全一致)。
搜索用 surrogate (快); 最终前沿设计再用全 ABAQUS FEA 验证 DSR>=1。

术语 (按本地 AI 建议): 这是“Pareto 参考解集”, 直接搜设计空间、不经 RL,
用于标定 RL 加权扫描得到的前沿是否完整 —— 不要叫它 method 的 baseline。

放置位置: scripts/nsga2_benchmark.py     依赖: pymoo, numpy
运行: python scripts/nsga2_benchmark.py --type flexible
"""
import os, csv, argparse
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from rl.guards import GuardConfig
from rl.mo_objectives import DesignCase, evaluate_design
from rl.carbon_factors import get_provider

OUT = "experiments/ltpp_data/deliverables/mo_pareto/"


class PavementMOO(ElementwiseProblem):
    # 注: pymoo>=0.6 用 n_ieq_constr; 旧版(0.5)用 n_constr
    def __init__(self, case: DesignCase, provider=None):
        cfg = GuardConfig.from_base_type(case.pavement_type)
        xl = np.array(list(cfg.h_min) + list(cfg.E_min), dtype=float)   # 10 维下界
        xu = np.array(list(cfg.h_max) + list(cfg.E_max), dtype=float)   # 10 维上界
        super().__init__(n_var=10, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)
        self.case = case
        self.provider = provider or get_provider("ec3", cache=True)

    def _evaluate(self, x, out, *args, **kwargs):
        obj = evaluate_design(x, case=self.case, provider=self.provider)  # surrogate, 快
        out["F"] = [obj.lcc, obj.gwp]      # 最小化 成本(USD/m²), 碳(kgCO2e/m²)
        out["G"] = [1.0 - obj.dsr]         # <=0 即合规 (dsr>=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="flexible", choices=["flexible", "semi_rigid"])
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--E_subgrade", type=float, default=60.0)
    ap.add_argument("--ckpt", default="output/surrogate_model/surrogate_v3.pt")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    case = DesignCase(pavement_type=a.type, E_subgrade=a.E_subgrade, surrogate_ckpt=a.ckpt)
    prob = PavementMOO(case)
    res = minimize(prob, NSGA2(pop_size=a.pop), ("n_gen", a.gen), seed=1, verbose=True)

    path = os.path.join(OUT, f"mo_pareto_nsga2_{a.type}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lcc_usd_m2", "gwp_kgco2e_m2", "thickness_m", "modulus_MPa"])
        for X, F in zip(np.atleast_2d(res.X), np.atleast_2d(res.F)):
            w.writerow([float(F[0]), float(F[1]),
                        list(np.round(X[:5], 4)), list(np.round(X[5:10], 1))])
    n = len(np.atleast_2d(res.F))
    print("saved", path, f"({n} front points)")
    print("提醒: 对前沿设计 res.X 再用全 ABAQUS FEA 验证 DSR>=1。")


if __name__ == "__main__":
    main()
