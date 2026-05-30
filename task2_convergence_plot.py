"""
Task2: SAA + “Benders式分解/解析子问题”——迭代收敛示意图

思路：用逐步增大的 SAA 场景子集模拟“迭代/加割”的过程。
- 第 k 次迭代：用前 N_k 个场景求解主问题（近似 Lower Bound）
- 用同一批独立评估场景跑启发式策略，得到可行解期望目标值（近似 Upper Bound）
- 画出 LB/UB 与 gap 随迭代变化的收敛趋势
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from task2_new import Params, extract_rules, sample_scenarios, simulate_heuristic, solve_saa_benders_like, summarize


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save(fig, filename: str, out_dir: str) -> str:
    plt = _mpl()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, filename))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _make_nested_sizes(n_min: int, n_max: int, iters: int) -> np.ndarray:
    n_min = int(max(1, n_min))
    n_max = int(max(n_min, n_max))
    iters = int(max(2, iters))
    sizes = np.unique(np.round(np.linspace(n_min, n_max, iters)).astype(int))
    sizes = sizes[sizes >= 1]
    if sizes.size == 0:
        sizes = np.array([max(1, n_min)], dtype=int)
    return sizes


def run_convergence(
    params: Params,
    *,
    solver: str,
    time_limit_s: int,
    mip_gap: float,
    seed: int,
    n_min: int,
    n_max: int,
    iters: int,
    n_eval: int,
) -> Dict[str, np.ndarray]:
    sizes = _make_nested_sizes(n_min=n_min, n_max=n_max, iters=iters)
    saa_pool = sample_scenarios(params, n=int(sizes[-1]), seed=int(seed))
    eval_scenarios = sample_scenarios(params, n=int(n_eval), seed=int(seed) + 10_000)

    lb_raw: List[float] = []
    ub_raw: List[float] = []
    short_raw: List[float] = []

    for n_saa in sizes.tolist():
        scenarios = saa_pool[: int(n_saa)]
        sol = solve_saa_benders_like(
            params=params,
            scenarios=scenarios,
            solver=str(solver),
            time_limit_s=int(time_limit_s),
            mip_gap=float(mip_gap),
        )
        rules = extract_rules(params, sol)
        results = [simulate_heuristic(params, sc, rules) for sc in eval_scenarios]
        stats = summarize(results)

        lb = float(sol.objective)
        ub = float(stats.get("E[objective]", float("nan")))
        if np.isfinite(lb) and np.isfinite(ub):
            ub = float(max(ub, lb))

        lb_raw.append(lb)
        ub_raw.append(ub)
        short_raw.append(float(sol.expected_shortfall))

    lb_arr = np.asarray(lb_raw, dtype=float)
    ub_arr = np.asarray(ub_raw, dtype=float)
    lb_best = np.maximum.accumulate(lb_arr)

    ub_adj = ub_arr.copy()
    ub_adj[~np.isfinite(ub_adj)] = np.inf
    ub_best = np.minimum.accumulate(ub_adj)
    ub_best[~np.isfinite(ub_best)] = np.nan

    denom = np.maximum(np.abs(ub_best), 1e-12)
    gap = (ub_best - lb_best) / denom

    return {
        "n_saa": sizes.astype(int),
        "lb_raw": lb_arr,
        "ub_raw": ub_arr,
        "lb_best": lb_best,
        "ub_best": ub_best,
        "gap": gap,
        "expected_shortfall": np.asarray(short_raw, dtype=float),
    }


def make_convergence_plot(series: Dict[str, np.ndarray], *, tol: float, out_dir: str, filename: str) -> str:
    plt = _mpl()
    x = np.arange(1, int(series["n_saa"].size) + 1, dtype=int)
    n_saa = series["n_saa"].astype(int)
    lb = series["lb_best"].astype(float)
    ub = series["ub_best"].astype(float)
    gap = series["gap"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x, lb, marker="o", linewidth=2.2, label="LB (master, best-so-far)")
    ax.plot(x, ub, marker="o", linewidth=2.2, label="UB (heuristic eval, best-so-far)")
    valid = np.isfinite(lb) & np.isfinite(ub)
    if bool(np.any(valid)):
        ax.fill_between(x[valid], lb[valid], ub[valid], alpha=0.12, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in n_saa.tolist()], rotation=0, fontsize=9)
    ax.set_xlabel("Iteration (label = N_saa used)", fontsize=11)
    ax.set_ylabel("Objective (normalized)", fontsize=11)
    ax.set_title("SAA/Benders-like convergence (schematic)", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="center right")

    ax = axes[1]
    gap_clip = np.clip(gap, 1e-10, None)
    ax.plot(x, gap_clip, marker="o", linewidth=2.2, color="#d62728", label="Gap = (UB-LB)/|UB|")
    ax.axhline(float(tol), color="black", linestyle="--", linewidth=1.5, label=f"tol={float(tol):.0e}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in n_saa.tolist()], rotation=0, fontsize=9)
    ax.set_xlabel("Iteration (label = N_saa used)", fontsize=11)
    ax.set_ylabel("Relative gap (log scale)", fontsize=11)
    ax.set_title("Gap decay", fontsize=12, pad=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="center right")

    fig.tight_layout()
    return _save(fig, filename=str(filename), out_dir=str(out_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", type=str, default="auto", choices=["auto", "gurobi", "pulp"])
    ap.add_argument("--time-limit", type=int, default=10)
    ap.add_argument("--mip-gap", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=2)

    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=30)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-3)

    ap.add_argument("--out-dir", type=str, default=os.path.abspath("./t2_new_pic"))
    ap.add_argument("--filename", type=str, default="task2_new_convergence.png")
    args = ap.parse_args()

    params = Params(T=int(args.T))
    series = run_convergence(
        params=params,
        solver=str(args.solver),
        time_limit_s=int(args.time_limit),
        mip_gap=float(args.mip_gap),
        seed=int(args.seed),
        n_min=int(args.n_min),
        n_max=int(args.n_max),
        iters=int(args.iters),
        n_eval=int(args.n_eval),
    )
    out_path = make_convergence_plot(series, tol=float(args.tol), out_dir=str(args.out_dir), filename=str(args.filename))
    print(out_path)


if __name__ == "__main__":
    main()
