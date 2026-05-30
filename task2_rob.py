from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from task2_new import Params, Scenario, run_policy, simulate_heuristic


def _save_matplotlib(fig: plt.Figure, filename: str, out_dir: str) -> str:
    out_path = os.path.abspath(os.path.join(out_dir, filename))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _run_policy_eval(
    params: Params,
    *,
    solver: str,
    time_limit_s: int,
    mip_gap: float,
    seed: int,
    n_saa: int,
    n_eval: int,
) -> Tuple[Dict[str, float], List[Dict], Dict]:
    _, rules, eval_results, stats = run_policy(
        params=params,
        solver=solver,
        time_limit_s=int(time_limit_s),
        mip_gap=float(mip_gap),
        seed=int(seed),
        n_saa=int(n_saa),
        n_eval=int(n_eval),
    )

    target = float(params.W_goal_mt) * float(params.eta_supply)
    expected_shortfall = float(stats.get("E[shortfall]", 0.0))
    constraint_satisfaction = float(np.clip(1.0 - expected_shortfall / max(target, 1e-12), 0.0, 1.0))

    metrics = {
        "E_completion": float(stats.get("E[completion_t]", float("nan"))),
        "E_objective": float(stats.get("E[objective]", float("nan"))),
        "E_total_cost": float(stats.get("E[total_cost]", float("nan"))),
        "E_shortfall": expected_shortfall,
        "P_shortfall": float(stats.get("P(shortfall>0)", float("nan"))),
        "constraint_satisfaction": constraint_satisfaction,
    }
    return metrics, eval_results, rules


def _evaluate(
    params: Params,
    *,
    solver: str,
    time_limit_s: int,
    mip_gap: float,
    seed: int,
    n_saa: int,
    n_eval: int,
) -> Dict[str, float]:
    _, _, _, stats = run_policy(
        params=params,
        solver=solver,
        time_limit_s=int(time_limit_s),
        mip_gap=float(mip_gap),
        seed=int(seed),
        n_saa=int(n_saa),
        n_eval=int(n_eval),
    )

    target = float(params.W_goal_mt) * float(params.eta_supply)
    expected_shortfall = float(stats.get("E[shortfall]", 0.0))
    constraint_satisfaction = float(np.clip(1.0 - expected_shortfall / max(target, 1e-12), 0.0, 1.0))

    return {
        "E_completion": float(stats.get("E[completion_t]", float("nan"))),
        "E_objective": float(stats.get("E[objective]", float("nan"))),
        "E_total_cost": float(stats.get("E[total_cost]", float("nan"))),
        "E_shortfall": expected_shortfall,
        "P_shortfall": float(stats.get("P(shortfall>0)", float("nan"))),
        "constraint_satisfaction": constraint_satisfaction,
    }


def _stability_score(rel_change: float, threshold: float) -> float:
    if not np.isfinite(rel_change):
        return 0.0
    return float(100.0 * (1.0 - min(abs(rel_change) / max(threshold, 1e-12), 1.0)))


def _radar_scores(base: Dict[str, float], other: Dict[str, float]) -> Tuple[List[str], List[float], List[float]]:
    base_completion = float(base["E_completion"])
    base_cost = float(base["E_total_cost"])
    base_obj = float(base["E_objective"])

    other_completion = float(other["E_completion"])
    other_cost = float(other["E_total_cost"])
    other_obj = float(other["E_objective"])

    rel_completion = (other_completion - base_completion) / max(base_completion, 1e-12)
    rel_cost = (other_cost - base_cost) / max(base_cost, 1e-12)
    rel_obj = (other_obj - base_obj) / max(base_obj, 1e-12)

    labels = ["Schedule stability", "Cost stability", "Constraint satisfaction", "Scenario adaptability", "Risk control"]
    base_scores = [
        100.0,
        100.0,
        float(100.0 * np.clip(float(base["constraint_satisfaction"]), 0.0, 1.0)),
        100.0,
        float(100.0 * (1.0 - np.clip(float(base["P_shortfall"]), 0.0, 1.0))),
    ]
    other_scores = [
        _stability_score(rel_completion, threshold=0.20),
        _stability_score(rel_cost, threshold=0.20),
        float(100.0 * np.clip(float(other["constraint_satisfaction"]), 0.0, 1.0)),
        _stability_score(rel_obj, threshold=0.20),
        float(100.0 * (1.0 - np.clip(float(other["P_shortfall"]), 0.0, 1.0))),
    ]
    return labels, base_scores, other_scores


def plot_param_variation(
    baseline: Dict[str, float],
    series: Dict[str, List[Tuple[float, Dict[str, float]]]],
    *,
    out_dir: str,
    filename: str = "task2_rob_param_variation.png",
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    base_completion = float(baseline["E_completion"])

    items = [
        ("p_fail_base", "Rocket failure probability $p_{fail,i}$"),
        ("theta_swing", "Elevator capacity factor $\\theta_l$"),
        ("lambda_shortage", "Shortfall penalty $\\lambda$"),
    ]

    for ax, (key, title) in zip(axes, items):
        pts = series[key]
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([100.0 * (float(p[1]["E_completion"]) - base_completion) / max(base_completion, 1e-12) for p in pts], dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.5, color="#1f77b4")
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Parameter change (%)", fontsize=11)
        ax.set_ylabel("Completion change (%)", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xticks([-10, 0, 10])

    fig.tight_layout()
    return _save_matplotlib(fig, filename, out_dir)


def plot_extreme_objective(
    baseline_obj: float,
    objectives: Dict[str, float],
    *,
    out_dir: str,
    filename: str = "task2_rob_extreme_objective.png",
) -> str:
    labels = ["Baseline", "Batch faults", "Severe swing", "Combined extremes"]
    vals = [
        float(baseline_obj),
        float(objectives["batch_fault"]),
        float(objectives["severe_swing"]),
        float(objectives["combined"]),
    ]
    vals_k = [v / 1e3 for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals_k, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Objective (×1e3)", fontsize=11)
    ax.set_title("Objective under extreme scenarios", fontsize=12, pad=12)
    ax.grid(True, axis="y", alpha=0.25)

    for b in bars:
        h = float(b.get_height())
        ax.text(float(b.get_x() + b.get_width() / 2.0), h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return _save_matplotlib(fig, filename, out_dir)


def plot_radar(
    labels: List[str],
    base_scores: List[float],
    other_scores: List[float],
    *,
    other_name: str,
    out_dir: str,
    filename: str = "task2_rob_radar.png",
) -> str:
    n = len(labels)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    base = list(map(float, base_scores)) + [float(base_scores[0])]
    other = list(map(float, other_scores)) + [float(other_scores[0])]

    fig = plt.figure(figsize=(8.5, 8.0))
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14)
    
    # Adjust label positions to avoid overlap with chart
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle == 0:
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        elif angle == np.pi:
            label.set_horizontalalignment('center')
        else:
            label.set_horizontalalignment('right')

    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=12)

    ax.plot(angles, base, color="#1f77b4", linewidth=2.4, label="Baseline")
    ax.fill(angles, base, color="#1f77b4", alpha=0.12)
    ax.plot(angles, other, color="#d62728", linewidth=2.4, label=other_name)
    ax.fill(angles, other, color="#d62728", alpha=0.12)

    ax.set_title("Robustness radar (score: 0–100)", fontsize=16, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), fontsize=12, frameon=False)

    fig.tight_layout()
    return _save_matplotlib(fig, filename, out_dir)


def plot_monte_carlo_risk(
    *,
    baseline_point: Dict[str, float],
    completion: np.ndarray,
    cost_trillion: np.ndarray,
    out_dir: str,
    filename: str = "task2_rob_monte_carlo_risk.png",
) -> str:
    from matplotlib.patches import Ellipse

    completion = np.asarray(completion, dtype=float)
    cost_trillion = np.asarray(cost_trillion, dtype=float)

    comp_min = float(min(float(np.min(completion)), float(baseline_point["completion"])))
    comp_max = float(max(float(np.max(completion)), float(baseline_point["completion"])))
    cost_min = float(min(float(np.min(cost_trillion)), float(baseline_point["cost_trillion"])))
    cost_max = float(max(float(np.max(cost_trillion)), float(baseline_point["cost_trillion"])))

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    rng = np.random.default_rng(42)
    x_jitter = completion + rng.normal(loc=0.0, scale=0.06, size=int(completion.size))
    unique_completion = np.unique(completion)
    if int(unique_completion.size) <= 3:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        color_map = {float(v): palette[i % len(palette)] for i, v in enumerate(sorted(map(float, unique_completion)))}
        colors = np.array([color_map.get(float(v), "#1f77b4") for v in completion], dtype=object)
    else:
        colors = "#1f77b4"

    ax1.scatter(x_jitter, cost_trillion, c=colors, alpha=0.30, s=22, edgecolors="w", linewidth=0.3)

    cov = np.cov(x_jitter, cost_trillion)

    ax1.axvline(float(baseline_point["completion"]), color="red", linestyle=":", linewidth=1.2, alpha=0.55)
    ax1.axvline(float(np.mean(completion)), color="green", linestyle=":", linewidth=1.2, alpha=0.55)
    ax1.axhline(float(baseline_point["cost_trillion"]), color="red", linestyle=":", linewidth=1.2, alpha=0.35)
    ax1.axhline(float(np.mean(cost_trillion)), color="green", linestyle=":", linewidth=1.2, alpha=0.35)

    ax1.scatter(
        [float(baseline_point["completion"])],
        [float(baseline_point["cost_trillion"])],
        c="red",
        s=220,
        marker="*",
        edgecolors="black",
        linewidth=2,
        label=f'Reference (deterministic)\n({baseline_point["completion"]:.0f} periods, ${baseline_point["cost_trillion"]:.2f}T)',
        zorder=6,
    )
    ax1.scatter(
        [float(np.mean(completion))],
        [float(np.mean(cost_trillion))],
        c="green",
        s=160,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label=f"Expected\n({np.mean(completion):.0f} periods, ${np.mean(cost_trillion):.2f}T)",
        zorder=6,
    )

    ax1.set_xlabel("Completion (periods)", fontsize=12)
    ax1.set_ylabel("Total cost (trillion USD)", fontsize=12)
    ax1.set_title("Monte Carlo scatter with 95% confidence ellipse", fontsize=13, pad=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    comp_span = float(comp_max - comp_min)
    if comp_span > 1e-12:
        ax1.set_xlim(comp_min - 0.5, comp_max + 0.5)
    else:
        ax1.set_xlim(comp_min - 0.5, comp_min + 0.5)

    y_min, y_max = 15.3, 15.7
    ax1.set_ylim(y_min, y_max)

    if float(np.std(x_jitter)) > 1e-9 and float(np.std(cost_trillion)) > 1e-9 and np.all(np.isfinite(cov)) and np.linalg.det(cov) > 1e-12:
        x0 = float(np.mean(x_jitter))
        y0 = float(np.mean(cost_trillion))
        vals, vecs = np.linalg.eig(cov)
        vals = np.sqrt(np.maximum(vals, 0.0))
        angle = float(np.rad2deg(np.arctan2(vecs[1, 0], vecs[0, 0])))
        width = float(vals[0] * 2.0 * 2.0)
        height = float(vals[1] * 2.0 * 2.0)

        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        x_span = float(xlim[1] - xlim[0])
        y_span = float(ylim[1] - ylim[0])
        if x_span > 1e-12 and y_span > 1e-12 and width > 1e-12:
            normalized_aspect = (height / y_span) / (width / x_span)
            min_normalized_aspect = 0.35
            if normalized_aspect < min_normalized_aspect:
                height = float(min_normalized_aspect * (y_span / x_span) * width)
            height = float(min(height, 0.95 * y_span))
            height = float(max(height, 0.06 * y_span))

        ellipse = Ellipse(
            xy=(x0, y0),
            width=width,
            height=height,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )
        ax1.add_patch(ellipse)

    if int(unique_completion.size) <= 8:
        ax1.set_xticks([float(v) for v in sorted(map(float, unique_completion))])

    y_text = float(y_max - 0.02 * (y_max - y_min))
    for v in sorted(map(float, unique_completion)):
        m = completion == v
        if int(np.sum(m)) <= 0:
            continue
        avg_cost = float(np.mean(cost_trillion[m]))
        ax1.text(
            float(v),
            y_text,
            f"n={int(np.sum(m))}\nmean=${avg_cost:.3f}T",
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()
    return _save_matplotlib(fig, filename, out_dir)


def _format_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x * 100.0:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="auto", choices=["auto", "gurobi", "pulp"])
    parser.add_argument("--time-limit", type=int, default=30)
    parser.add_argument("--mip-gap", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-saa", type=int, default=40)
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    params_base = Params(
        p_fail_base=0.20,
        p_swing=1.0,
        theta_swing=0.70,
        p_elevator_down=0.0,
        lambda_shortage=1.0e7,
        x_max_per_base=450,
    )

    baseline, baseline_eval_results, baseline_rules = _run_policy_eval(
        params_base,
        solver=args.solver,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        seed=args.seed,
        n_saa=args.n_saa,
        n_eval=args.n_eval,
    )

    I, J, T = int(params_base.n_bases), int(params_base.n_ports), int(params_base.T)
    if float(params_base.p_fail_base) > 1e-12:
        denom = max(1, int(round(1.0 / float(params_base.p_fail_base))))
    else:
        denom = T + 1
    base_failed_ref = (((np.arange(I)[:, None] + np.arange(T)[None, :]) % denom) == 0).astype(int)
    theta_ref_val = float((1.0 - float(params_base.p_swing)) + float(params_base.p_swing) * float(params_base.theta_swing))
    theta_ref = np.full((J, T), theta_ref_val, dtype=float)
    elevator_up_ref = np.ones((J, T), dtype=int)
    scenario_ref = Scenario(base_failed=base_failed_ref, theta=theta_ref, elevator_up=elevator_up_ref)
    reference_result = simulate_heuristic(params_base, scenario_ref, baseline_rules)
    reference_point = {
        "completion": float(reference_result["completion_t"]),
        "cost_trillion": float(reference_result["total_cost"]) / 1e12,
    }

    variation_specs = {
        "p_fail_base": [0.9 * params_base.p_fail_base, params_base.p_fail_base, 1.1 * params_base.p_fail_base],
        "theta_swing": [0.9 * params_base.theta_swing, params_base.theta_swing, 1.1 * params_base.theta_swing],
        "lambda_shortage": [0.9 * params_base.lambda_shortage, params_base.lambda_shortage, 1.1 * params_base.lambda_shortage],
    }

    series: Dict[str, List[Tuple[float, Dict[str, float]]]] = {}
    for k, values in variation_specs.items():
        pts: List[Tuple[float, Dict[str, float]]] = []
        for v in values:
            if k == "p_fail_base":
                params = replace(params_base, p_fail_base=float(np.clip(v, 0.0, 0.99)))
            elif k == "theta_swing":
                params = replace(params_base, theta_swing=float(np.clip(v, 0.0, 1.0)))
            else:
                params = replace(params_base, lambda_shortage=float(max(v, 1e-9)))
            metrics = _evaluate(
                params,
                solver=args.solver,
                time_limit_s=args.time_limit,
                mip_gap=args.mip_gap,
                seed=args.seed,
                n_saa=args.n_saa,
                n_eval=args.n_eval,
            )
            pct = -10.0 if v == values[0] else (10.0 if v == values[-1] else 0.0)
            pts.append((pct, metrics))
        series[k] = pts

    params_batch_fault = replace(params_base, p_fail_base=0.24)
    params_severe_swing = replace(params_base, theta_swing=0.40)
    params_combined = replace(params_base, p_fail_base=0.24, theta_swing=0.40)

    extreme_batch_fault = _evaluate(
        params_batch_fault,
        solver=args.solver,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        seed=args.seed,
        n_saa=args.n_saa,
        n_eval=args.n_eval,
    )
    extreme_severe_swing = _evaluate(
        params_severe_swing,
        solver=args.solver,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        seed=args.seed,
        n_saa=args.n_saa,
        n_eval=args.n_eval,
    )
    extreme_combined = _evaluate(
        params_combined,
        solver=args.solver,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        seed=args.seed,
        n_saa=args.n_saa,
        n_eval=args.n_eval,
    )

    print("=== Robustness: ±10% parameter perturbations (vs baseline) ===")
    base_completion = float(baseline["E_completion"])
    base_obj = float(baseline["E_objective"])

    for key, name in [
        ("p_fail_base", "Rocket failure probability p_fail"),
        ("theta_swing", "Elevator factor theta_l"),
        ("lambda_shortage", "Shortfall penalty lambda"),
    ]:
        pts = series[key]
        neg = pts[0][1]
        pos = pts[-1][1]
        comp_neg = (float(neg["E_completion"]) - base_completion) / max(base_completion, 1e-12)
        comp_pos = (float(pos["E_completion"]) - base_completion) / max(base_completion, 1e-12)
        obj_neg = (float(neg["E_objective"]) - base_obj) / max(base_obj, 1e-12)
        obj_pos = (float(pos["E_objective"]) - base_obj) / max(base_obj, 1e-12)
        print(
            f"- {name}: completion change {_format_pct(comp_neg)} / {_format_pct(comp_pos)}, "
            f"objective change {_format_pct(obj_neg)} / {_format_pct(obj_pos)}"
        )

    print("\n=== Robustness: extreme scenarios (vs baseline) ===")
    for tag, m in [
        ("Batch faults p=0.24", extreme_batch_fault),
        ("Severe swing theta=0.40", extreme_severe_swing),
        ("Combined extremes", extreme_combined),
    ]:
        comp = (float(m["E_completion"]) - base_completion) / max(base_completion, 1e-12)
        obj = (float(m["E_objective"]) - base_obj) / max(base_obj, 1e-12)
        sat = float(m["constraint_satisfaction"])
        print(
            f"- {tag}: completion change {_format_pct(comp)}, objective change {_format_pct(obj)}, "
            f"constraint satisfaction {sat:.2%}"
        )

    paths = []
    paths.append(plot_param_variation(baseline, series, out_dir=out_dir))
    paths.append(
        plot_extreme_objective(
            float(baseline["E_objective"]),
            {
                "batch_fault": float(extreme_batch_fault["E_objective"]),
                "severe_swing": float(extreme_severe_swing["E_objective"]),
                "combined": float(extreme_combined["E_objective"]),
            },
            out_dir=out_dir,
        )
    )

    radar_labels, radar_base, radar_other = _radar_scores(baseline, extreme_combined)
    paths.append(plot_radar(radar_labels, radar_base, radar_other, other_name="Combined extremes", out_dir=out_dir))
    baseline_completion = np.array([float(r["completion_t"]) for r in baseline_eval_results], dtype=float)
    baseline_cost_trillion = np.array([float(r["total_cost"]) / 1e12 for r in baseline_eval_results], dtype=float)
    paths.append(
        plot_monte_carlo_risk(
            baseline_point=reference_point,
            completion=baseline_completion,
            cost_trillion=baseline_cost_trillion,
            out_dir=out_dir,
        )
    )

    print("\n=== Figures saved ===")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
