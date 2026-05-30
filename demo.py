import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save_matplotlib(fig: plt.Figure, n: int) -> str:
    out_path = os.path.abspath(f"demo_{n}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig1_scurve_3d_projection() -> str:
    from mpl_toolkits.mplot3d import Axes3D

    font_scale = 1.2
    # label_fs = int(round(12 * font_scale))
    label_fs = 12
    title_fs = int(round(14 * font_scale))
    legend_fs = int(round(11 * font_scale))
    anno_fs = int(round(10 * font_scale))
    # tick_fs = int(round(11 * font_scale))
    tick_fs = 9

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.55)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])

    years = np.arange(2050, 2201)
    y_mixed = 1e8 / (1 + np.exp(-0.05 * (years - 2120)))
    y_rocket = 3e5 * (years - 2050)
    y_rocket[y_rocket > 1e8] = 1e8
    y_elevator = np.where(years < 2060, 0, 5.37e5 * (years - 2060))
    y_elevator[y_elevator > 1e8] = 1e8

    for y_data, label, color in zip(
        [y_mixed, y_rocket, y_elevator],
        ["Hybrid", "All-Rocket", "All-Elevator"],
        ["#1f77b4", "#ff7f0e", "#2ca02c"],
    ):
        cost = 0.0001 * y_data + 0.000001 * (years - 2050) ** 2
        ax1.plot(
            years,
            y_data / 1e6,
            cost / 1e12,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=0.8,
        )
        ax2.plot(years, y_data / 1e6, label=label, color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Year", fontsize=label_fs, labelpad=8)
    ax1.set_ylabel("Cumulative payload (million tons)", fontsize=label_fs, labelpad=8)
    # ax1.set_zlabel("Cumulative cost (trillion USD)", fontsize=label_fs, labelpad=20)
    ax1.set_title("Cumulative Payload–Cost Trajectories (3D)", fontsize=title_fs, pad=24)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=legend_fs)
    ax1.tick_params(labelsize=tick_fs)

    ax2.set_xlabel("Year", fontsize=label_fs)
    ax2.set_ylabel("Cumulative payload (million tons)", fontsize=label_fs)
    ax2.set_title("Cumulative S-Curves (2D projection)", fontsize=title_fs, pad=18)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=legend_fs)
    ax2.tick_params(labelsize=tick_fs)

    try:
        ax2.set_box_aspect(1)
    except Exception:
        ax2.set_aspect("equal", adjustable="box")

    critical_years = [2060, 2169]
    for cy in critical_years:
        if cy == 2060:
            ax2.axvline(x=cy, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
            ax2.text(cy + 2, 20, "Elevator online", rotation=90, fontsize=anno_fs, color="red")
        else:
            ax2.axvline(x=cy, color="blue", linestyle="--", alpha=0.6, linewidth=1.5)
            ax2.text(cy + 2, -2, "Rocket phase-out", rotation=90, fontsize=anno_fs, color="blue")

    fig.subplots_adjust(wspace=0.55)
    return _save_matplotlib(fig, 1)


def fig2_pareto_frontier() -> str:
    np.random.seed(42)
    n_points = 50
    time = np.random.uniform(120, 350, n_points)
    cost = np.random.uniform(0.3, 7.0, n_points)
    env = np.random.uniform(0, 7.0, n_points)

    cost = cost * (350 - time) / 230
    env = env * (350 - time) / 230

    def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1) | ~np.all(
                    points[is_efficient] <= c, axis=1
                )
                is_efficient[i] = True
        return is_efficient

    points = np.column_stack([time, cost, env])
    pareto_mask = is_pareto_efficient(points)
    pareto_points = points[pareto_mask]

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c="gray", alpha=0.3, s=20, label="Feasible")
    ax1.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        pareto_points[:, 2],
        c="red",
        s=50,
        label="Pareto frontier",
        edgecolors="black",
    )

    ax1.scatter(129, 3.31, 2.95, c="blue", s=150, marker="*", label="Hybrid (recommended)", edgecolors="black", linewidth=2)
    ax1.scatter(196, 0.43, 0.01, c="green", s=150, marker="^", label="All-Elevator", edgecolors="black", linewidth=1)
    ax1.scatter(334, 6.67, 6.67, c="orange", s=150, marker="s", label="All-Rocket", edgecolors="black", linewidth=1)

    ax1.set_xlabel("Build time (years)", fontsize=11, labelpad=10)
    ax1.set_ylabel("Total cost (trillion USD)", fontsize=11, labelpad=10)
    ax1.set_zlabel("Environmental impact (0.1B tCO₂e)", fontsize=11, labelpad=10)
    ax1.set_title("3D Pareto surface (sampled)", fontsize=13, pad=15)
    ax1.legend(loc="upper right", fontsize=9)

    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 1], c="gray", alpha=0.3, s=20)
    ax2.scatter(pareto_points[:, 0], pareto_points[:, 1], c="red", s=40)
    ax2.scatter([129], [3.31], c="blue", s=120, marker="*", edgecolors="black", linewidth=2)
    ax2.scatter([196], [0.43], c="green", s=120, marker="^", edgecolors="black", linewidth=2)
    ax2.scatter([334], [6.67], c="orange", s=120, marker="s", edgecolors="black", linewidth=2)
    ax2.set_xlabel("Build time (years)", fontsize=11)
    ax2.set_ylabel("Total cost (trillion USD)", fontsize=11)
    ax2.set_title("Time–cost trade-off (2D projection)", fontsize=13, pad=10)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133)
    sc = ax3.scatter(points[:, 1], points[:, 2], c=points[:, 0], cmap="viridis", s=30, alpha=0.7)
    ax3.scatter([3.31], [2.95], c="blue", s=120, marker="*", edgecolors="black", linewidth=2)
    ax3.scatter([0.43], [0.01], c="green", s=120, marker="^", edgecolors="black", linewidth=2)
    ax3.scatter([6.67], [6.67], c="orange", s=120, marker="s", edgecolors="black", linewidth=2)
    ax3.set_xlabel("Total cost (trillion USD)", fontsize=11)
    ax3.set_ylabel("Environmental impact (0.1B tCO₂e)", fontsize=11)
    ax3.set_title("Cost–environment relation (color = time)", fontsize=13, pad=10)
    fig.colorbar(sc, ax=ax3, label="Build time (years)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=5.0)
    return _save_matplotlib(fig, 2)


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError("Missing dependency: plotly. Install with: uv add plotly kaleido") from e

    try:
        import kaleido
    except Exception as e:
        raise RuntimeError("Missing dependency: kaleido (for PNG export). Install with: uv add kaleido") from e


def fig3_sankey() -> str:
    _require_plotly()
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.8),
                    label=[
                        "Earth resources",
                        "Rocket transport",
                        "Elevator transport",
                        "Lunar colony",
                        "Time",
                        "Cost",
                        "Environment",
                        "129 years",
                        "$3.31T",
                        "2.95 (0.1B tCO₂e)",
                        "196 years",
                        "$0.43T",
                        "~0",
                        "334 years",
                        "$6.67T",
                        "6.67 (0.1B tCO₂e)",
                    ],
                    color=[
                        "#636efa",
                        "#ef553b",
                        "#00cc96",
                        "#ab63fa",
                        "#ffa15a",
                        "#19d3f3",
                        "#ff6692",
                        "#1f77b4",
                        "#1f77b4",
                        "#1f77b4",
                        "#2ca02c",
                        "#2ca02c",
                        "#2ca02c",
                        "#ff7f0e",
                        "#ff7f0e",
                        "#ff7f0e",
                    ],
                ),
                link=dict(
                    source=[0, 0, 1, 1, 1, 2, 2, 2, 4, 5, 6, 4, 5, 6, 4, 5, 6],
                    target=[1, 2, 4, 5, 6, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    value=[
                        55,
                        45,
                        40,
                        85,
                        95,
                        60,
                        15,
                        5,
                        100,
                        100,
                        100,
                        0,
                        100,
                        100,
                        100,
                        0,
                        100,
                    ],
                    color=[
                        "rgba(31, 119, 180, 0.3)",
                        "rgba(31, 119, 180, 0.3)",
                        "rgba(255, 127, 14, 0.4)",
                        "rgba(255, 127, 14, 0.6)",
                        "rgba(255, 127, 14, 0.8)",
                        "rgba(44, 160, 44, 0.4)",
                        "rgba(44, 160, 44, 0.6)",
                        "rgba(44, 160, 44, 0.8)",
                        "rgba(31, 119, 180, 0.7)",
                        "rgba(31, 119, 180, 0.7)",
                        "rgba(31, 119, 180, 0.7)",
                        "rgba(44, 160, 44, 0.7)",
                        "rgba(44, 160, 44, 0.7)",
                        "rgba(44, 160, 44, 0.7)",
                        "rgba(255, 127, 14, 0.7)",
                        "rgba(255, 127, 14, 0.7)",
                        "rgba(255, 127, 14, 0.7)",
                    ],
                    label=[
                        "55%",
                        "45%",
                        "Time 40%",
                        "Cost 85%",
                        "Env 95%",
                        "Time 60%",
                        "Cost 15%",
                        "Env 5%",
                        "Hybrid",
                        "Hybrid",
                        "Hybrid",
                        "All-Elevator",
                        "All-Elevator",
                        "All-Elevator",
                        "All-Rocket",
                        "All-Rocket",
                        "All-Rocket",
                    ],
                ),
            )
        ]
    )

    fig.update_layout(
        title=None,
        font=dict(size=12, family="Arial"),
        width=1200,
        height=700,
    )

    out_path = os.path.abspath("demo_3.png")
    fig.write_image(out_path, width=1200, height=700, scale=2)
    return out_path


def fig4_monte_carlo_risk() -> str:
    from matplotlib.patches import Ellipse
    from scipy.stats import gaussian_kde

    np.random.seed(42)
    n_simulations = 1000
    baseline = {"time": 129, "cost": 3.31, "env": 2.95}

    time_perturbed = baseline["time"] * (1 + np.random.beta(2, 5, n_simulations) * 0.20)
    cost_perturbed = baseline["cost"] * (1 + np.random.beta(2, 4, n_simulations) * 0.25)

    n_outliers = int(0.05 * n_simulations)
    outlier_idx = np.random.choice(n_simulations, n_outliers, replace=False)
    time_perturbed[outlier_idx] *= np.random.uniform(1.3, 2.0, n_outliers)
    cost_perturbed[outlier_idx] *= np.random.uniform(1.4, 2.5, n_outliers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1 = axes[0, 0]
    ax1.scatter(time_perturbed, cost_perturbed, c="blue", alpha=0.5, s=30, edgecolors="w", linewidth=0.5)

    cov = np.cov(time_perturbed, cost_perturbed)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ellipse = Ellipse(
        xy=(np.mean(time_perturbed), np.mean(cost_perturbed)),
        width=lambda_[0] * 2 * 2,
        height=lambda_[1] * 2 * 2,
        angle=np.rad2deg(np.arccos(v[0, 0])),
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        linestyle="--",
        alpha=0.8,
    )
    ax1.add_patch(ellipse)

    ax1.scatter(
        baseline["time"],
        baseline["cost"],
        c="red",
        s=200,
        marker="*",
        edgecolors="black",
        linewidth=2,
        label=f'Baseline\n({baseline["time"]} years, ${baseline["cost"]}T)',
    )
    ax1.scatter(
        np.mean(time_perturbed),
        np.mean(cost_perturbed),
        c="green",
        s=150,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label=f"Expected\n({np.mean(time_perturbed):.0f} years, ${np.mean(cost_perturbed):.2f}T)",
    )

    ax1.set_xlabel("Build time (years)", fontsize=12)
    ax1.set_ylabel("Total cost (trillion USD)", fontsize=12)
    ax1.set_title("Monte Carlo scatter with 95% confidence ellipse", fontsize=13, pad=12)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(time_perturbed, bins=40, density=True, alpha=0.6, color="skyblue", edgecolor="black", linewidth=0.5)
    kde_time = gaussian_kde(time_perturbed)
    x_time = np.linspace(time_perturbed.min(), time_perturbed.max(), 200)
    ax2.plot(x_time, kde_time(x_time), "b-", linewidth=2)
    ax2.axvline(baseline["time"], color="red", linestyle="--", linewidth=2, label=f'Baseline: {baseline["time"]}y')
    ax2.axvline(np.mean(time_perturbed), color="green", linestyle="--", linewidth=2, label=f"Expected: {np.mean(time_perturbed):.0f}y")
    ax2.set_xlabel("Build time (years)", fontsize=12)
    ax2.set_ylabel("Probability density", fontsize=12)
    ax2.set_title("Build-time distribution with KDE", fontsize=13, pad=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.hist(cost_perturbed, bins=40, density=True, alpha=0.6, color="lightcoral", edgecolor="black", linewidth=0.5)
    kde_cost = gaussian_kde(cost_perturbed)
    x_cost = np.linspace(cost_perturbed.min(), cost_perturbed.max(), 200)
    ax3.plot(x_cost, kde_cost(x_cost), "r-", linewidth=2)
    ax3.axvline(baseline["cost"], color="red", linestyle="--", linewidth=2, label=f'Baseline: ${baseline["cost"]}T')
    ax3.axvline(np.mean(cost_perturbed), color="green", linestyle="--", linewidth=2, label=f"Expected: ${np.mean(cost_perturbed):.2f}T")
    ax3.set_xlabel("Total cost (trillion USD)", fontsize=12)
    ax3.set_ylabel("Probability density", fontsize=12)
    ax3.set_title("Cost distribution with KDE", fontsize=13, pad=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    sorted_time = np.sort(time_perturbed)
    cdf_time = np.arange(1, len(sorted_time) + 1) / len(sorted_time)
    ax4.plot(sorted_time, cdf_time, "b-", linewidth=2.5, label="Build time CDF")

    sorted_cost = np.sort(cost_perturbed)
    cdf_cost = np.arange(1, len(sorted_cost) + 1) / len(sorted_cost)
    ax4.plot(sorted_cost / 2, cdf_cost, "r-", linewidth=2.5, label="Cost CDF (scaled)")

    for quantile, color in zip([0.25, 0.5, 0.75, 0.95], ["gray", "black", "gray", "red"]):
        time_q = np.percentile(time_perturbed, quantile * 100)
        cost_q = np.percentile(cost_perturbed, quantile * 100)
        ax4.axvline(time_q, color=color, linestyle=":", alpha=0.7)
        ax4.axvline(cost_q / 2, color=color, linestyle=":", alpha=0.7)
        ax4.text(time_q, quantile + 0.02, f"{quantile*100:.0f}%", fontsize=9, color=color, ha="center")

    ax4.set_xlabel("Build time (years) / total cost (trillion USD, scaled)", fontsize=12)
    ax4.set_ylabel("Cumulative probability", fontsize=12)
    ax4.set_title("CDF of key metrics", fontsize=13, pad=12)
    ax4.legend(fontsize=10, loc="lower right")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_matplotlib(fig, 4)


def fig5_radar_interactive_static() -> str:
    _require_plotly()
    import plotly.graph_objects as go

    categories = ["Time", "Cost", "Environment", "Tech risk", "Ops complexity", "Strategic value"]
    data = {
        "Hybrid": [8.5, 7.0, 6.5, 6.0, 5.5, 9.0],
        "All-Elevator": [4.0, 9.5, 10.0, 2.0, 7.0, 8.0],
        "All-Rocket": [3.0, 2.0, 1.0, 8.0, 3.0, 6.0],
    }

    colors = {"Hybrid": "#1f77b4", "All-Elevator": "#2ca02c", "All-Rocket": "#ff7f0e"}

    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure()

    for scenario, scores in data.items():
        fig.add_trace(
            go.Scatterpolar(
                r=scores + scores[:1],
                theta=categories + categories[:1],
                name=scenario,
                fill="toself",
                fillcolor=hex_to_rgba(colors[scenario], 0.25),
                line=dict(color=colors[scenario], width=2.5),
                opacity=0.9,
            )
        )

    weights_efficiency = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
    weighted_scores = []
    for i in range(len(categories)):
        weighted_score = (
            weights_efficiency[i] * data["Hybrid"][i]
            + weights_efficiency[i] * data["All-Elevator"][i]
            + weights_efficiency[i] * data["All-Rocket"][i]
        )
        weighted_scores.append(weighted_score)

    fig.add_trace(
        go.Scatterpolar(
            r=weighted_scores + weighted_scores[:1],
            theta=categories + categories[:1],
            name="Weight profile",
            mode="lines+markers",
            line=dict(color="purple", width=3, dash="dash"),
            marker=dict(size=8, color="purple"),
        )
    )

    fig.update_layout(
        title=None,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=11), gridcolor="lightgray", gridwidth=1),
            angularaxis=dict(tickfont=dict(size=12), gridcolor="lightgray", gridwidth=1, rotation=90),
            bgcolor="rgba(245, 245, 245, 0.8)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        width=900,
        height=700,
        template="plotly_white",
    )

    fig.add_annotation(
        text="<b>Weight mockup</b><br>Time: ████████░░ 80%<br>Cost: █████░░░░░ 50%<br>Env: ████░░░░░░ 40%",
        xref="paper",
        yref="paper",
        x=1.05,
        y=0.5,
        showarrow=False,
        bordercolor="lightgray",
        borderwidth=1,
        borderpad=10,
        bgcolor="white",
        font=dict(size=12),
    )

    out_path = os.path.abspath("demo_5.png")
    fig.write_image(out_path, width=900, height=700, scale=2)
    return out_path


def fig6_spatiotemporal_heatmap() -> str:
    import seaborn as sns

    years = np.arange(2050, 2180)
    n_years = len(years)

    rocket_ratio = np.zeros(n_years)
    rocket_ratio[:10] = 100
    rocket_ratio[10:120] = np.linspace(100, 0, 110)
    rocket_ratio[120:] = 0

    elevator_ratio = 100 - rocket_ratio

    total_capacity = np.ones(n_years)
    total_capacity[10:120] = 1.5
    total_capacity[120:] = 0.8

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [1, 1, 0.5]})

    im1 = axes[0].imshow(
        rocket_ratio.reshape(1, -1), aspect="auto", cmap="Oranges", vmin=0, vmax=100, extent=[2050, 2179, 0, 1]
    )
    axes[0].set_ylabel("Rocket share\n(%)", fontsize=12, rotation=0, labelpad=30, ha="right")
    axes[0].set_yticks([])
    axes[0].set_title("Rocket capacity share over time", fontsize=13, pad=10)
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation="vertical", pad=0.01)
    cbar1.set_label("Share (%)", fontsize=10)

    im2 = axes[1].imshow(
        elevator_ratio.reshape(1, -1), aspect="auto", cmap="Greens", vmin=0, vmax=100, extent=[2050, 2179, 0, 1]
    )
    axes[1].set_ylabel("Elevator share\n(%)", fontsize=12, rotation=0, labelpad=30, ha="right")
    axes[1].set_yticks([])
    axes[1].set_title("Elevator capacity share over time", fontsize=13, pad=10)
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation="vertical", pad=0.01)
    cbar2.set_label("Share (%)", fontsize=10)

    for ax in axes[:2]:
        for year, label in [(2060, "Elevator online"), (2169, "Rocket phase-out"), (2179, "Mission complete")]:
            ax.axvline(x=year, color="red" if year == 2060 else "blue", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.text(
                year + 1,
                0.5,
                label,
                rotation=90,
                fontsize=10,
                color="red" if year == 2060 else "blue",
                verticalalignment="center",
            )

    axes[2].fill_between(years, 0, total_capacity, color="purple", alpha=0.3, label="Total capacity index")
    axes[2].plot(years, rocket_ratio / 100, "orange", linewidth=2.5, label="Rocket share")
    axes[2].plot(years, elevator_ratio / 100, "green", linewidth=2.5, label="Elevator share")
    axes[2].set_xlabel("Year", fontsize=12)
    axes[2].set_ylabel("Synergy index", fontsize=12)
    axes[2].set_title("Rocket–elevator synergy over time", fontsize=13, pad=10)
    axes[2].set_xlim(2050, 2179)
    axes[2].set_ylim(0, 1.6)
    axes[2].legend(loc="upper right", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    axes[2].axvspan(2080, 2120, alpha=0.2, color="gold", label="Best synergy window (2080–2120)")
    axes[2].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return _save_matplotlib(fig, 6)


def fig7_cost_waterfall() -> str:
    _require_plotly()
    import plotly.graph_objects as go

    categories = ["Total", "Elevator build", "Rocket launches", "Elevator ops", "Repairs", "Carbon tax", "Net"]
    values = [6.67, -3.00, -2.95, -0.01, 0.15, 0.30, 3.31]
    baseline = 6.67

    fig = go.Figure(
        go.Waterfall(
            name="Cost breakdown",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
            x=categories,
            y=values,
            text=[f"${v:.2f}T" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#EF553B"}},
            decreasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#636EFA"}},
        )
    )

    fig.add_shape(type="line", x0=-0.5, x1=6.5, y0=baseline, y1=baseline, line=dict(color="gray", width=2, dash="dash"))

    fig.add_annotation(
        x=6.5,
        y=baseline,
        text=f"All-Rocket baseline: ${baseline}T",
        showarrow=True,
        arrowhead=2,
        ax=50,
        ay=-30,
        font=dict(size=11, color="gray"),
    )

    fig.add_trace(
        go.Scatter(
            x=["All-Elevator cost", "Hybrid net cost"],
            y=[0.43, 3.31],
            mode="markers+text",
            marker=dict(size=15, color=["#2CA02C", "#1F77B4"]),
            text=["$0.43T", "$3.31T"],
            textposition="top center",
            name="Other scenarios",
        )
    )

    fig.update_layout(
        title=None,
        xaxis=dict(title="Cost components", tickfont=dict(size=12)),
        yaxis=dict(title="Cost (trillion USD)", range=[0, 7.5], tickfont=dict(size=11)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        width=1000,
        height=600,
        template="plotly_white",
        hovermode="x unified",
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: $%{y:.2f}T<br>%{text}",
        textposition="outside",
        selector=dict(type="waterfall"),
    )

    out_path = os.path.abspath("demo_7.png")
    fig.write_image(out_path, width=1000, height=600, scale=2)
    return out_path


def fig8_sensitivity_surface() -> str:
    from matplotlib import cm
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import Axes3D

    rng = np.random.default_rng(42)

    def response_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        baseline_period = 129
        period = baseline_period * (1 / x**0.3) * (y**0.4)
        noise = rng.normal(0, 0.02, x.shape)
        return period * (1 + noise)

    x = np.linspace(0.5, 1.5, 30)
    y = np.linspace(0.5, 1.5, 30)
    X, Y = np.meshgrid(x, y)
    Z = response_function(X, Y)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9, antialiased=True, linewidth=0.5, edgecolor="gray")

    ax1.scatter([1.0], [1.0], [129], color="red", s=100, marker="*", edgecolors="black", label="Baseline\n(1.0, 1.0, 129)")
    ax1.quiver(1.0, 1.0, 129, 0.3, 0, -20, color="blue", arrow_length_ratio=0.1, linewidth=2, label="Elevator capacity sensitivity")
    ax1.quiver(1.0, 1.0, 129, 0, 0.3, 15, color="orange", arrow_length_ratio=0.1, linewidth=2, label="Rocket cost sensitivity")

    ax1.set_xlabel("Elevator capacity multiplier\n(vs baseline)", fontsize=11, labelpad=12)
    ax1.set_ylabel("Rocket cost multiplier\n(vs baseline)", fontsize=11, labelpad=12)
    ax1.set_zlabel("Build time (years)", fontsize=11, labelpad=12)
    ax1.set_title("3D response surface", fontsize=13, pad=15)
    ax1.view_init(elev=25, azim=-45)
    ax1.legend(fontsize=9, loc="upper left")

    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, 15, cmap="viridis", alpha=0.8)
    contour_lines = ax2.contour(X, Y, Z, 10, colors="black", alpha=0.5, linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=9, fmt="%d")
    ax2.scatter(1.0, 1.0, color="red", s=150, marker="*", edgecolors="black", label="Baseline (1.0, 1.0)")

    arrow1 = FancyArrowPatch((1.0, 1.0), (1.3, 1.0), arrowstyle="->", color="blue", linewidth=2.5)
    arrow2 = FancyArrowPatch((1.0, 1.0), (1.0, 1.3), arrowstyle="->", color="orange", linewidth=2.5)
    ax2.add_patch(arrow1)
    ax2.add_patch(arrow2)

    ax2.text(
        1.15,
        0.95,
        "Elevator cap ↑\nTime ↓12%",
        fontsize=10,
        color="blue",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )
    ax2.text(
        0.95,
        1.15,
        "Rocket cost ↑\nTime ↑8%",
        fontsize=10,
        color="orange",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    ax2.set_xlabel("Elevator capacity multiplier (vs baseline)", fontsize=11)
    ax2.set_ylabel("Rocket launch cost multiplier (vs baseline)", fontsize=11)
    ax2.set_title("Contour view", fontsize=13, pad=12)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)

    cbar = fig.colorbar(contour, ax=ax2, pad=0.1)
    cbar.set_label("Build time (years)", fontsize=11)

    fig.tight_layout()
    return _save_matplotlib(fig, 8)


def main() -> None:
    outputs = [
        fig1_scurve_3d_projection(),
        fig2_pareto_frontier(),
        fig3_sankey(),
        fig4_monte_carlo_risk(),
        fig5_radar_interactive_static(),
        fig6_spatiotemporal_heatmap(),
        fig7_cost_waterfall(),
        fig8_sensitivity_surface(),
    ]
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
