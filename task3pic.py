from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

try:
    import matplotlib

    if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.transforms import ScaledTranslation
except Exception as e:
    raise RuntimeError("matplotlib 不可用，无法生成图表") from e


def _set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _save_matplotlib(fig: plt.Figure, path: str, *, dpi: int = 300, close: bool = True) -> str:
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    if close:
        plt.close(fig)
    return out_path


def create_3d_inventory_gantt(*, seed: Optional[int] = 42) -> plt.Figure:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection="3d")

    weeks = np.arange(1, 101)

    water_early = rng.uniform(80, 120, 100) * (1 - 0.02 * weeks)
    water_late = rng.uniform(10, 30, 100)
    supply_water = np.where(weeks <= 20, water_early, water_late)

    equip_early = rng.uniform(5, 15, 100)
    equip_late = rng.uniform(1, 3, 100)
    supply_equipment = np.where(weeks <= 20, equip_early, equip_late)

    consumption = rng.uniform(90, 110, 100)
    inventory = np.cumsum(supply_water - consumption) + 1000
    recycling_rate = np.minimum(0.8 + 0.01 * weeks, 0.99)

    xpos, ypos = np.meshgrid(weeks[::2], [0, 1, 2])
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)

    dx = 0.8 * np.ones_like(zpos, dtype=float)
    dy = 0.8 * np.ones_like(zpos, dtype=float)
    dz: list[float] = []
    for i, _week in enumerate(weeks[::2]):
        dz.extend([float(supply_water[i * 2]), float(supply_equipment[i * 2]), float(consumption[i * 2])])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["Water supply", "Equipment supply", "Consumption"]

    dz_arr = np.asarray(dz, dtype=float)
    for i in range(3):
        idx = np.arange(i, len(dz_arr), 3)
        ax1.bar3d(
            xpos[idx],
            ypos[idx],
            zpos[idx],
            dx[idx],
            dy[idx],
            dz_arr[idx],
            color=colors[i],
            alpha=0.8,
            label=labels[i],
        )

    ax1.set_xlabel("Week")
    ax1.set_ylabel("Material")
    ax1.set_zlabel("ton")
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["Water supply", "Equipment supply", "Consumption"])
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2_secondary = ax2.twinx()

    line1 = ax2.plot(weeks, inventory, "b-", linewidth=3, label="Inventory", alpha=0.7)
    ax2.fill_between(weeks, 810, inventory, alpha=0.2, color="blue")
    ax2.axhline(y=810, color="r", linestyle="--", alpha=0.5, label="Safety stock")

    line2 = ax2_secondary.plot(weeks, recycling_rate * 100, "g-", linewidth=3, label="Recycling rate", alpha=0.7)
    ax2_secondary.set_ylim(75, 100)

    lines = line1 + line2
    legend_labels = [l.get_label() for l in lines] + ["Safety stock"]
    ax2.legend(lines + [Line2D([0], [0], color="red", linestyle="--")], legend_labels, loc="upper left")

    ax2.set_xlabel("Week")
    ax2.set_ylabel("Inventory (ton)", color="b")
    ax2_secondary.set_ylabel("Recycling rate (%)", color="g")

    plt.tight_layout()
    return fig


def create_radar_stream() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection="polar"))

    categories = ["Equipment input", "Water supply", "Recycling rate", "Launch rate", "Stock safety", "Cost efficiency"]
    n = len(categories)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    values1 = [85, 35, 45, 90, 60, 40]
    values1 += values1[:1]

    values2 = [15, 25, 95, 10, 95, 90]
    values2 += values2[:1]

    def place_category_labels(ax: plt.Axes, *, is_left: bool) -> None:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        r_label = 110
        for theta, label in zip(angles[:-1], categories):
            c = float(np.cos(theta))
            s = float(np.sin(theta))

            ha = "left" if c > 0.2 else ("right" if c < -0.2 else "center")
            va = "bottom" if s > 0.2 else ("top" if s < -0.2 else "center")

            if is_left:
                dx = -18 if c > 0.35 else (-8 if c < -0.35 else 0)
            else:
                dx = 18 if c < -0.35 else (8 if c > 0.35 else 0)

            trans = ax.transData + ScaledTranslation(dx / 72.0, 0.0, fig.dpi_scale_trans)
            ax.text(theta, r_label, label, fontsize=14, ha=ha, va=va, transform=trans, clip_on=False)

    ax1 = axes[0]
    ax1.plot(angles, values1, "o-", linewidth=2, color="#FF6B6B")
    ax1.fill(angles, values1, alpha=0.25, color="#FF6B6B")
    place_category_labels(ax1, is_left=True)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    for i in range(n):
        ax1.annotate(
            "",
            xy=(angles[i], values1[i]),
            xytext=(angles[i], 0),
            arrowprops=dict(arrowstyle="->", color="#FF6B6B", alpha=0.5, lw=1),
        )

    ax2 = axes[1]
    ax2.plot(angles, values2, "o-", linewidth=2, color="#4ECDC4")
    ax2.fill(angles, values2, alpha=0.25, color="#4ECDC4")
    place_category_labels(ax2, is_left=False)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    for i in range(n):
        ax2.annotate(
            "",
            xy=(angles[i], values2[i]),
            xytext=(angles[i], 0),
            arrowprops=dict(arrowstyle="->", color="#4ECDC4", alpha=0.5, lw=1),
        )

    fig.subplots_adjust(wspace=0.55)
    plt.tight_layout()
    return fig


def create_3d_recycling_evolution() -> plt.Figure:
    fig = plt.figure(figsize=(16, 8))

    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    ax1 = fig.add_subplot(121, projection="3d")

    weeks = np.arange(0, 101, 5)
    equipment_investment = np.linspace(0, 2, 21)

    x, y = np.meshgrid(weeks, equipment_investment)
    z = np.zeros_like(x, dtype=float)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            week = float(x[i, j])
            equip = float(y[i, j])
            base = 0.8
            improvement = 0.001 * equip * 1000
            saturation = 0.99
            z[i, j] = min(base + improvement * (1 - np.exp(-week / 20.0)), saturation)

    surf = ax1.plot_surface(x, y, z * 100, cmap=cmap, alpha=0.9, linewidth=0.1, antialiased=True)

    ax1.contour(x, y, z * 100, zdir="z", offset=75, cmap="viridis", alpha=0.3)
    ax1.contour(x, y, z * 100, zdir="x", offset=0, cmap="viridis", alpha=0.3)
    ax1.contour(x, y, z * 100, zdir="y", offset=0, cmap="viridis", alpha=0.3)

    ax1.set_xlabel("Operating week", labelpad=10)
    ax1.set_ylabel("Equipment input (ton)", labelpad=10)
    ax1.set_zlabel("Recycling rate (%)", labelpad=10)
    ax1.view_init(elev=25, azim=135)

    cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    cbar.set_label("Recycling rate (%)")

    ax2 = fig.add_subplot(122)

    weeks_fine = np.arange(0, 101)
    equip_fine = np.linspace(0, 2, 101)
    x_fine, y_fine = np.meshgrid(weeks_fine, equip_fine)

    z_fine = np.zeros_like(x_fine, dtype=float)
    for i in range(x_fine.shape[0]):
        for j in range(x_fine.shape[1]):
            week = float(x_fine[i, j])
            equip = float(y_fine[i, j])
            base = 0.8
            improvement = 0.001 * equip * 1000
            saturation = 0.99
            z_fine[i, j] = min(base + improvement * (1 - np.exp(-week / 20.0)), saturation)

    im = ax2.imshow(z_fine * 100, aspect="auto", cmap=cmap, extent=[0, 100, 0, 2], origin="lower")

    contour = ax2.contour(
        weeks_fine,
        equip_fine,
        z_fine * 100,
        levels=[85, 90, 95, 98, 99],
        colors="white",
        alpha=0.7,
        linewidths=1,
    )
    ax2.clabel(contour, inline=True, fontsize=8, fmt="%.0f%%")

    optimal_path = np.minimum(1.5 * np.exp(-weeks_fine / 40.0), 0.2)
    ax2.plot(weeks_fine, optimal_path, "r--", linewidth=3, label="Optimal investment path")

    ax2.set_xlabel("Operating week")
    ax2.set_ylabel("Equipment input (ton)")
    ax2.legend(loc="upper right")

    cbar2 = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar2.set_label("Recycling rate (%)")
    plt.tight_layout()
    return fig



def _require_plotly():
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError("plotly 不可用，无法生成桑基图（建议：uv add plotly kaleido）") from e
    return go


def create_sankey_sensitivity():
    go = _require_plotly()

    labels = [
        "Rocket payload",
        "Conversion factor",
        "Safety stock",
        "Unit transport cost",
        "Equipment efficiency",
        "Inventory turnover",
        "Total transport cost",
        "Launch count",
        "Water self-sufficiency",
    ]

    sources = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 3, 4, 5]
    targets = [3, 6, 4, 7, 5, 8, 6, 7, 7, 8, 8, 7, 7, 8, 6]
    values = [45, 25, 30, 20, 15, 10, 35, 30, 25, 20, 15, 10, 25, 20, 15]

    link_colors = ["rgba(231, 111, 81, 0.6)" if t in [6, 7] else "rgba(42, 157, 143, 0.6)" for t in targets]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=[
                        "#264653",
                        "#2a9d8f",
                        "#e9c46a",
                        "#f4a261",
                        "#e76f51",
                        "#b5838d",
                        "#264653",
                        "#2a9d8f",
                        "#e9c46a",
                    ],
                ),
                link=dict(source=sources, target=targets, value=values, color=link_colors),
            )
        ]
    )

    fig.update_layout(
        title_text=None,
        font_size=12,
        height=600,
        annotations=[
            dict(x=0.1, y=1.1, xref="paper", yref="paper", text="Inputs", showarrow=False, font=dict(size=14, color="#264653")),
            dict(x=0.5, y=1.1, xref="paper", yref="paper", text="Intermediate metrics", showarrow=False, font=dict(size=14, color="#f4a261")),
            dict(x=0.9, y=1.1, xref="paper", yref="paper", text="Outcomes", showarrow=False, font=dict(size=14, color="#e76f51")),
        ],
    )

    return fig


def create_robustness_network() -> plt.Figure:
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    def wrap_label(text: str) -> str:
        parts = text.split()
        if len(parts) <= 1:
            return text
        if len(text) <= 12:
            return text
        return "\n".join(parts)

    scenarios = ["Baseline", "Equipment shortage", "Demand surge", "Launch constrained", "Stock depletion"]
    metrics = ["Feasibility", "Cost increase", "Launch increase", "Safety", "Satisfaction"]

    edges_with_weights = [
        ("Baseline", "Feasibility", 1.0),
        ("Baseline", "Cost increase", 0.0),
        ("Baseline", "Launch increase", 0.0),
        ("Baseline", "Safety", 1.0),
        ("Baseline", "Satisfaction", 1.0),
        ("Equipment shortage", "Feasibility", 1.0),
        ("Equipment shortage", "Cost increase", 0.053),
        ("Equipment shortage", "Launch increase", 0.062),
        ("Equipment shortage", "Safety", 1.0),
        ("Equipment shortage", "Satisfaction", 1.0),
        ("Demand surge", "Feasibility", 1.0),
        ("Demand surge", "Cost increase", 0.045),
        ("Demand surge", "Launch increase", 0.051),
        ("Demand surge", "Safety", 1.0),
        ("Demand surge", "Satisfaction", 1.0),
        ("Launch constrained", "Feasibility", 1.0),
        ("Launch constrained", "Cost increase", 0.070),
        ("Launch constrained", "Launch increase", 0.083),
        ("Launch constrained", "Safety", 1.0),
        ("Launch constrained", "Satisfaction", 1.0),
        ("Stock depletion", "Feasibility", 1.0),
        ("Stock depletion", "Cost increase", 0.038),
        ("Stock depletion", "Launch increase", 0.045),
        ("Stock depletion", "Safety", 1.0),
        ("Stock depletion", "Satisfaction", 1.0),
    ]

    pos: dict[str, tuple[float, float]] = {}
    scenario_ys = np.linspace(0.88, 0.12, len(scenarios))
    metric_ys = np.linspace(0.90, 0.10, len(metrics))
    for i, scenario in enumerate(scenarios):
        pos[scenario] = (0.18, float(scenario_ys[i]))
    for i, metric in enumerate(metrics):
        pos[metric] = (0.82, float(metric_ys[i]))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    for node in scenarios:
        x0, y0 = pos[node]
        ax.scatter([x0], [y0], s=2800, c="#264653", alpha=0.92, zorder=3)
        label = wrap_label(node)
        font_size = 10 if "\n" in label else 11
        ax.text(
            x0,
            y0,
            label,
            fontsize=font_size,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            linespacing=0.95,
            zorder=4,
        )

    for node in metrics:
        x0, y0 = pos[node]
        ax.scatter([x0], [y0], s=2200, c="#2a9d8f", alpha=0.92, zorder=3)
        label = wrap_label(node)
        font_size = 10 if "\n" in label else 11
        ax.text(
            x0,
            y0,
            label,
            fontsize=font_size,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            linespacing=0.95,
            zorder=4,
        )

    for u, v, w in edges_with_weights:
        w = float(w)
        if w < 0.02:
            edge_color = "#2a9d8f"
        elif w < 0.05:
            edge_color = "#e9c46a"
        else:
            edge_color = "#e76f51"

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        i = scenarios.index(u)
        j = metrics.index(v)
        rad = 0.28 * (float(j - i) / float(max(len(scenarios) - 1, 1)))

        edge = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-",
            linewidth=(0.9 + w * 35),
            color=edge_color,
            alpha=0.55,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(edge)

        if w >= 0.04:
            t = 0.62
            xm = x1 + t * (x2 - x1)
            ym = y1 + t * (y2 - y1)
            center_i = (len(scenarios) - 1) / 2.0
            center_j = (len(metrics) - 1) / 2.0
            jitter_y = (float(i) - center_i) * 0.020 + (float(j) - center_j) * 0.014
            jitter_x = (float(j - i)) * 0.006
            group_y = 0.012 if w >= 0.05 else (-0.006 if w >= 0.02 else 0.0)
            xm = float(np.clip(xm + jitter_x, 0.02, 0.98))
            ym = float(np.clip(ym + rad * 0.14 + jitter_y + group_y, 0.02, 0.98))
            ax.text(
                xm,
                ym,
                f"{w:.1%}",
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.78),
                zorder=2,
            )

    ax.scatter([], [], c="#264653", s=240, label="Scenario node")
    ax.scatter([], [], c="#2a9d8f", s=240, label="Metric node")
    ax.plot([], [], color="#2a9d8f", linewidth=3, label="Low impact (<2%)")
    ax.plot([], [], color="#e9c46a", linewidth=3, label="Medium impact (2-5%)")
    ax.plot([], [], color="#e76f51", linewidth=3, label="High impact (>5%)")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        labelspacing=1.2,
        handlelength=2.2,
        handletextpad=0.8,
        columnspacing=1.6,
        borderpad=0.8,
    )
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=".", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于图1模拟数据）")
    parser.add_argument("--show", action="store_true", help="生成后弹窗显示（无GUI时无效）")
    args = parser.parse_args()

    _set_plot_style()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    close = not bool(args.show)

    out1 = _save_matplotlib(create_3d_inventory_gantt(seed=int(args.seed)), os.path.join(out_dir, "task3pic_1_inventory_gantt_3d.png"), close=close)
    out2 = _save_matplotlib(create_radar_stream(), os.path.join(out_dir, "task3pic_2_radar_stream.png"), close=close)
    out3 = _save_matplotlib(create_3d_recycling_evolution(), os.path.join(out_dir, "task3pic_3_recycling_evolution.png"), close=close)

    sankey_fig = create_sankey_sensitivity()
    out4 = os.path.abspath(os.path.join(out_dir, "task3pic_4_sankey_sensitivity.png"))
    try:
        sankey_fig.write_image(out4, scale=2)
    except Exception as e:
        raise RuntimeError("桑基图PNG导出失败（需要kaleido）。建议：uv add kaleido") from e

    out5 = _save_matplotlib(create_robustness_network(), os.path.join(out_dir, "task3pic_5_robustness_network.png"), close=close)

    print(out1)
    print(out2)
    print(out3)
    print(out4)
    print(out5)

    if bool(args.show):
        plt.show()


if __name__ == "__main__":
    main()
