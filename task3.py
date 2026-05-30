"""
【模型建立模块】Q3：水资源动态补给优化模型（可视化脚本）

输出：
- task3_tank_model.png：Eq.(2) 的水箱模型图（Tank Model）
- task3_dashboard.png：任务3核心结果仪表板（火箭、库存、供给构成、回收率）
- task3_cumulative.png：累计火箭数与累计成本
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
except Exception as e:
    raise RuntimeError("matplotlib 不可用，无法输出PNG图表") from e


def _set_plot_style() -> None:
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save_matplotlib(fig: plt.Figure, filename: str) -> str:
    out_path = os.path.abspath(filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


@dataclass(frozen=True)
class Task3Params:
    T: int = 520
    cap_ton_per_rocket: float = 100.0
    c_launch_usd_per_kg: float = 500.0
    alpha_per_ton: float = 0.001
    eta_initial: float = 0.85
    eta_max: float = 0.99
    population: int = 100_000
    w_per_l_per_person_per_day: float = 200.0
    s_init_ton: float = 500.0
    s_safe_ton: float = 810.0

    @property
    def demand_ton_per_week(self) -> float:
        return float(self.population) * float(self.w_per_l_per_person_per_day) * 7.0 / 1000.0

    @property
    def cost_per_rocket_musd(self) -> float:
        return float(self.cap_ton_per_rocket) * 1000.0 * float(self.c_launch_usd_per_kg) / 1e6


def _generate_reference_policy(params: Task3Params) -> pd.DataFrame:
    T = int(params.T)
    demand = float(params.demand_ton_per_week)

    weeks = np.arange(T, dtype=int)
    rockets = np.zeros(T, dtype=float)
    m_water = np.zeros(T, dtype=float)
    m_equip = np.zeros(T, dtype=float)
    stock = np.zeros(T, dtype=float)
    eta = np.zeros(T, dtype=float)
    recycled = np.zeros(T, dtype=float)
    cum_rockets = np.zeros(T, dtype=float)
    cum_cost_m = np.zeros(T, dtype=float)
    cum_equip = np.zeros(T, dtype=float)
    self_suff = np.zeros(T, dtype=float)

    equip_needed = max(0.0, (float(params.eta_max) - float(params.eta_initial)) / max(float(params.alpha_per_ton), 1e-12))
    m_equip[0] = float(equip_needed)
    eta[0] = min(float(params.eta_max), float(params.eta_initial) + float(params.alpha_per_ton) * m_equip[0])

    recycled[0] = 0.0
    m_water[0] = demand + (float(params.s_safe_ton) - float(params.s_init_ton))
    total0 = m_water[0] + m_equip[0]
    rockets[0] = float(math.ceil(total0 / float(params.cap_ton_per_rocket)))
    stock[0] = float(params.s_init_ton) + m_water[0] + recycled[0] - demand
    self_suff[0] = 0.0

    for t in range(1, T):
        cum_equip[t] = cum_equip[t - 1] + m_equip[t - 1]
        eta[t] = min(float(params.eta_max), float(params.eta_initial) + float(params.alpha_per_ton) * float(cum_equip[t]))

        recycled[t] = demand * float(eta[t - 1])
        self_suff[t] = float(np.clip(recycled[t] / max(demand, 1e-12), 0.0, 1.0))

        m_water[t] = max(0.0, demand - recycled[t])
        rockets[t] = float(math.ceil((m_water[t] + m_equip[t]) / float(params.cap_ton_per_rocket)))
        stock[t] = float(stock[t - 1] + m_water[t] + recycled[t] - demand)

    cum_equip = np.cumsum(m_equip)
    cum_rockets = np.cumsum(rockets)
    cum_cost_m = cum_rockets * float(params.cost_per_rocket_musd)

    df = pd.DataFrame(
        {
            "Week": weeks,
            "Rockets": rockets,
            "Water_Shipped_ton": m_water,
            "Equipment_Shipped_ton": m_equip,
            "Total_Shipped_ton": m_water + m_equip,
            "Stock_Level_ton": stock,
            "Recycling_Rate": eta,
            "Recycled_Water_ton": recycled,
            "Water_Demand_ton": np.full(T, demand, dtype=float),
            "Cumulative_Rockets": cum_rockets,
            "Cumulative_Cost_M": cum_cost_m,
            "Cumulative_Equipment": cum_equip,
            "Self_Sufficiency": self_suff,
        }
    )
    return df


def load_or_build_results(
    *,
    params: Task3Params,
    csv_path: str,
    force_rebuild: bool,
) -> Tuple[pd.DataFrame, str]:
    csv_path_abs = os.path.abspath(csv_path)
    if (not force_rebuild) and os.path.exists(csv_path_abs):
        return pd.read_csv(csv_path_abs), csv_path_abs

    df = _generate_reference_policy(params)
    df.to_csv(csv_path_abs, index=False)
    return df, csv_path_abs


def fig_tank_model(save_path: str) -> str:
    _set_plot_style()

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    tank = FancyBboxPatch(
        (3.0, 1.2),
        4.0,
        3.6,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2.0,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(tank)

    water_rect = Rectangle((3.15, 1.35), 3.7, 2.3, linewidth=0.0, facecolor="#4C78A8", alpha=0.65)
    ax.add_patch(water_rect)

    ax.text(5.0, 5.25, "Tank Model (Eq.(2))", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.text(
        5.0,
        0.55,
        r"$S_t = S_{t-1} + m_{water,t} + R_t - D_{total,t}$",
        ha="center",
        va="center",
        fontsize=13,
    )

    ax.text(5.0, 2.7, r"Stock $S_t$", ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    arrow1 = FancyArrowPatch((1.0, 3.9), (3.0, 3.9), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#1f77b4")
    arrow2 = FancyArrowPatch((1.0, 2.6), (3.0, 2.6), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#2ca02c")
    arrow3 = FancyArrowPatch((7.0, 3.25), (9.0, 3.25), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#d62728")
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)

    ax.text(2.0, 4.15, r"Inflow: $m_{water,t}$", ha="center", va="bottom", fontsize=11, color="#1f77b4")
    ax.text(2.0, 2.85, r"Inflow: $R_t$", ha="center", va="bottom", fontsize=11, color="#2ca02c")
    ax.text(8.0, 3.5, r"Outflow: $D_{total,t}$", ha="center", va="bottom", fontsize=11, color="#d62728")

    ax.text(7.0, 1.05, r"$R_t = D_{total,t-1}\cdot \eta_{t-1}$", ha="center", va="center", fontsize=11)

    fig.tight_layout()
    return _save_matplotlib(fig, save_path)


def fig_dashboard(df: pd.DataFrame, params: Task3Params, save_path: str) -> str:
    _set_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Task3 Water Replenishment Model Results", fontsize=14, fontweight="bold")

    x = df["Week"].to_numpy()

    ax = axes[0, 0]
    ax.plot(x, df["Rockets"], linewidth=1.8, label="Rockets/week")
    ax.set_xlabel("Week")
    ax.set_ylabel("Rockets")
    ax.set_title("Rocket launches over time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, df["Stock_Level_ton"], linewidth=2.0, label="Stock $S_t$")
    ax.axhline(y=float(params.s_safe_ton), linestyle="--", linewidth=1.5, alpha=0.8, label="Safety stock $S_{safe}$")
    ax.set_xlabel("Week")
    ax.set_ylabel("ton")
    ax.set_title("Inventory level (Eq.(2))")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    demand = df["Water_Demand_ton"].to_numpy()
    recycled = df["Recycled_Water_ton"].to_numpy()
    shipped = df["Water_Shipped_ton"].to_numpy()
    ax.plot(x, demand, linewidth=2.0, color="black", label="Demand $D_{total}$")
    ax.fill_between(x, 0, recycled, alpha=0.35, color="#2ca02c", label="Recycled $R_t$")
    ax.fill_between(x, recycled, recycled + shipped, alpha=0.35, color="#1f77b4", label="Shipped $m_{water}$")
    ax.set_xlabel("Week")
    ax.set_ylabel("ton/week")
    ax.set_title("Demand decomposition (recycle + shipped)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(x, df["Recycling_Rate"], linewidth=2.0, color="#9467bd", label="Recycling rate $\\eta_t$")
    ax.set_xlabel("Week")
    ax.set_ylabel("$\\eta$")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Efficiency evolution (Eq.(4)–(5))")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, df["Cumulative_Equipment"], linewidth=1.8, color="#ff7f0e", label="Cumulative equipment (ton)")
    ax2.set_ylabel("ton")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_matplotlib(fig, save_path)


def fig_cumulative(df: pd.DataFrame, params: Task3Params, save_path: str) -> str:
    _set_plot_style()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    x = df["Week"].to_numpy()
    ax1.plot(x, df["Cumulative_Rockets"], linewidth=2.2, color="#1f77b4", label="Cumulative rockets")
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Rockets")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, df["Cumulative_Cost_M"], linewidth=2.2, color="#d62728", label="Cumulative cost (M USD)")
    ax2.set_ylabel("M USD")

    ax1.set_title(f"Cumulative rockets and cost (unit cost = {params.cost_per_rocket_musd:.1f} M USD/rocket)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    return _save_matplotlib(fig, save_path)


def fig_initial_zoom(df: pd.DataFrame, params: Task3Params, save_path: str, n_weeks: int = 30) -> str:
    _set_plot_style()

    n = int(max(1, min(int(n_weeks), len(df))))
    d = df.iloc[:n].copy()
    x = d["Week"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(x, d["Rockets"], width=0.85, alpha=0.75, color="#1f77b4", label="Rockets/week")
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Rockets")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, d["Stock_Level_ton"], linewidth=2.2, color="#2ca02c", label="Stock $S_t$")
    ax2.axhline(y=float(params.s_safe_ton), linestyle="--", linewidth=1.5, alpha=0.8, color="#d62728", label="Safety $S_{safe}$")
    ax2.set_ylabel("ton")

    ax1.set_title("Initial replenishment spike (zoomed)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    return _save_matplotlib(fig, save_path)


def main() -> None:
    params = Task3Params()
    results_csv = os.path.join(os.path.dirname(__file__), "lunar_water_optimization_results.csv")
    df, csv_path = load_or_build_results(params=params, csv_path=results_csv, force_rebuild=False)

    out1 = fig_tank_model("task3_tank_model.png")
    out2 = fig_dashboard(df, params, "task3_dashboard.png")
    out3 = fig_cumulative(df, params, "task3_cumulative.png")
    out4 = fig_initial_zoom(df, params, "task3_initial_zoom.png", n_weeks=30)

    print(csv_path)
    print(out1)
    print(out2)
    print(out3)
    print(out4)


if __name__ == "__main__":
    main()
