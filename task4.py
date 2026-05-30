"""
【模型建立模块】问题4：评估并最小化环境影响
方案1：混合整数规划（MILP）+ 碳排放约束松弛
方案2：多目标（环境/成本/时间）+ AHP阶段权重 + 遗传算法（启发式）

输出：
- task4_flowchart.png：建模流程图（Matplotlib绘制）
- task4_basic_dashboard.png：方案1核心结果图表
- task4_pareto.png：方案2候选解近似帕累托前沿
- task4_basic_results.csv：方案1逐年结果表
- task4_innovative_top3.csv：方案2 Top3方案汇总
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon
except Exception as e:
    raise RuntimeError("matplotlib 不可用，无法输出PNG图表") from e

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    LpStatus,
    lpSum,
    value,
    PULP_CBC_CMD,
)


@dataclass(frozen=True)
class BasicEnvParams:
    T: int = 130
    n_bases: int = 10
    n_ports: int = 3
    q_rocket_wan_ton: float = 0.015
    y_max_wan_ton: float = 17.9
    x_max_per_base: int = 200
    q_total_wan_ton: float = 10000.0
    e1_tco2_per_launch: float = 500.0
    e2_tco2_per_wan_ton: float = 100.0
    slack_penalty_M: float = 3.0
    e_limit_start: float = 180_000.0
    e_limit_end: float = 260_000.0

    def x_max_vec(self) -> np.ndarray:
        return np.full(self.n_bases, int(self.x_max_per_base), dtype=int)

    def e_limit_vec(self) -> np.ndarray:
        return np.linspace(self.e_limit_start, self.e_limit_end, self.T, dtype=float)


def _set_plot_style():
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def solve_basic_milp(params: BasicEnvParams, solver: str = "auto", time_limit_s: int = 180) -> Dict:
    T = params.T
    I = params.n_bases
    J = params.n_ports
    x_max = params.x_max_vec()
    e_limit = params.e_limit_vec()

    use_gurobi = (solver == "gurobi") or (solver == "auto" and GUROBI_AVAILABLE)
    if use_gurobi:
        m = gp.Model("Task4_Basic_MILP")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = float(time_limit_s)

        x = m.addVars(I, T, vtype=GRB.INTEGER, lb=0.0, name="x")
        y = m.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
        sigma = m.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, name="sigma")

        for i in range(I):
            for t in range(T):
                m.addConstr(x[i, t] <= float(x_max[i]))
        for j in range(J):
            for t in range(T):
                m.addConstr(y[j, t] <= float(params.y_max_wan_ton))

        for t in range(T):
            e_total = (
                params.e1_tco2_per_launch * gp.quicksum(x[i, t] for i in range(I))
                + params.e2_tco2_per_wan_ton * gp.quicksum(y[j, t] for j in range(J))
            )
            m.addConstr(e_total - sigma[t] <= float(e_limit[t]))

        q_total = gp.quicksum(
            params.q_rocket_wan_ton * gp.quicksum(x[i, t] for i in range(I))
            + gp.quicksum(y[j, t] for j in range(J))
            for t in range(T)
        )
        m.addConstr(q_total >= float(params.q_total_wan_ton))

        obj = gp.quicksum(
            params.e1_tco2_per_launch * gp.quicksum(x[i, t] for i in range(I))
            + params.e2_tco2_per_wan_ton * gp.quicksum(y[j, t] for j in range(J))
            + params.slack_penalty_M * sigma[t]
            for t in range(T)
        )
        m.setObjective(obj, GRB.MINIMIZE)

        m.optimize()
        status = str(m.Status)

        x_sol = np.array([[x[i, t].X for t in range(T)] for i in range(I)], dtype=float)
        y_sol = np.array([[y[j, t].X for t in range(T)] for j in range(J)], dtype=float)
        sigma_sol = np.array([sigma[t].X for t in range(T)], dtype=float)
        objective = float(m.ObjVal) if m.SolCount > 0 else float("nan")
        return {
            "status": status,
            "objective": objective,
            "x": x_sol,
            "y": y_sol,
            "sigma": sigma_sol,
            "params": params,
        }

    prob = LpProblem("Task4_Basic_MILP", LpMinimize)
    x = LpVariable.dicts("x", (range(I), range(T)), lowBound=0, cat="Integer")
    y = LpVariable.dicts("y", (range(J), range(T)), lowBound=0, cat="Continuous")
    sigma = LpVariable.dicts("sigma", range(T), lowBound=0, cat="Continuous")

    for i in range(I):
        for t in range(T):
            prob += x[i][t] <= int(x_max[i])
    for j in range(J):
        for t in range(T):
            prob += y[j][t] <= float(params.y_max_wan_ton)

    for t in range(T):
        e_total = (
            params.e1_tco2_per_launch * lpSum(x[i][t] for i in range(I))
            + params.e2_tco2_per_wan_ton * lpSum(y[j][t] for j in range(J))
        )
        prob += e_total - sigma[t] <= float(e_limit[t])

    q_total = lpSum(
        params.q_rocket_wan_ton * lpSum(x[i][t] for i in range(I)) + lpSum(y[j][t] for j in range(J))
        for t in range(T)
    )
    prob += q_total >= float(params.q_total_wan_ton)

    prob += lpSum(
        params.e1_tco2_per_launch * lpSum(x[i][t] for i in range(I))
        + params.e2_tco2_per_wan_ton * lpSum(y[j][t] for j in range(J))
        + params.slack_penalty_M * sigma[t]
        for t in range(T)
    )

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
    status = LpStatus.get(prob.status, str(prob.status))

    x_sol = np.array([[value(x[i][t]) for t in range(T)] for i in range(I)], dtype=float)
    y_sol = np.array([[value(y[j][t]) for t in range(T)] for j in range(J)], dtype=float)
    sigma_sol = np.array([value(sigma[t]) for t in range(T)], dtype=float)
    objective = float(value(prob.objective))
    return {
        "status": status,
        "objective": objective,
        "x": x_sol,
        "y": y_sol,
        "sigma": sigma_sol,
        "params": params,
    }


def summarize_basic_solution(sol: Dict) -> pd.DataFrame:
    params: BasicEnvParams = sol["params"]
    x = np.asarray(sol["x"], dtype=float)
    y = np.asarray(sol["y"], dtype=float)
    sigma = np.asarray(sol["sigma"], dtype=float)
    e_limit = params.e_limit_vec()

    rocket_launches = x.sum(axis=0)
    elevator_flow = y.sum(axis=0)
    rocket_flow = params.q_rocket_wan_ton * rocket_launches
    total_flow = rocket_flow + elevator_flow

    e_rocket = params.e1_tco2_per_launch * rocket_launches
    e_elevator = params.e2_tco2_per_wan_ton * elevator_flow
    e_total = e_rocket + e_elevator

    df = pd.DataFrame(
        {
            "YearIndex": np.arange(1, params.T + 1, dtype=int),
            "Rocket_Launches": rocket_launches,
            "Rocket_Flow_wan_ton": rocket_flow,
            "Elevator_Flow_wan_ton": elevator_flow,
            "Total_Flow_wan_ton": total_flow,
            "E_Rocket_tCO2e": e_rocket,
            "E_Elevator_tCO2e": e_elevator,
            "E_Total_tCO2e": e_total,
            "E_Limit_tCO2e": e_limit,
            "Slack_tCO2e": sigma,
            "Cum_Flow_wan_ton": np.cumsum(total_flow),
        }
    )
    return df


def plot_basic_dashboard(df: pd.DataFrame, save_path: str):
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Task4 Basic Model Results (MILP + Emission Slack)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["YearIndex"], df["Cum_Flow_wan_ton"], linewidth=2, label="Cumulative Supply")
    ax.axhline(y=df["Cum_Flow_wan_ton"].iloc[-1], linestyle=":", alpha=0.5)
    ax.axhline(y=10000.0, color="tab:red", linestyle="--", label="Demand (100 million tons)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Flow (10k tons)")
    ax.set_title("Cumulative Supply vs Demand")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["YearIndex"], df["E_Total_tCO2e"], linewidth=2, label="E_total")
    ax.plot(df["YearIndex"], df["E_Limit_tCO2e"], linewidth=2, linestyle="--", label="E_limit")
    ax.fill_between(
        df["YearIndex"],
        df["E_Limit_tCO2e"],
        df["E_Total_tCO2e"],
        where=(df["E_Total_tCO2e"] > df["E_Limit_tCO2e"]),
        alpha=0.25,
        label="Slack Area",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("tCO2e")
    ax.set_title("Emissions vs Limit")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    width = 0.9
    ax.bar(df["YearIndex"], df["Elevator_Flow_wan_ton"], width=width, label="Elevator", color="tab:green", alpha=0.75)
    ax.bar(
        df["YearIndex"],
        df["Rocket_Flow_wan_ton"],
        width=width,
        bottom=df["Elevator_Flow_wan_ton"],
        label="Rocket",
        color="tab:blue",
        alpha=0.75,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Flow (10k tons)")
    ax.set_title("Annual Supply Composition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(df["YearIndex"], df["Rocket_Launches"], linewidth=2, color="tab:purple", label="Launches")
    ax.set_xlabel("Year")
    ax.set_ylabel("Launches")
    ax.set_title("Annual Rocket Launches")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    backend = str(plt.get_backend()).lower() if hasattr(plt, "get_backend") else ""
    if "agg" not in backend:
        plt.show()
    plt.close(fig)


def plot_flowchart(save_path: str):
    _set_plot_style()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def add_box(x, y, w, h, text, fc="#F2F4F8", ec="#2D3A4A"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)
        return patch

    def add_diamond(cx, cy, w, h, text, fc="#FFF7E6", ec="#2D3A4A"):
        pts = np.array([[cx, cy + h / 2], [cx + w / 2, cy], [cx, cy - h / 2], [cx - w / 2, cy]])
        patch = Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.2)
        ax.add_patch(patch)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=10, wrap=True)
        return patch

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2, color="#2D3A4A"))

    x0, w, h = 0.12, 0.76, 0.065
    y = 0.92
    boxes = []
    steps = [
        "1 Abstract decision scenario\n(10 bases + 3 ports; Plan 2 uses stages)",
        "2 Define variables & constraints\n(integer / binary / continuous)",
        "3 Set & validate assumptions\n(5 core assumptions)",
        "4 Derive intermediate variables\n(E_total, Q_total, cost/time)",
        "5 Formulate objectives & constraints\n(Plan 1 slack; Plan 2 multi-objective + AHP)",
    ]
    for s in steps:
        boxes.append(add_box(x0, y, w, h, s))
        y -= 0.085

    d1 = add_diamond(0.5, y + 0.01, 0.55, 0.09, "6 Constraint compatibility check\nHard conflict?")
    y -= 0.12
    b7 = add_box(x0, y, w, h, "7 Solve & iterate\n(Plan 1 MILP; Plan 2 Genetic Algorithm)")
    y -= 0.085
    d2 = add_diamond(0.5, y + 0.01, 0.55, 0.09, "8 Robustness validation\nPass disturbance scenarios?")
    y -= 0.12
    d3 = add_diamond(0.5, y + 0.01, 0.55, 0.09, "9 Feasibility check\nDecision implementable?")
    y -= 0.12
    b10 = add_box(x0, y, w, h, "10 Final decision output\n(tables / curves / Pareto set)")

    for i in range(len(boxes) - 1):
        arrow(0.5, boxes[i].get_y(), 0.5, boxes[i + 1].get_y() + h)
    arrow(0.5, boxes[-1].get_y(), 0.5, 0.92 - 5 * 0.085 - 0.01)
    arrow(0.5, 0.92 - 5 * 0.085 - 0.11, 0.5, b7.get_y() + h)
    arrow(0.5, b7.get_y(), 0.5, 0.92 - 5 * 0.085 - 0.11 - 0.085 - 0.01)
    arrow(0.5, 0.92 - 5 * 0.085 - 0.11 - 0.085 - 0.12, 0.5, b10.get_y() + h + 0.18)

    ax.text(0.18, d1.get_xy()[0][1] + 0.05, "No", fontsize=10, color="#2D3A4A")
    ax.text(0.78, d1.get_xy()[1][1] + 0.05, "Yes: relax / adjust weights", fontsize=10, ha="right", color="#2D3A4A")

    arrow(0.775, d1.get_xy()[1][1], 0.88, d1.get_xy()[1][1])
    arrow(0.88, d1.get_xy()[1][1], 0.88, boxes[2].get_y() + h / 2)
    arrow(0.88, boxes[2].get_y() + h / 2, 0.78, boxes[2].get_y() + h / 2)

    ax.text(0.18, d2.get_xy()[0][1] + 0.05, "Yes", fontsize=10, color="#2D3A4A")
    ax.text(0.78, d2.get_xy()[1][1] + 0.05, "No: back to solve & iterate", fontsize=10, ha="right", color="#2D3A4A")
    arrow(0.775, d2.get_xy()[1][1], 0.88, d2.get_xy()[1][1])
    arrow(0.88, d2.get_xy()[1][1], 0.88, b7.get_y() + h / 2)
    arrow(0.88, b7.get_y() + h / 2, 0.78, b7.get_y() + h / 2)

    ax.text(0.18, d3.get_xy()[0][1] + 0.05, "Yes", fontsize=10, color="#2D3A4A")
    ax.text(0.78, d3.get_xy()[1][1] + 0.05, "No: back to solve & iterate", fontsize=10, ha="right", color="#2D3A4A")
    arrow(0.775, d3.get_xy()[1][1], 0.88, d3.get_xy()[1][1])
    arrow(0.88, d3.get_xy()[1][1], 0.88, b7.get_y() + h / 2)
    arrow(0.88, b7.get_y() + h / 2, 0.78, b7.get_y() + h / 2)

    fig.suptitle("Task4 Modeling Flowchart (from task4modle.md)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    backend = str(plt.get_backend()).lower() if hasattr(plt, "get_backend") else ""
    if "agg" not in backend:
        plt.show()
    plt.close(fig)


@dataclass(frozen=True)
class InnovativeParams:
    T_max: int = 130
    stage_years: Tuple[int, int, int] = (30, 40, 60)
    stage_require_frac: Tuple[float, float, float] = (0.22, 0.30, 0.48)
    n_bases: int = 10
    n_ports: int = 3
    x_max_per_base: int = 260
    y_max_wan_ton: float = 17.9
    q_rocket_wan_ton: float = 0.015
    q_total_wan_ton: float = 10000.0
    e1_tco2_per_launch: float = 500.0
    e2_tco2_per_wan_ton: float = 100.0
    c1_usd_per_launch: float = 100_000_000.0
    c2_usd_per_wan_ton: float = 5_000_000.0

    def base_profiles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(11)
        cap_delta = rng.integers(-40, 41, size=self.n_bases, dtype=int)
        cap = np.clip(int(self.x_max_per_base) + cap_delta, 200, 320).astype(int)
        e1 = (self.e1_tco2_per_launch * rng.uniform(0.85, 1.25, size=self.n_bases)).astype(float)
        c1 = (self.c1_usd_per_launch * rng.uniform(0.8, 1.2, size=self.n_bases)).astype(float)
        return cap, e1, c1

    def x_max_vec(self) -> np.ndarray:
        cap, _, _ = self.base_profiles()
        return cap

    def stage_require_wan_ton(self) -> np.ndarray:
        return self.q_total_wan_ton * np.array(self.stage_require_frac, dtype=float)

    def stage_weights(self) -> np.ndarray:
        w = np.array(
            [
                [0.2, 0.3, 0.5],
                [0.2, 0.5, 0.3],
                [0.6, 0.3, 0.1],
            ],
            dtype=float,
        )
        w = w / w.sum(axis=1, keepdims=True)
        return w


def _decode_schedule(params: InnovativeParams, z: np.ndarray, r_frac: np.ndarray, e_frac: np.ndarray) -> Dict:
    z = np.asarray(z, dtype=int).clip(0, 1)
    r_frac = np.asarray(r_frac, dtype=float).clip(0.0, 1.0)
    e_frac = np.asarray(e_frac, dtype=float).clip(0.0, 1.0)

    T1, T2, T3 = params.stage_years
    stage_of_year = np.zeros(params.T_max, dtype=int)
    stage_of_year[:T1] = 0
    stage_of_year[T1 : T1 + T2] = 1
    stage_of_year[T1 + T2 :] = 2

    x_max, e1_vec, c1_vec = params.base_profiles()
    x_year_base = np.zeros((params.n_bases, params.T_max), dtype=int)
    y_year_port = np.zeros((params.n_ports, params.T_max), dtype=float)
    for t in range(params.T_max):
        s = stage_of_year[t]
        for i in range(params.n_bases):
            x_year_base[i, t] = int(round(r_frac[s] * x_max[i] * z[i]))
        for j in range(params.n_ports):
            y_year_port[j, t] = float(e_frac[s] * params.y_max_wan_ton)

    rocket_launches = x_year_base.sum(axis=0).astype(float)
    elevator_flow = y_year_port.sum(axis=0).astype(float)
    rocket_flow = params.q_rocket_wan_ton * rocket_launches
    total_flow = rocket_flow + elevator_flow
    cum_flow = np.cumsum(total_flow)
    meet_idx = np.where(cum_flow >= params.q_total_wan_ton)[0]
    completion_year = int(meet_idx[0] + 1) if meet_idx.size > 0 else params.T_max + 1

    e_rocket = (e1_vec.reshape(-1, 1) * x_year_base).sum(axis=0).astype(float)
    c_rocket = (c1_vec.reshape(-1, 1) * x_year_base).sum(axis=0).astype(float)
    e_total = e_rocket + params.e2_tco2_per_wan_ton * elevator_flow
    c_total = c_rocket + params.c2_usd_per_wan_ton * elevator_flow

    return {
        "x": x_year_base,
        "y": y_year_port,
        "rocket_launches": rocket_launches,
        "elevator_flow": elevator_flow,
        "rocket_flow": rocket_flow,
        "total_flow": total_flow,
        "cum_flow": cum_flow,
        "completion_year": completion_year,
        "E_year": e_total,
        "C_year": c_total,
        "E_total": float(e_total[: min(completion_year, params.T_max)].sum()),
        "C_total": float(c_total[: min(completion_year, params.T_max)].sum()),
    }


def _evaluate_innovative(params: InnovativeParams, z: np.ndarray, r_frac: np.ndarray, e_frac: np.ndarray) -> Dict:
    sch = _decode_schedule(params, z=z, r_frac=r_frac, e_frac=e_frac)
    stage_req = params.stage_require_wan_ton()
    T1, T2, T3 = params.stage_years

    stage_ranges = [(0, T1), (T1, T1 + T2), (T1 + T2, T1 + T2 + T3)]
    stage_flow = np.array([sch["total_flow"][a:b].sum() for a, b in stage_ranges], dtype=float)
    feasible_stage = np.all(stage_flow + 1e-9 >= stage_req)
    feasible_total = sch["cum_flow"][-1] + 1e-9 >= params.q_total_wan_ton
    feasible_complete = sch["completion_year"] <= params.T_max
    feasible_z = 3 <= int(np.sum(z)) <= 7
    feasible = bool(feasible_stage and feasible_total and feasible_complete and feasible_z)

    w = params.stage_weights()
    E_stage = np.array([sch["E_year"][a:b].sum() for a, b in stage_ranges], dtype=float)
    C_stage = np.array([sch["C_year"][a:b].sum() for a, b in stage_ranges], dtype=float)
    T_stage = np.array([b - a for a, b in stage_ranges], dtype=float)

    z_all = np.ones(params.n_bases, dtype=int)
    f1_max_sch = _decode_schedule(params, z=z_all, r_frac=np.ones(3), e_frac=np.ones(3))
    E_stage_max = np.array([f1_max_sch["E_year"][a:b].sum() for a, b in stage_ranges], dtype=float) + 1e-9
    C_stage_max = np.array([f1_max_sch["C_year"][a:b].sum() for a, b in stage_ranges], dtype=float) + 1e-9
    T_stage_max = T_stage.copy() + 1e-9

    mix = float(np.sum(w[:, 0] * (E_stage / E_stage_max) + w[:, 1] * (C_stage / C_stage_max) + w[:, 2] * (T_stage / T_stage_max)))

    penalty = 0.0
    if not feasible_z:
        penalty += 5.0 + abs(np.sum(z) - 5) * 0.5
    if not feasible_stage:
        penalty += float(np.sum(np.maximum(0.0, stage_req - stage_flow)) / 1000.0) * 10.0 + 10.0
    if not feasible_total:
        penalty += float(max(0.0, params.q_total_wan_ton - sch["cum_flow"][-1]) / 1000.0) * 20.0 + 10.0
    if not feasible_complete:
        penalty += 10.0 + (sch["completion_year"] - params.T_max) * 0.5

    return {
        "feasible": feasible,
        "fitness": mix + penalty,
        "mix": mix,
        "E_total": sch["E_total"],
        "C_total": sch["C_total"],
        "completion_year": sch["completion_year"],
        "z": z.astype(int),
        "r_frac": r_frac.astype(float),
        "e_frac": e_frac.astype(float),
    }


def run_ga_innovative(params: InnovativeParams, seed: int = 7, pop_size: int = 140, generations: int = 70) -> Tuple[pd.DataFrame, Dict]:
    random.seed(seed)
    np.random.seed(seed)

    def rand_z():
        k = random.randint(3, 7)
        idx = random.sample(range(params.n_bases), k)
        z = np.zeros(params.n_bases, dtype=int)
        z[idx] = 1
        return z

    def rand_frac():
        return np.random.beta(2.0, 2.0, size=3).astype(float)

    def mutate_z(z: np.ndarray, p: float = 0.15) -> np.ndarray:
        z2 = z.copy()
        for i in range(params.n_bases):
            if random.random() < p:
                z2[i] = 1 - z2[i]
        k = int(z2.sum())
        if k < 3:
            zeros = np.where(z2 == 0)[0].tolist()
            for i in random.sample(zeros, 3 - k):
                z2[i] = 1
        if k > 7:
            ones = np.where(z2 == 1)[0].tolist()
            for i in random.sample(ones, k - 7):
                z2[i] = 0
        return z2

    def mutate_frac(f: np.ndarray, sigma: float = 0.12) -> np.ndarray:
        f2 = f + np.random.normal(0.0, sigma, size=f.shape)
        return np.clip(f2, 0.0, 1.0)

    def crossover(a: Dict, b: Dict) -> Tuple[Dict, Dict]:
        mask = np.random.rand(params.n_bases) < 0.5
        z1 = np.where(mask, a["z"], b["z"]).astype(int)
        z2 = np.where(mask, b["z"], a["z"]).astype(int)
        if random.random() < 0.6:
            cut = random.randint(1, 2)
            r1 = np.concatenate([a["r_frac"][:cut], b["r_frac"][cut:]])
            r2 = np.concatenate([b["r_frac"][:cut], a["r_frac"][cut:]])
            e1 = np.concatenate([a["e_frac"][:cut], b["e_frac"][cut:]])
            e2 = np.concatenate([b["e_frac"][:cut], a["e_frac"][cut:]])
        else:
            r1, r2 = a["r_frac"].copy(), b["r_frac"].copy()
            e1, e2 = a["e_frac"].copy(), b["e_frac"].copy()
        return (
            {"z": z1, "r_frac": r1, "e_frac": e1},
            {"z": z2, "r_frac": r2, "e_frac": e2},
        )

    def tournament(pop: List[Dict], k: int = 3) -> Dict:
        cand = random.sample(pop, k)
        cand.sort(key=lambda d: d["eval"]["fitness"])
        return cand[0]

    population: List[Dict] = []
    for _ in range(pop_size):
        ind = {"z": rand_z(), "r_frac": rand_frac(), "e_frac": rand_frac()}
        ind["eval"] = _evaluate_innovative(params, ind["z"], ind["r_frac"], ind["e_frac"])
        population.append(ind)

    best = min(population, key=lambda d: d["eval"]["fitness"])

    for _g in range(generations):
        population.sort(key=lambda d: d["eval"]["fitness"])
        elite = population[: max(4, pop_size // 20)]

        children: List[Dict] = []
        while len(children) < pop_size - len(elite):
            p1 = tournament(population)
            p2 = tournament(population)
            c1, c2 = crossover(p1, p2)
            if random.random() < 0.7:
                c1["z"] = mutate_z(c1["z"])
                c1["r_frac"] = mutate_frac(c1["r_frac"])
                c1["e_frac"] = mutate_frac(c1["e_frac"])
            if random.random() < 0.7:
                c2["z"] = mutate_z(c2["z"])
                c2["r_frac"] = mutate_frac(c2["r_frac"])
                c2["e_frac"] = mutate_frac(c2["e_frac"])

            c1["eval"] = _evaluate_innovative(params, c1["z"], c1["r_frac"], c1["e_frac"])
            c2["eval"] = _evaluate_innovative(params, c2["z"], c2["r_frac"], c2["e_frac"])
            children.append(c1)
            if len(children) < pop_size - len(elite):
                children.append(c2)

        population = elite + children
        cur_best = min(population, key=lambda d: d["eval"]["fitness"])
        if cur_best["eval"]["fitness"] < best["eval"]["fitness"]:
            best = cur_best

    rows = []
    for ind in population:
        ev = ind["eval"]
        rows.append(
            {
                "feasible": ev["feasible"],
                "fitness": ev["fitness"],
                "mix": ev["mix"],
                "E_total": ev["E_total"],
                "C_total": ev["C_total"],
                "completion_year": ev["completion_year"],
                "z": "".join(map(str, ev["z"].tolist())),
                "r_frac": ",".join(f"{x:.3f}" for x in ev["r_frac"]),
                "e_frac": ",".join(f"{x:.3f}" for x in ev["e_frac"]),
                "enabled_bases": int(ev["z"].sum()),
            }
        )
    df = pd.DataFrame(rows).sort_values(["feasible", "fitness"], ascending=[False, True]).reset_index(drop=True)
    return df, best["eval"]


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    cand = df[df["feasible"] == True].copy()
    if cand.empty:
        return cand
    pts = cand[["E_total", "C_total", "completion_year"]].to_numpy(dtype=float)
    keep = np.ones(len(cand), dtype=bool)
    for i in range(len(cand)):
        if not keep[i]:
            continue
        dom = (pts[:, 0] <= pts[i, 0]) & (pts[:, 1] <= pts[i, 1]) & (pts[:, 2] <= pts[i, 2])
        strict = (pts[:, 0] < pts[i, 0]) | (pts[:, 1] < pts[i, 1]) | (pts[:, 2] < pts[i, 2])
        dominated_by_any = np.any(dom & strict)
        if dominated_by_any:
            keep[i] = False
    return cand.loc[keep].sort_values(["E_total", "C_total", "completion_year"]).reset_index(drop=True)


def plot_pareto(df: pd.DataFrame, save_path: str):
    _set_plot_style()
    pf = pareto_front(df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title("Task4 Innovative Model: Approx. Pareto (GA + AHP weights)", fontsize=13, fontweight="bold")

    feas = df[df["feasible"] == True]
    infeas = df[df["feasible"] == False]
    if not infeas.empty:
        ax.scatter(infeas["E_total"], infeas["C_total"], s=14, alpha=0.12, color="gray", label="Infeasible")
    if not feas.empty:
        sc = ax.scatter(feas["E_total"], feas["C_total"], c=feas["completion_year"], s=18, cmap="viridis", alpha=0.75, label="Feasible")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Completion Year")
    if not pf.empty:
        ax.plot(pf["E_total"], pf["C_total"], linewidth=2.2, color="tab:red", label="Pareto Front (filtered)")

    ax.set_xlabel("Total Emissions (tCO2e)")
    ax.set_ylabel("Total Cost (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    backend = str(plt.get_backend()).lower() if hasattr(plt, "get_backend") else ""
    if "agg" not in backend:
        plt.show()
    plt.close(fig)


def pick_top3(df: pd.DataFrame) -> pd.DataFrame:
    feas = df[df["feasible"] == True].copy()
    if feas.empty:
        return feas

    def sig(row: pd.Series) -> Tuple[str, str, str]:
        return str(row["z"]), str(row["r_frac"]), str(row["e_frac"])

    def pick_one(order: List[str], profile: str, chosen: set) -> Optional[pd.Series]:
        view = feas.sort_values(order, ascending=True)
        for _, r in view.iterrows():
            s = sig(r)
            if s not in chosen:
                chosen.add(s)
                r = r.copy()
                r["profile"] = profile
                return r
        return None

    chosen: set = set()
    picks: List[pd.Series] = []
    for order, profile in [
        (["E_total", "C_total", "completion_year"], "env"),
        (["C_total", "E_total", "completion_year"], "cost"),
        (["fitness", "E_total", "C_total"], "balanced"),
    ]:
        r = pick_one(order, profile, chosen)
        if r is not None:
            picks.append(r)

    if len(picks) < 3:
        view = feas.sort_values(["fitness", "E_total", "C_total"], ascending=True)
        for _, r in view.iterrows():
            s = sig(r)
            if s in chosen:
                continue
            chosen.add(s)
            r = r.copy()
            r["profile"] = f"alt_{len(picks)+1}"
            picks.append(r)
            if len(picks) >= 3:
                break

    top = pd.DataFrame(picks)
    return top[["profile", "enabled_bases", "completion_year", "E_total", "C_total", "mix", "fitness", "z", "r_frac", "e_frac"]].reset_index(drop=True)


def main():
    basic_params = BasicEnvParams()
    print("=" * 70)
    print("Task4: 基础模型（MILP + 约束松弛）求解中...")
    print("=" * 70)
    basic_sol = solve_basic_milp(basic_params, solver="auto")
    df_basic = summarize_basic_solution(basic_sol)
    df_basic.to_csv("task4_basic_results.csv", index=False, encoding="utf-8")

    demand_met_year = int(np.where(df_basic["Cum_Flow_wan_ton"].to_numpy() >= basic_params.q_total_wan_ton)[0][0] + 1)
    total_slack = float(df_basic["Slack_tCO2e"].sum())
    total_emission = float(df_basic["E_Total_tCO2e"].sum())
    total_flow = float(df_basic["Total_Flow_wan_ton"].sum())

    print(f"[Basic] Status: {basic_sol['status']}")
    print(f"[Basic] Objective: {basic_sol['objective']:.4e}")
    print(f"[Basic] Total Flow: {total_flow:.2f} 万公吨")
    print(f"[Basic] Demand Met Year: {demand_met_year}/{basic_params.T}")
    print(f"[Basic] Total Emission: {total_emission:.2e} tCO2e")
    print(f"[Basic] Total Slack: {total_slack:.2e} tCO2e")

    plot_basic_dashboard(df_basic, "task4_basic_dashboard.png")
    plot_flowchart("task4_flowchart.png")

    print("\n" + "=" * 70)
    print("Task4: 创新模型（GA + AHP）搜索中...")
    print("=" * 70)
    inv_params = InnovativeParams(T_max=basic_params.T)
    df_ga, best_eval = run_ga_innovative(inv_params, seed=7, pop_size=140, generations=70)
    plot_pareto(df_ga, "task4_pareto.png")
    top3 = pick_top3(df_ga)
    top3.to_csv("task4_innovative_top3.csv", index=False, encoding="utf-8")

    if not top3.empty:
        print("[Innovative] Top3 (env/cost/balanced) 已导出到 task4_innovative_top3.csv")
        print(top3.to_string(index=False))
    else:
        print("[Innovative] 未找到可行解，已输出候选分布图 task4_pareto.png 供调参")

    print("\n输出文件：")
    print("- task4_flowchart.png")
    print("- task4_basic_dashboard.png")
    print("- task4_pareto.png")
    print("- task4_basic_results.csv")
    print("- task4_innovative_top3.csv")


if __name__ == "__main__":
    main()
