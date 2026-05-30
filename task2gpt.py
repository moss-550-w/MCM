"""         
已根据 [task2modle.md](file:///Users/a11111/Downloads/%E7%BE%8E%E8%B5%9B/Code/task2modle.md) 落地生成可运行的鲁棒优化脚本 [task2.py](file:///Users/a11111/Downloads/%E7%BE%8E%E8%B5%9B/Code/task2.py)。

- 参数与随机场景：在 [task2.py:L29-L90](file:///Users/a11111/Downloads/%E7%BE%8E%E8%B5%9B/Code/task2.py#L29-L90) 定义了 $Q_{rock},C_{elevator},p_{fail},\\theta$ 等，并按蒙特卡洛生成 $\delta_{i,t}$（基地故障）、摆动折减系数与电梯停运状态。
- DT-MILP 求解：在 [solve_scenario](file:///Users/a11111/Downloads/%E7%BE%8E%E8%B5%9B/Code/task2.py#L100-L270) 中实现离散时间 MILP（优先 Gurobi，否则 PuLP-CBC），核心变量为火箭发射次数整数 $x_{i,t}$、电梯运量连续 $y_{j,t}$、电梯建设二元 $w_j$，并用累计供给与缺口惩罚对应你文档里的总量约束与 $(\cdot)^+$ 逻辑。
- 蒙特卡洛验证与偏差率：在 [run_monte_carlo](file:///Users/a11111/Downloads/%E7%BE%8E%E8%B5%9B/Code/task2.py#L273-L332) 计算 $F_{det}$、$\\mathbb{E}[F_{robust}]$、完成周期统计与偏差率（对应公式(8)的验证口径）。

运行示例（仓库根目录）：
```bash
python task2.py --solver pulp --n-sim 50 --seed 42 --time-limit 120
# 或用 Gurobi：
python task2.py --solver gurobi --n-sim 200 --seed 42 --time-limit 120 --mip-gap 0.02
```
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

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
class RobustParams:
    T: int = 160
    n_bases: int = 10
    n_ports: int = 3

    W_goal_mt: float = 100_000_000.0
    Q_rock_mt: float = 150.0
    x_max_per_base: int = 200

    C_elevator_mt_per_year: float = 179_000.0
    elevator_setup_years: int = 10

    p_fail_base: float = 0.20
    p_swing: float = 0.10
    theta_swing: float = 0.70
    p_elevator_down: float = 0.02

    rocket_cost_per_launch_usd: float = 50_000_000.0
    elevator_cost_per_mt_usd: float = 500.0
    elevator_setup_cost_usd: float = 500_000_000_000.0
    C_repair_usd: float = 1_000_000_000.0

    rocket_env_tco2_per_launch: float = 500.0
    elevator_env_tco2_per_mt: float = 0.01

    alpha_time: float = 1.0
    beta_cost: float = 1.0
    gamma_env: float = 1.0
    lambda_shortage: float = 50.0

    def setup_periods(self) -> int:
        return int(max(0, self.elevator_setup_years))

    def x_max_vec(self) -> np.ndarray:
        return np.full(self.n_bases, int(self.x_max_per_base), dtype=int)

    def y_max_vec(self, theta: np.ndarray, elevator_up: np.ndarray) -> np.ndarray:
        cap = float(self.C_elevator_mt_per_year)
        return cap * theta * elevator_up


@dataclass(frozen=True)
class ScenarioData:
    base_failed: np.ndarray
    theta: np.ndarray
    elevator_up: np.ndarray


def sample_scenario(params: RobustParams, rng: np.random.Generator, mode: str) -> ScenarioData:
    I, J, T = params.n_bases, params.n_ports, params.T
    if mode == "deterministic":
        base_failed = np.zeros((I, T), dtype=int)
        theta = np.ones((J, T), dtype=float)
        elevator_up = np.ones((J, T), dtype=int)
        return ScenarioData(base_failed=base_failed, theta=theta, elevator_up=elevator_up)

    base_failed = (rng.random((I, T)) < float(params.p_fail_base)).astype(int)
    swing = (rng.random((J, T)) < float(params.p_swing)).astype(int)
    theta = np.where(swing > 0, float(params.theta_swing), 1.0).astype(float)
    elevator_up = (rng.random((J, T)) >= float(params.p_elevator_down)).astype(int)
    return ScenarioData(base_failed=base_failed, theta=theta, elevator_up=elevator_up)


def _completion_period(cum: np.ndarray, goal: float) -> Optional[int]:
    idx = np.where(cum + 1e-9 >= goal)[0]
    if idx.size == 0:
        return None
    return int(idx[0] + 1)


def solve_scenario(
    params: RobustParams,
    scenario: ScenarioData,
    solver: str = "auto",
    time_limit_s: int = 120,
    mip_gap: float = 0.02,
) -> Dict:
    T = params.T
    I = params.n_bases
    J = params.n_ports

    base_failed = np.asarray(scenario.base_failed, dtype=int)
    theta = np.asarray(scenario.theta, dtype=float)
    elevator_up = np.asarray(scenario.elevator_up, dtype=int)

    x_max = params.x_max_vec()
    y_max = params.y_max_vec(theta=theta, elevator_up=elevator_up)
    setup_periods = params.setup_periods()

    repair_events = int(base_failed.sum() + int((1 - elevator_up).sum()))
    repair_cost = float(params.C_repair_usd) * float(repair_events)

    use_gurobi = (solver == "gurobi") or (solver == "auto" and GUROBI_AVAILABLE)
    if use_gurobi:
        m = gp.Model("Task2_Robust_MILP")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = float(time_limit_s)
        m.Params.MIPGap = float(mip_gap)

        x = m.addVars(I, T, vtype=GRB.INTEGER, lb=0.0, name="x")
        y = m.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
        w = m.addVars(J, vtype=GRB.BINARY, name="w")
        cum = m.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, name="cum")
        rem = m.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, name="rem")
        slack = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="slack")

        for i in range(I):
            for t in range(T):
                m.addConstr(x[i, t] <= float(x_max[i]))

        for j in range(J):
            for t in range(T):
                if t < setup_periods:
                    m.addConstr(y[j, t] <= 0.0)
                else:
                    m.addConstr(y[j, t] <= float(y_max[j, t]) * w[j])

        for t in range(T):
            delivered_rocket = gp.quicksum(float(params.Q_rock_mt) * x[i, t] * (1.0 - float(base_failed[i, t])) for i in range(I))
            delivered = delivered_rocket + gp.quicksum(y[j, t] for j in range(J))
            if t == 0:
                m.addConstr(cum[t] == delivered)
            else:
                m.addConstr(cum[t] == cum[t - 1] + delivered)

            m.addConstr(rem[t] >= float(params.W_goal_mt) - cum[t])

        m.addConstr(slack >= float(params.W_goal_mt) - cum[T - 1])

        cost = gp.quicksum(float(params.rocket_cost_per_launch_usd) * x[i, t] for i in range(I) for t in range(T))
        cost += gp.quicksum(float(params.elevator_cost_per_mt_usd) * y[j, t] for j in range(J) for t in range(T))
        cost += gp.quicksum(float(params.elevator_setup_cost_usd) * w[j] for j in range(J))
        cost += float(repair_cost)

        emission = gp.quicksum(float(params.rocket_env_tco2_per_launch) * x[i, t] for i in range(I) for t in range(T))
        emission += gp.quicksum(float(params.elevator_env_tco2_per_mt) * y[j, t] for j in range(J) for t in range(T))

        time_term = gp.quicksum(rem[t] for t in range(T)) / float(params.W_goal_mt)
        shortage_term = slack / float(params.W_goal_mt)

        obj = (
            float(params.alpha_time) * time_term
            + float(params.beta_cost) * (cost / 1e12)
            + float(params.gamma_env) * (emission / 1e9)
            + float(params.lambda_shortage) * shortage_term
        )
        m.setObjective(obj, GRB.MINIMIZE)

        m.optimize()
        status = str(m.Status)

        x_sol = np.array([[x[i, t].X for t in range(T)] for i in range(I)], dtype=float)
        y_sol = np.array([[y[j, t].X for t in range(T)] for j in range(J)], dtype=float)
        w_sol = np.array([w[j].X for j in range(J)], dtype=float)
        cum_sol = np.array([cum[t].X for t in range(T)], dtype=float)
        rem_sol = np.array([rem[t].X for t in range(T)], dtype=float)
        slack_sol = float(slack.X)
        objective = float(m.ObjVal) if m.SolCount > 0 else float("nan")

        return {
            "status": status,
            "objective": objective,
            "x": x_sol,
            "y": y_sol,
            "w": w_sol,
            "cum": cum_sol,
            "rem": rem_sol,
            "slack": slack_sol,
            "repair_cost": repair_cost,
        }

    prob = LpProblem("Task2_Robust_MILP", LpMinimize)
    x = LpVariable.dicts("x", (range(I), range(T)), lowBound=0, cat="Integer")
    y = LpVariable.dicts("y", (range(J), range(T)), lowBound=0, cat="Continuous")
    w = LpVariable.dicts("w", range(J), cat="Binary")
    cum = LpVariable.dicts("cum", range(T), lowBound=0, cat="Continuous")
    rem = LpVariable.dicts("rem", range(T), lowBound=0, cat="Continuous")
    slack = LpVariable("slack", lowBound=0, cat="Continuous")

    for i in range(I):
        for t in range(T):
            prob += x[i][t] <= int(x_max[i])

    for j in range(J):
        for t in range(T):
            if t < setup_periods:
                prob += y[j][t] <= 0.0
            else:
                prob += y[j][t] <= float(y_max[j, t]) * w[j]

    for t in range(T):
        delivered_rocket = lpSum(float(params.Q_rock_mt) * x[i][t] * (1.0 - float(base_failed[i, t])) for i in range(I))
        delivered = delivered_rocket + lpSum(y[j][t] for j in range(J))
        if t == 0:
            prob += cum[t] == delivered
        else:
            prob += cum[t] == cum[t - 1] + delivered
        prob += rem[t] >= float(params.W_goal_mt) - cum[t]

    prob += slack >= float(params.W_goal_mt) - cum[T - 1]

    cost = lpSum(float(params.rocket_cost_per_launch_usd) * x[i][t] for i in range(I) for t in range(T))
    cost += lpSum(float(params.elevator_cost_per_mt_usd) * y[j][t] for j in range(J) for t in range(T))
    cost += lpSum(float(params.elevator_setup_cost_usd) * w[j] for j in range(J))
    cost += float(repair_cost)

    emission = lpSum(float(params.rocket_env_tco2_per_launch) * x[i][t] for i in range(I) for t in range(T))
    emission += lpSum(float(params.elevator_env_tco2_per_mt) * y[j][t] for j in range(J) for t in range(T))

    time_term = lpSum(rem[t] for t in range(T)) / float(params.W_goal_mt)
    shortage_term = slack / float(params.W_goal_mt)

    prob += (
        float(params.alpha_time) * time_term
        + float(params.beta_cost) * (cost / 1e12)
        + float(params.gamma_env) * (emission / 1e9)
        + float(params.lambda_shortage) * shortage_term
    )

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=int(time_limit_s)))
    status = LpStatus.get(prob.status, str(prob.status))

    x_sol = np.array([[value(x[i][t]) for t in range(T)] for i in range(I)], dtype=float)
    y_sol = np.array([[value(y[j][t]) for t in range(T)] for j in range(J)], dtype=float)
    w_sol = np.array([value(w[j]) for j in range(J)], dtype=float)
    cum_sol = np.array([value(cum[t]) for t in range(T)], dtype=float)
    rem_sol = np.array([value(rem[t]) for t in range(T)], dtype=float)
    slack_sol = float(value(slack))
    objective = float(value(prob.objective))

    return {
        "status": status,
        "objective": objective,
        "x": x_sol,
        "y": y_sol,
        "w": w_sol,
        "cum": cum_sol,
        "rem": rem_sol,
        "slack": slack_sol,
        "repair_cost": repair_cost,
    }


def run_monte_carlo(
    params: RobustParams,
    n_sim: int,
    seed: int,
    solver: str,
    time_limit_s: int,
    mip_gap: float,
) -> Dict:
    rng = np.random.default_rng(int(seed))

    det = sample_scenario(params, rng=rng, mode="deterministic")
    det_sol = solve_scenario(params, det, solver=solver, time_limit_s=time_limit_s, mip_gap=mip_gap)
    det_completion = _completion_period(det_sol["cum"], params.W_goal_mt)

    objs = []
    completions = []
    slacks = []

    for _ in range(int(n_sim)):
        scen = sample_scenario(params, rng=rng, mode="stochastic")
        sol = solve_scenario(params, scen, solver=solver, time_limit_s=time_limit_s, mip_gap=mip_gap)
        objs.append(float(sol["objective"]))
        completions.append(_completion_period(sol["cum"], params.W_goal_mt))
        slacks.append(float(sol["slack"]))

    obj_arr = np.asarray(objs, dtype=float)
    slack_arr = np.asarray(slacks, dtype=float)
    comp_arr = np.array([c if c is not None else np.nan for c in completions], dtype=float)

    det_obj = float(det_sol["objective"])
    robust_obj_mean = float(np.nanmean(obj_arr))
    obj_valid = np.isfinite(obj_arr)
    robust_obj_std = float(np.std(obj_arr[obj_valid], ddof=1)) if int(obj_valid.sum()) > 1 else 0.0

    comp_mean = float(np.nanmean(comp_arr))
    comp_valid = np.isfinite(comp_arr)
    comp_std = float(np.std(comp_arr[comp_valid], ddof=1)) if int(comp_valid.sum()) > 1 else 0.0

    denom = abs(det_obj) if abs(det_obj) > 1e-9 else 1.0
    deviation = float(abs(robust_obj_mean - det_obj) / denom)

    return {
        "deterministic": {
            "status": det_sol["status"],
            "objective": det_obj,
            "completion_period": det_completion,
            "slack": float(det_sol["slack"]),
            "w": det_sol["w"].copy(),
        },
        "robust_mc": {
            "n": int(n_sim),
            "seed": int(seed),
            "objective_mean": robust_obj_mean,
            "objective_std": robust_obj_std,
            "completion_mean": comp_mean,
            "completion_std": comp_std,
            "slack_mean": float(np.mean(slack_arr)),
            "deviation_ratio": deviation,
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=160)
    p.add_argument("--n-sim", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--solver", type=str, default="auto", choices=["auto", "gurobi", "pulp"])
    p.add_argument("--time-limit", type=int, default=120)
    p.add_argument("--mip-gap", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    params = RobustParams(T=int(args.T))
    res = run_monte_carlo(
        params=params,
        n_sim=int(args.n_sim),
        seed=int(args.seed),
        solver=str(args.solver),
        time_limit_s=int(args.time_limit),
        mip_gap=float(args.mip_gap),
    )

    det = res["deterministic"]
    mc = res["robust_mc"]

    det_completion = det["completion_period"]
    det_completion_s = "未完成" if det_completion is None else str(det_completion)

    comp_mean = mc["completion_mean"]
    comp_mean_s = "未完成" if not math.isfinite(comp_mean) else f"{comp_mean:.2f}"

    print("=" * 70)
    print("Task2: 非完美工况鲁棒性优化模型（DT-MILP + 蒙特卡洛）")
    print("=" * 70)
    print(f"求解器: {args.solver} (Gurobi可用={GUROBI_AVAILABLE})")
    print(f"T={params.T}, 基地={params.n_bases}, 港口={params.n_ports}")
    print(f"W_goal={params.W_goal_mt:.0f} MT")
    print("-" * 70)
    print(f"确定性基准: status={det['status']}, F_det={det['objective']:.6g}, 完成周期={det_completion_s}")
    print(f"电梯建设决策w: {np.round(det['w'], 3).tolist()}")
    print("-" * 70)
    print(f"蒙特卡洛: N={mc['n']}, seed={mc['seed']}")
    print(f"E[F_robust]={mc['objective_mean']:.6g} ± {mc['objective_std']:.3g}")
    print(f"E[完成周期]={comp_mean_s} ± {mc['completion_std']:.3g}")
    print(f"偏差率 |E[F]-F_det|/|F_det| = {mc['deviation_ratio']:.3%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
