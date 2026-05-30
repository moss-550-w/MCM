"""
- SAA + “Benders式分解”求解 ：先采样 SAA 场景集；主问题只保留离散决策（火箭在“正常/故障”两种状态下的预案发射量、是否建设电梯港口），子问题用“解析可消去”的方式把电梯最大可补偿运力并入，从而把每个场景的缺口表示成线性短缺变量，等价于把子问题的最优值回传到主问题（可理解为解析子问题后的 Benders 形式）。
- 提炼调度规则 ：从最优解中提取每期总发射目标、建设决策、以及期望到货轨迹（planned cumulative）。
- 转化为启发式并验证 ：启发式按“电梯优先 + 火箭补足 + 追赶项(catch-up) + 末期必达(required-rate)”动态调度，并在独立随机场景集上输出统计结果（完成期、缺口概率、成本、目标值等）。
运行方式

- 小规模快速跑通：
  ```
  python task2_new.py --solver pulp --time-limit 
  30 --n-saa 30 --n-eval 30 --seed 1
  ```
- 默认自动优先用 Gurobi（可用则用）：
  ```
  python task2_new.py --solver auto --n-saa 200 
  --n-eval 200 --seed 42
  ```
如果你希望“规则提炼”输出更像论文里的可直接引用条目（例如分建设期/运营期的明确阈值、故障比例触发的增发系数等），我可以在不改核心算法的前提下，把提炼结果结构化成更清晰的规则表。
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)


@dataclass(frozen=True)
class Params:
    T: int = 160
    n_bases: int = 10
    n_ports: int = 3

    W_goal_mt: float = 100_000_000.0
    eta_supply: float = 1.0

    Q_rock_mt: float = 150.0
    x_max_per_base: int = 200

    C_elevator_mt_per_year: float = 179_000.0
    elevator_setup_years: int = 10

    p_fail_base: float = 0.20
    p_swing: float = 0.10
    theta_swing: float = 0.70
    p_elevator_down: float = 0.02

    rocket_cost_per_launch_usd: float = 50_000_000.0
    rocket_env_tco2_per_launch: float = 500.0

    C_repair_usd: float = 1_000_000_000.0

    alpha_time: float = 1.0
    beta_cost: float = 1.0
    gamma_env: float = 1.0
    lambda_shortage: float = 50.0

    catchup_gain: float = 0.8

    def setup_periods(self) -> int:
        return int(max(0, self.elevator_setup_years))

    def expected_theta(self) -> float:
        return (1.0 - float(self.p_swing)) + float(self.p_swing) * float(self.theta_swing)

    def expected_elevator_up(self) -> float:
        return 1.0 - float(self.p_elevator_down)


@dataclass(frozen=True)
class Scenario:
    base_failed: np.ndarray
    theta: np.ndarray
    elevator_up: np.ndarray


@dataclass(frozen=True)
class SAASolution:
    x_ok: np.ndarray
    x_fail: np.ndarray
    w: np.ndarray
    expected_shortfall: float
    expected_repair_cost: float
    objective: float


def _safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return float(a) / float(b)


def sample_scenarios(params: Params, n: int, seed: int) -> List[Scenario]:
    rng = np.random.default_rng(int(seed))
    I, J, T = params.n_bases, params.n_ports, params.T

    scenarios: List[Scenario] = []
    for _ in range(int(n)):
        base_failed = (rng.random((I, T)) < float(params.p_fail_base)).astype(int)
        swing = (rng.random((J, T)) < float(params.p_swing)).astype(int)
        theta = np.where(swing > 0, float(params.theta_swing), 1.0).astype(float)
        elevator_up = (rng.random((J, T)) >= float(params.p_elevator_down)).astype(int)
        scenarios.append(Scenario(base_failed=base_failed, theta=theta, elevator_up=elevator_up))
    return scenarios


def _scenario_repair_cost(params: Params, scenario: Scenario) -> float:
    repair_events = int(scenario.base_failed.sum() + int((1 - scenario.elevator_up).sum()))
    return float(params.C_repair_usd) * float(repair_events)


def solve_saa_benders_like(
    params: Params,
    scenarios: List[Scenario],
    solver: str,
    time_limit_s: int,
    mip_gap: float,
) -> SAASolution:
    I, J, T = params.n_bases, params.n_ports, params.T
    N = int(len(scenarios))
    setup_periods = params.setup_periods()

    repair_costs = np.array([_scenario_repair_cost(params, sc) for sc in scenarios], dtype=float)
    expected_repair_cost = float(repair_costs.mean()) if N > 0 else 0.0

    def _should_fallback_to_pulp(err: BaseException) -> bool:
        msg = str(err).lower()
        if "model too large" in msg:
            return True
        if "size-limited license" in msg:
            return True
        if "restricted license" in msg:
            return True
        return False

    use_gurobi = (solver == "gurobi") or (solver == "auto" and GUROBI_AVAILABLE)
    if use_gurobi:
        try:
            m = gp.Model("Task2_SAA_Benders_Master")
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = float(time_limit_s)
            m.Params.MIPGap = float(mip_gap)

            x_ok = m.addVars(I, T, vtype=GRB.INTEGER, lb=0.0, ub=float(params.x_max_per_base), name="x_ok")
            x_fail = m.addVars(I, T, vtype=GRB.INTEGER, lb=0.0, ub=float(params.x_max_per_base), name="x_fail")
            w = m.addVars(J, vtype=GRB.BINARY, name="w")
            s = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, name="shortfall")

            target = float(params.W_goal_mt) * float(params.eta_supply)

            for m_idx, sc in enumerate(scenarios):
                base_failed = np.asarray(sc.base_failed, dtype=float)
                theta = np.asarray(sc.theta, dtype=float)
                elevator_up = np.asarray(sc.elevator_up, dtype=float)

                rocket_delivered = gp.quicksum(
                    float(params.Q_rock_mt)
                    * ((1.0 - base_failed[i, t]) * x_ok[i, t] + base_failed[i, t] * x_fail[i, t])
                    for i in range(I)
                    for t in range(T)
                )

                elevator_cap = gp.quicksum(
                    float(params.C_elevator_mt_per_year) * theta[j, t] * elevator_up[j, t] * w[j]
                    for j in range(J)
                    for t in range(setup_periods, T)
                )

                m.addConstr(s[m_idx] >= target - rocket_delivered - elevator_cap)

            rocket_cost = gp.quicksum(
                float(params.rocket_cost_per_launch_usd) * (x_ok[i, t] + x_fail[i, t]) for i in range(I) for t in range(T)
            )
            rocket_env = gp.quicksum(
                float(params.rocket_env_tco2_per_launch) * (x_ok[i, t] + x_fail[i, t]) for i in range(I) for t in range(T)
            )

            expected_short = gp.quicksum(s[m_idx] for m_idx in range(N)) / float(max(N, 1))

            time_term = float(T) / float(max(T, 1))
            cost_term = (rocket_cost + float(expected_repair_cost)) / 1e12
            env_term = rocket_env / 1e9
            shortage_term = expected_short / float(params.W_goal_mt)

            obj = (
                float(params.alpha_time) * time_term
                + float(params.beta_cost) * cost_term
                + float(params.gamma_env) * env_term
                + float(params.lambda_shortage) * shortage_term
            )
            m.setObjective(obj, GRB.MINIMIZE)

            m.optimize()

            x_ok_sol = np.array([[x_ok[i, t].X for t in range(T)] for i in range(I)], dtype=float)
            x_fail_sol = np.array([[x_fail[i, t].X for t in range(T)] for i in range(I)], dtype=float)
            w_sol = np.array([w[j].X for j in range(J)], dtype=float)
            expected_shortfall = float(np.mean([s[m_idx].X for m_idx in range(N)])) if N > 0 else 0.0
            objective = float(m.ObjVal) if m.SolCount > 0 else float("nan")

            return SAASolution(
                x_ok=x_ok_sol,
                x_fail=x_fail_sol,
                w=w_sol,
                expected_shortfall=expected_shortfall,
                expected_repair_cost=expected_repair_cost,
                objective=objective,
            )
        except Exception as e:
            if _should_fallback_to_pulp(e) or solver == "auto":
                pass
            else:
                raise

    prob = LpProblem("Task2_SAA_Benders_Master", LpMinimize)
    x_ok = LpVariable.dicts("x_ok", (range(I), range(T)), lowBound=0, upBound=int(params.x_max_per_base), cat="Integer")
    x_fail = LpVariable.dicts("x_fail", (range(I), range(T)), lowBound=0, upBound=int(params.x_max_per_base), cat="Integer")
    w = LpVariable.dicts("w", range(J), cat="Binary")
    s = LpVariable.dicts("shortfall", range(N), lowBound=0, cat="Continuous")

    target = float(params.W_goal_mt) * float(params.eta_supply)
    for m_idx, sc in enumerate(scenarios):
        base_failed = np.asarray(sc.base_failed, dtype=float)
        theta = np.asarray(sc.theta, dtype=float)
        elevator_up = np.asarray(sc.elevator_up, dtype=float)

        rocket_delivered = lpSum(
            float(params.Q_rock_mt)
            * ((1.0 - float(base_failed[i, t])) * x_ok[i][t] + float(base_failed[i, t]) * x_fail[i][t])
            for i in range(I)
            for t in range(T)
        )

        elevator_cap = lpSum(
            float(params.C_elevator_mt_per_year) * float(theta[j, t]) * float(elevator_up[j, t]) * w[j]
            for j in range(J)
            for t in range(setup_periods, T)
        )
        prob += s[m_idx] >= target - rocket_delivered - elevator_cap

    rocket_cost = lpSum(float(params.rocket_cost_per_launch_usd) * (x_ok[i][t] + x_fail[i][t]) for i in range(I) for t in range(T))
    rocket_env = lpSum(float(params.rocket_env_tco2_per_launch) * (x_ok[i][t] + x_fail[i][t]) for i in range(I) for t in range(T))

    expected_short = lpSum(s[m_idx] for m_idx in range(N)) / float(max(N, 1))

    time_term = float(T) / float(max(T, 1))
    cost_term = (rocket_cost + float(expected_repair_cost)) / 1e12
    env_term = rocket_env / 1e9
    shortage_term = expected_short / float(params.W_goal_mt)

    prob += (
        float(params.alpha_time) * time_term
        + float(params.beta_cost) * cost_term
        + float(params.gamma_env) * env_term
        + float(params.lambda_shortage) * shortage_term
    )

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=int(time_limit_s)))
    status = LpStatus.get(prob.status, str(prob.status))

    x_ok_sol = np.array([[value(x_ok[i][t]) for t in range(T)] for i in range(I)], dtype=float)
    x_fail_sol = np.array([[value(x_fail[i][t]) for t in range(T)] for i in range(I)], dtype=float)
    w_sol = np.array([value(w[j]) for j in range(J)], dtype=float)
    expected_shortfall = float(np.mean([value(s[m_idx]) for m_idx in range(N)])) if N > 0 else 0.0
    objective = float(value(prob.objective))

    if status not in {"Optimal", "Not Solved"}:
        objective = objective

    return SAASolution(
        x_ok=x_ok_sol,
        x_fail=x_fail_sol,
        w=w_sol,
        expected_shortfall=expected_shortfall,
        expected_repair_cost=expected_repair_cost,
        objective=objective,
    )


def _expected_plan_trajectory(params: Params, sol: SAASolution) -> Tuple[np.ndarray, np.ndarray]:
    I, T = params.n_bases, params.T
    setup = params.setup_periods()

    p = float(params.p_fail_base)
    exp_theta = float(params.expected_theta())
    exp_up = float(params.expected_elevator_up())

    expected_rocket = np.zeros(T, dtype=float)
    for t in range(T):
        expected_rocket[t] = float(params.Q_rock_mt) * float(np.sum((1.0 - p) * sol.x_ok[:, t] + p * sol.x_fail[:, t]))

    expected_elevator = np.zeros(T, dtype=float)
    for t in range(T):
        if t < setup:
            expected_elevator[t] = 0.0
        else:
            expected_elevator[t] = float(params.C_elevator_mt_per_year) * float(exp_theta) * float(exp_up) * float(np.sum(sol.w))

    expected_delivery = expected_rocket + expected_elevator
    planned_cum = np.cumsum(expected_delivery)
    return expected_delivery, planned_cum


def extract_rules(params: Params, sol: SAASolution) -> Dict:
    setup = params.setup_periods()
    total_launches_per_t = np.sum(sol.x_ok + sol.x_fail, axis=0)
    build_mean = float(np.mean(total_launches_per_t[:setup])) if setup > 0 else float(np.mean(total_launches_per_t))
    post_mean = float(np.mean(total_launches_per_t[setup:])) if setup < params.T else 0.0

    expected_delivery, planned_cum = _expected_plan_trajectory(params, sol)

    return {
        "w": sol.w.copy(),
        "total_launches_per_t": total_launches_per_t.copy(),
        "build_mean_launches": build_mean,
        "post_mean_launches": post_mean,
        "expected_delivery_per_t": expected_delivery,
        "planned_cum_delivery": planned_cum,
    }


def _allocate_launches_evenly(
    launches_needed: int,
    available: np.ndarray,
    x_max: int,
) -> np.ndarray:
    I = int(available.size)
    x = np.zeros(I, dtype=int)
    idx = np.where(available > 0)[0]
    if idx.size == 0 or launches_needed <= 0:
        return x

    k = int(idx.size)
    cap = int(x_max) * k
    launches = min(int(launches_needed), cap)
    base = launches // k
    rem = launches % k
    x[idx] = int(base)
    if rem > 0:
        x[idx[:rem]] += 1
    return x


def simulate_heuristic(params: Params, scenario: Scenario, rules: Dict) -> Dict:
    I, J, T = params.n_bases, params.n_ports, params.T
    setup = params.setup_periods()

    total_launches_target = np.asarray(rules["total_launches_per_t"], dtype=float)
    planned_cum = np.asarray(rules["planned_cum_delivery"], dtype=float)
    w = np.asarray(rules["w"], dtype=float)

    W = 0.0
    rocket_launches_total = 0.0
    elevator_total = 0.0

    for t in range(T):
        if W >= float(params.W_goal_mt):
            break

        remaining = float(params.W_goal_mt) - float(W)

        base_failed_t = np.asarray(scenario.base_failed[:, t], dtype=int)
        available = (1 - base_failed_t).astype(int)

        elevator_cap = 0.0
        if t >= setup:
            theta_t = np.asarray(scenario.theta[:, t], dtype=float)
            up_t = np.asarray(scenario.elevator_up[:, t], dtype=float)
            elevator_cap = float(params.C_elevator_mt_per_year) * float(np.sum(w * theta_t * up_t))

        target_cum_t = float(planned_cum[t])
        launch_target = int(round(float(total_launches_target[t])))
        desired = min(remaining, float(rules["expected_delivery_per_t"][t]))
        desired = max(desired, min(remaining, float(launch_target) * float(params.Q_rock_mt)))
        desired += float(params.catchup_gain) * max(0.0, target_cum_t - float(W))
        periods_left = max(1, int(T - t))
        required_rate = float(remaining) / float(periods_left)
        max_rocket_launches = int(np.sum(available)) * int(params.x_max_per_base)
        max_delivery = float(elevator_cap) + float(max_rocket_launches) * float(params.Q_rock_mt)
        desired = max(desired, min(remaining, 1.05 * required_rate))
        desired = min(desired, remaining, max_delivery)

        elevator_deliver = min(float(elevator_cap), float(desired))
        desired_after_elevator = max(0.0, float(desired) - float(elevator_deliver))

        launches_needed = int(math.ceil(desired_after_elevator / float(params.Q_rock_mt))) if desired_after_elevator > 0 else 0
        launches_needed = min(launches_needed, max_rocket_launches)
        x = _allocate_launches_evenly(launches_needed=launches_needed, available=available, x_max=int(params.x_max_per_base))
        rocket_deliver = float(np.sum(x)) * float(params.Q_rock_mt)

        W += float(elevator_deliver) + float(rocket_deliver)
        rocket_launches_total += float(np.sum(x))
        elevator_total += float(elevator_deliver)

    shortfall = max(0.0, float(params.W_goal_mt) - float(W))

    repair_cost = _scenario_repair_cost(params, scenario)
    rocket_cost = float(params.rocket_cost_per_launch_usd) * float(rocket_launches_total)
    total_cost = rocket_cost + float(repair_cost)

    rocket_env = float(params.rocket_env_tco2_per_launch) * float(rocket_launches_total)

    objective = (
        float(params.alpha_time) * (float(t + 1) / float(max(T, 1)))
        + float(params.beta_cost) * (total_cost / 1e12)
        + float(params.gamma_env) * (rocket_env / 1e9)
        + float(params.lambda_shortage) * (shortfall / float(params.W_goal_mt))
    )

    return {
        "completion_t": int(t + 1),
        "shortfall": float(shortfall),
        "rocket_launches": float(rocket_launches_total),
        "elevator_mt": float(elevator_total),
        "repair_cost": float(repair_cost),
        "total_cost": float(total_cost),
        "rocket_env": float(rocket_env),
        "objective": float(objective),
    }


def simulate_heuristic_trace(params: Params, scenario: Scenario, rules: Dict) -> Dict:
    I, J, T = params.n_bases, params.n_ports, params.T
    setup = params.setup_periods()

    total_launches_target = np.asarray(rules["total_launches_per_t"], dtype=float)
    planned_cum = np.asarray(rules["planned_cum_delivery"], dtype=float)
    expected_delivery_per_t = np.asarray(rules["expected_delivery_per_t"], dtype=float)
    w = np.asarray(rules["w"], dtype=float)

    rocket_mt_t = np.zeros(T, dtype=float)
    elevator_mt_t = np.zeros(T, dtype=float)
    cum_mt_t = np.zeros(T, dtype=float)
    launches_i_t = np.zeros((I, T), dtype=float)
    available_bases_t = np.zeros(T, dtype=float)

    W = 0.0
    completion_t = T
    for t in range(T):
        if W >= float(params.W_goal_mt):
            completion_t = t
            cum_mt_t[t:] = float(W)
            break

        remaining = float(params.W_goal_mt) - float(W)

        base_failed_t = np.asarray(scenario.base_failed[:, t], dtype=int)
        available = (1 - base_failed_t).astype(int)
        available_bases_t[t] = float(np.sum(available))

        elevator_cap = 0.0
        if t >= setup:
            theta_t = np.asarray(scenario.theta[:, t], dtype=float)
            up_t = np.asarray(scenario.elevator_up[:, t], dtype=float)
            elevator_cap = float(params.C_elevator_mt_per_year) * float(np.sum(w * theta_t * up_t))

        target_cum_t = float(planned_cum[t])
        launch_target = int(round(float(total_launches_target[t])))
        desired = min(remaining, float(expected_delivery_per_t[t]))
        desired = max(desired, min(remaining, float(launch_target) * float(params.Q_rock_mt)))
        desired += float(params.catchup_gain) * max(0.0, target_cum_t - float(W))
        periods_left = max(1, int(T - t))
        required_rate = float(remaining) / float(periods_left)
        max_rocket_launches = int(np.sum(available)) * int(params.x_max_per_base)
        max_delivery = float(elevator_cap) + float(max_rocket_launches) * float(params.Q_rock_mt)
        desired = max(desired, min(remaining, 1.05 * required_rate))
        desired = min(desired, remaining, max_delivery)

        elevator_deliver = min(float(elevator_cap), float(desired))
        desired_after_elevator = max(0.0, float(desired) - float(elevator_deliver))
        launches_needed = int(math.ceil(desired_after_elevator / float(params.Q_rock_mt))) if desired_after_elevator > 0 else 0
        launches_needed = min(launches_needed, max_rocket_launches)
        x = _allocate_launches_evenly(launches_needed=launches_needed, available=available, x_max=int(params.x_max_per_base))
        rocket_deliver = float(np.sum(x)) * float(params.Q_rock_mt)

        launches_i_t[:, t] = x.astype(float)
        rocket_mt_t[t] = float(rocket_deliver)
        elevator_mt_t[t] = float(elevator_deliver)
        W += float(elevator_deliver) + float(rocket_deliver)
        cum_mt_t[t] = float(W)

    shortfall = max(0.0, float(params.W_goal_mt) - float(W))
    repair_cost = _scenario_repair_cost(params, scenario)
    rocket_launches_total = float(np.sum(launches_i_t))
    rocket_cost = float(params.rocket_cost_per_launch_usd) * float(rocket_launches_total)
    total_cost = rocket_cost + float(repair_cost)
    rocket_env = float(params.rocket_env_tco2_per_launch) * float(rocket_launches_total)
    objective = (
        float(params.alpha_time) * (float(completion_t + 1) / float(max(T, 1)))
        + float(params.beta_cost) * (total_cost / 1e12)
        + float(params.gamma_env) * (rocket_env / 1e9)
        + float(params.lambda_shortage) * (shortfall / float(params.W_goal_mt))
    )

    return {
        "completion_t": int(min(T, completion_t + 1)),
        "shortfall": float(shortfall),
        "rocket_launches": float(rocket_launches_total),
        "elevator_mt": float(float(np.sum(elevator_mt_t))),
        "repair_cost": float(repair_cost),
        "total_cost": float(total_cost),
        "rocket_env": float(rocket_env),
        "objective": float(objective),
        "rocket_mt_t": rocket_mt_t,
        "elevator_mt_t": elevator_mt_t,
        "cum_mt_t": cum_mt_t,
        "launches_i_t": launches_i_t,
        "available_bases_t": available_bases_t,
    }


def summarize(results: List[Dict]) -> Dict:
    if not results:
        return {}
    keys = ["completion_t", "shortfall", "rocket_launches", "elevator_mt", "repair_cost", "total_cost", "rocket_env", "objective"]
    out: Dict[str, float] = {}
    for k in keys:
        arr = np.array([float(r[k]) for r in results], dtype=float)
        out[f"E[{k}]"] = float(np.mean(arr))
        out[f"Std[{k}]"] = float(np.std(arr))
    out["P(shortfall>0)"] = float(np.mean([r["shortfall"] > 1e-9 for r in results]))
    out["Max(shortfall)"] = float(np.max([r["shortfall"] for r in results]))
    return out


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save_matplotlib(fig, filename: str, out_dir: str) -> str:
    out_path = os.path.abspath(os.path.join(out_dir, filename))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt = _mpl()
    plt.close(fig)
    return out_path


def run_policy(
    params: Params,
    solver: str,
    time_limit_s: int,
    mip_gap: float,
    seed: int,
    n_saa: int,
    n_eval: int,
) -> Tuple[SAASolution, Dict, List[Dict], Dict]:
    scenarios = sample_scenarios(params, n=int(n_saa), seed=int(seed))
    sol = solve_saa_benders_like(
        params=params,
        scenarios=scenarios,
        solver=solver,
        time_limit_s=int(time_limit_s),
        mip_gap=float(mip_gap),
    )

    rules = extract_rules(params, sol)
    eval_scenarios = sample_scenarios(params, n=int(n_eval), seed=int(seed) + 10_000)
    heuristic_results = [simulate_heuristic(params, sc, rules) for sc in eval_scenarios]
    stats = summarize(heuristic_results)
    return sol, rules, heuristic_results, stats


def make_robustness_plots(
    params: Params,
    sol: SAASolution,
    eval_results: List[Dict],
    out_dir: str,
    prefix: str = "task2_new",
) -> List[str]:
    plt = _mpl()
    from matplotlib.patches import Ellipse

    completion = np.array([float(r["completion_t"]) for r in eval_results], dtype=float)
    total_cost_b = np.array([float(r["total_cost"]) / 1e9 for r in eval_results], dtype=float)
    objective = np.array([float(r["objective"]) for r in eval_results], dtype=float)
    shortfall_mt = np.array([float(r["shortfall"]) for r in eval_results], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax = axes[0, 0]
    ax.scatter(completion, total_cost_b, alpha=0.35, s=18, c="#1f77b4", edgecolors="w", linewidth=0.3)
    ax.scatter(
        [float(np.mean(completion))],
        [float(np.mean(total_cost_b))],
        c="green",
        s=120,
        marker="o",
        edgecolors="black",
        linewidth=1.5,
        label="Expected",
        zorder=5,
    )

    cov = np.cov(completion, total_cost_b)
    if np.all(np.isfinite(cov)) and np.linalg.det(cov) > 1e-12:
        vals, vecs = np.linalg.eig(cov)
        vals = np.sqrt(np.maximum(vals, 0.0))
        angle = float(np.rad2deg(np.arctan2(vecs[1, 0], vecs[0, 0])))
        ellipse = Ellipse(
            xy=(float(np.mean(completion)), float(np.mean(total_cost_b))),
            width=float(vals[0] * 2.0 * 2.0),
            height=float(vals[1] * 2.0 * 2.0),
            angle=angle,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            linestyle="--",
            alpha=0.85,
        )
        ax.add_patch(ellipse)

    ax.set_xlabel("Completion period", fontsize=11)
    ax.set_ylabel("Total cost (Billion USD)", fontsize=11)
    ax.set_title("Robustness scatter (completion vs cost)", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper left")

    ax = axes[0, 1]
    ax.hist(completion, bins=min(30, max(10, int(np.sqrt(completion.size)))), color="steelblue", edgecolor="black", alpha=0.75)
    ax.axvline(float(np.mean(completion)), color="green", linestyle="--", linewidth=2, label=f"Expected={np.mean(completion):.1f}")
    ax.set_xlabel("Completion period", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Completion distribution", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[1, 0]
    ax.hist(objective, bins=min(40, max(12, int(np.sqrt(objective.size)))), color="coral", edgecolor="black", alpha=0.75)
    ax.axvline(float(np.mean(objective)), color="green", linestyle="--", linewidth=2, label=f"Expected={np.mean(objective):.3f}")
    if np.isfinite(sol.objective):
        ax.axvline(float(sol.objective), color="red", linestyle=":", linewidth=2, label="SAA-opt (master)")
    ax.set_xlabel("Objective", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Objective distribution", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[1, 1]
    sorted_sf = np.sort(shortfall_mt)
    cdf = (np.arange(1, sorted_sf.size + 1) / float(max(1, sorted_sf.size))).astype(float)
    ax.plot(sorted_sf / 1e6, cdf, linewidth=2.5, color="#9467bd")
    ax.set_xlabel("Shortfall (million tons)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Shortfall risk (CDF)", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    p0 = float(np.mean(shortfall_mt <= 1e-9))
    ax.text(0.02, 0.08, f"P(shortfall=0)={p0:.1%}", transform=ax.transAxes, fontsize=10)

    fig.tight_layout()
    paths = [_save_matplotlib(fig, f"{prefix}_robustness.png", out_dir)]
    return paths


def make_sensitivity_plots(
    params_base: Params,
    solver: str,
    time_limit_s: int,
    mip_gap: float,
    seed: int,
    n_saa: int,
    n_eval: int,
    out_dir: str,
    prefix: str = "task2_new",
) -> List[str]:
    plt = _mpl()

    p_grid = np.array([0.05, 0.10, 0.20, 0.30, 0.40], dtype=float)
    theta_grid = np.array([0.50, 0.70, 0.85, 1.00], dtype=float)
    lambda_grid = np.array([5.0, 20.0, 50.0, 100.0, 200.0], dtype=float)

    p_out = []
    for v in p_grid:
        params = replace(params_base, p_fail_base=float(v))
        _, _, _, stats = run_policy(params, solver, time_limit_s, mip_gap, seed, n_saa, n_eval)
        p_out.append((float(v), float(stats["E[completion_t]"]), float(stats["E[objective]"]), float(stats["P(shortfall>0)"])))

    theta_out = []
    for v in theta_grid:
        params = replace(params_base, theta_swing=float(v))
        _, _, _, stats = run_policy(params, solver, time_limit_s, mip_gap, seed, n_saa, n_eval)
        theta_out.append((float(v), float(stats["E[completion_t]"]), float(stats["E[objective]"]), float(stats["P(shortfall>0)"])))

    lambda_out = []
    for v in lambda_grid:
        params = replace(params_base, lambda_shortage=float(v))
        _, _, _, stats = run_policy(params, solver, time_limit_s, mip_gap, seed, n_saa, n_eval)
        lambda_out.append((float(v), float(stats["E[completion_t]"]), float(stats["E[objective]"]), float(stats["P(shortfall>0)"])))

    p_arr = np.array(p_out, dtype=float)
    th_arr = np.array(theta_out, dtype=float)
    la_arr = np.array(lambda_out, dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    ax = axes[0, 0]
    ax.plot(p_arr[:, 0], p_arr[:, 1], marker="o", linewidth=2)
    ax.set_title("p_fail vs Expected completion", fontsize=12, pad=10)
    ax.set_xlabel("p_fail_base")
    ax.set_ylabel("E[completion_t]")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(p_arr[:, 0], p_arr[:, 2], marker="o", linewidth=2, color="tab:orange")
    ax.set_title("p_fail vs Expected objective", fontsize=12, pad=10)
    ax.set_xlabel("p_fail_base")
    ax.set_ylabel("E[objective]")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(th_arr[:, 0], th_arr[:, 1], marker="o", linewidth=2, color="tab:green")
    ax.set_title("theta_swing vs Expected completion", fontsize=12, pad=10)
    ax.set_xlabel("theta_swing")
    ax.set_ylabel("E[completion_t]")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(th_arr[:, 0], th_arr[:, 3] * 100.0, marker="o", linewidth=2, color="tab:red")
    ax.set_title("theta_swing vs Shortfall probability", fontsize=12, pad=10)
    ax.set_xlabel("theta_swing")
    ax.set_ylabel("P(shortfall>0) (%)")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 0]
    ax.plot(la_arr[:, 0], la_arr[:, 2], marker="o", linewidth=2, color="#9467bd")
    ax.set_title("lambda_shortage vs Expected objective", fontsize=12, pad=10)
    ax.set_xlabel("lambda_shortage")
    ax.set_ylabel("E[objective]")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[2, 1]
    ax.plot(la_arr[:, 0], la_arr[:, 3] * 100.0, marker="o", linewidth=2, color="#8c564b")
    ax.set_title("lambda_shortage vs Shortfall probability", fontsize=12, pad=10)
    ax.set_xlabel("lambda_shortage")
    ax.set_ylabel("P(shortfall>0) (%)")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    paths = [_save_matplotlib(fig, f"{prefix}_sensitivity.png", out_dir)]
    return paths


def make_trajectory_plots(
    params: Params,
    rules: Dict,
    eval_scenarios: List[Scenario],
    out_dir: str,
    prefix: str = "task2_new",
) -> List[str]:
    plt = _mpl()
    T = params.T
    setup = params.setup_periods()
    n = int(len(eval_scenarios))
    if n <= 0:
        return []

    traces = [simulate_heuristic_trace(params, sc, rules) for sc in eval_scenarios]
    cum = np.stack([tr["cum_mt_t"] for tr in traces], axis=0)
    rocket = np.stack([tr["rocket_mt_t"] for tr in traces], axis=0)
    elevator = np.stack([tr["elevator_mt_t"] for tr in traces], axis=0)

    p10 = np.percentile(cum, 10, axis=0)
    p50 = np.percentile(cum, 50, axis=0)
    p90 = np.percentile(cum, 90, axis=0)

    mean_rocket = np.mean(rocket, axis=0)
    mean_elevator = np.mean(elevator, axis=0)

    planned_cum = np.asarray(rules["planned_cum_delivery"], dtype=float)

    x = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.fill_between(x, p10 / 1e6, p90 / 1e6, color="#1f77b4", alpha=0.18, label="P10–P90 (sim)")
    ax.plot(x, p50 / 1e6, color="#1f77b4", linewidth=2.5, label="Median (sim)")
    ax.plot(x, planned_cum / 1e6, color="#ff7f0e", linewidth=2.5, linestyle="--", label="Planned (SAA rule)")
    ax.axhline(float(params.W_goal_mt) / 1e6, color="black", linewidth=1.5, linestyle=":")
    ax.axvline(setup, color="gray", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("Cumulative delivered (million tons)", fontsize=11)
    ax.set_title("Cumulative delivery trajectories (uncertainty band)", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="lower right")

    ax = axes[1]
    ax.stackplot(
        x,
        mean_rocket / 1e6,
        mean_elevator / 1e6,
        labels=["Rocket (mean)", "Elevator (mean)"],
        colors=["#d62728", "#2ca02c"],
        alpha=0.7,
    )
    ax.axvline(setup, color="gray", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("Delivered per period (million tons)", fontsize=11)
    ax.set_title("Mean per-period delivery mix", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    return [_save_matplotlib(fig, f"{prefix}_trajectories.png", out_dir)]


def make_dispatch_heatmap(
    params: Params,
    rules: Dict,
    eval_scenarios: List[Scenario],
    out_dir: str,
    prefix: str = "task2_new",
) -> List[str]:
    plt = _mpl()
    I, T = params.n_bases, params.T
    n = int(len(eval_scenarios))
    if n <= 0:
        return []

    traces = [simulate_heuristic_trace(params, sc, rules) for sc in eval_scenarios]
    launches = np.stack([tr["launches_i_t"] for tr in traces], axis=0)
    mean_launches = np.mean(launches, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    im = ax.imshow(mean_launches, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("Rocket base index", fontsize=11)
    ax.set_title("Mean rocket dispatch heatmap (launches per base per period)", fontsize=12, pad=10)
    ax.set_yticks(np.arange(I))
    ax.set_yticklabels([str(i + 1) for i in range(I)])
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Launches", fontsize=10)
    fig.tight_layout()
    return [_save_matplotlib(fig, f"{prefix}_dispatch_heatmap.png", out_dir)]


def make_tradeoff_plots(
    eval_results: List[Dict],
    out_dir: str,
    prefix: str = "task2_new",
) -> List[str]:
    plt = _mpl()
    if not eval_results:
        return []
    completion = np.array([float(r["completion_t"]) for r in eval_results], dtype=float)
    objective = np.array([float(r["objective"]) for r in eval_results], dtype=float)
    total_cost_b = np.array([float(r["total_cost"]) / 1e9 for r in eval_results], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sc = ax.scatter(completion, objective, c=total_cost_b, cmap="viridis", s=35, alpha=0.75, edgecolors="w", linewidth=0.3)
    best = int(np.nanargmin(objective)) if np.any(np.isfinite(objective)) else 0
    ax.scatter([completion[best]], [objective[best]], c="red", s=120, marker="*", edgecolors="black", linewidth=1.5, label="Best objective")
    ax.set_xlabel("Completion period", fontsize=11)
    ax.set_ylabel("Objective", fontsize=11)
    ax.set_title("Trade-off view (color = total cost)", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Total cost (Billion USD)", fontsize=10)
    fig.tight_layout()
    return [_save_matplotlib(fig, f"{prefix}_tradeoff.png", out_dir)]


def make_scurve_3d_projection_plot(
    params: Params,
    sol: SAASolution,
    rules: Dict,
    out_dir: str,
    prefix: str = "task2_new",
) -> List[str]:
    plt = _mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    T = params.T
    setup = params.setup_periods()
    x = np.arange(1, T + 1)

    cum_hybrid = np.asarray(rules["planned_cum_delivery"], dtype=float).copy()
    cum_hybrid = np.minimum(cum_hybrid, float(params.W_goal_mt))
    launches_hybrid = np.sum(sol.x_ok + sol.x_fail, axis=0).astype(float)
    cost_hybrid = np.cumsum(launches_hybrid * float(params.rocket_cost_per_launch_usd))
    if np.isfinite(sol.expected_repair_cost) and sol.expected_repair_cost > 0:
        cost_hybrid = cost_hybrid + float(sol.expected_repair_cost) * (x / float(T))

    per_period_launches_allrocket = float(params.W_goal_mt) / (float(params.Q_rock_mt) * float(T))
    launches_allrocket = np.full(T, float(per_period_launches_allrocket), dtype=float)
    cum_allrocket = np.cumsum(launches_allrocket * float(params.Q_rock_mt))
    cum_allrocket = np.minimum(cum_allrocket, float(params.W_goal_mt))
    cost_allrocket = np.cumsum(launches_allrocket * float(params.rocket_cost_per_launch_usd))

    elevator_cap = float(params.C_elevator_mt_per_year) * float(params.n_ports) * float(params.expected_theta()) * float(params.expected_elevator_up())
    prebuild_target = min(float(params.W_goal_mt) * 0.35, float(params.W_goal_mt))
    prebuild_launches = 0.0
    if setup > 0:
        prebuild_launches = prebuild_target / (float(params.Q_rock_mt) * float(setup))
    launches_elevator_focus = np.zeros(T, dtype=float)
    if setup > 0:
        launches_elevator_focus[:setup] = float(prebuild_launches)
    cum_elevator_focus = np.zeros(T, dtype=float)
    cost_elevator_focus = np.zeros(T, dtype=float)
    delivered = 0.0
    for t in range(T):
        delivered += launches_elevator_focus[t] * float(params.Q_rock_mt)
        if t >= setup:
            delivered += float(elevator_cap)
        cum_elevator_focus[t] = min(delivered, float(params.W_goal_mt))
        cost_elevator_focus[t] = (cost_elevator_focus[t - 1] if t > 0 else 0.0) + launches_elevator_focus[t] * float(params.rocket_cost_per_launch_usd)

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    for y_data, z_data, label, color in [
        (cum_hybrid, cost_hybrid, "Hybrid (SAA-rule)", "#1f77b4"),
        (cum_allrocket, cost_allrocket, "All-Rocket (linear)", "#ff7f0e"),
        (cum_elevator_focus, cost_elevator_focus, "Rocket+Elevator focus", "#2ca02c"),
    ]:
        ax1.plot(
            x,
            y_data / 1e6,
            z_data / 1e12,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=0.85,
        )
        ax2.plot(x, y_data / 1e6, label=label, color=color, linewidth=2.2, alpha=0.85)

    ax1.set_xlabel("Period", fontsize=12, labelpad=15)
    ax1.set_ylabel("Cumulative payload (million tons)", fontsize=12, labelpad=15)
    ax1.set_zlabel("Cumulative rocket cost (trillion USD)", fontsize=12, labelpad=15)
    ax1.set_title("Cumulative Payload–Cost Trajectories (3D)", fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=10)

    ax2.set_xlabel("Period", fontsize=12)
    ax2.set_ylabel("Cumulative payload (million tons)", fontsize=12)
    ax2.set_title("Cumulative S-Curves (2D projection)", fontsize=14, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.axvline(x=float(setup), color="gray", linestyle="--", alpha=0.6, linewidth=1.5)
    ax2.text(float(setup) + 1.5, 5, "Elevator online", rotation=90, fontsize=10, color="gray")

    fig.tight_layout(w_pad=5.0)
    return [_save_matplotlib(fig, f"{prefix}_scurve_3d.png", out_dir)]


def make_pareto_frontier_plot(
    params_base: Params,
    solver: str,
    time_limit_s: int,
    mip_gap: float,
    seed: int,
    n_saa: int,
    n_eval: int,
    out_dir: str,
    prefix: str = "task2_new",
) -> List[str]:
    plt = _mpl()

    rng = np.random.default_rng(int(seed) + 99)
    n_points = 45
    points = []
    meta = []
    for i in range(n_points):
        w = rng.random(3)
        w = w / float(np.sum(w))
        alpha = float(w[0] * 2.0)
        beta = float(w[1] * 2.0)
        gamma = float(w[2] * 2.0)
        params = replace(params_base, alpha_time=alpha, beta_cost=beta, gamma_env=gamma)
        _, _, _, stats = run_policy(params, solver, time_limit_s, mip_gap, seed + i * 7, n_saa, n_eval)
        time_v = float(stats["E[completion_t]"])
        cost_v = float(stats["E[total_cost]"]) / 1e12
        env_v = float(stats["E[rocket_env]"]) / 1e9
        points.append([time_v, cost_v, env_v])
        meta.append({"alpha": alpha, "beta": beta, "gamma": gamma, "objective": float(stats["E[objective]"])})

    pts = np.asarray(points, dtype=float)

    def is_pareto_efficient(arr: np.ndarray) -> np.ndarray:
        efficient = np.ones(arr.shape[0], dtype=bool)
        for i, c in enumerate(arr):
            if efficient[i]:
                efficient[efficient] = np.any(arr[efficient] < c, axis=1) | ~np.all(arr[efficient] <= c, axis=1)
                efficient[i] = True
        return efficient

    pareto_mask = is_pareto_efficient(pts)
    pareto_pts = pts[pareto_mask]

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="gray", alpha=0.25, s=20, label="Feasible")
    ax1.scatter(pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2], c="red", s=45, label="Pareto frontier", edgecolors="black")

    idx_time = int(np.argmin(pts[:, 0]))
    idx_cost = int(np.argmin(pts[:, 1]))
    idx_env = int(np.argmin(pts[:, 2]))
    idx_obj = int(np.argmin([m["objective"] for m in meta]))
    special = [
        (idx_time, "Min-Time", "blue", "*"),
        (idx_cost, "Min-Cost", "green", "^"),
        (idx_env, "Min-Env", "orange", "s"),
        (idx_obj, "Min-Objective", "purple", "D"),
    ]
    for idx, label, color, marker in special:
        ax1.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], c=color, s=120, marker=marker, label=label, edgecolors="black", linewidth=1.2)

    ax1.set_xlabel("Build time (periods)", fontsize=10, labelpad=10)
    ax1.set_ylabel("Total cost (trillion USD)", fontsize=10, labelpad=10)
    ax1.set_zlabel("Rocket env (0.1B tCO₂e)", fontsize=10, labelpad=10)
    ax1.set_title("3D Pareto surface (sampled weights)", fontsize=12, pad=12)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(132)
    ax2.scatter(pts[:, 0], pts[:, 1], c="gray", alpha=0.25, s=20)
    ax2.scatter(pareto_pts[:, 0], pareto_pts[:, 1], c="red", s=35)
    for idx, label, color, marker in special:
        ax2.scatter([pts[idx, 0]], [pts[idx, 1]], c=color, s=90, marker=marker, edgecolors="black", linewidth=1.1)
    ax2.set_xlabel("Build time (periods)", fontsize=10)
    ax2.set_ylabel("Total cost (trillion USD)", fontsize=10)
    ax2.set_title("Time–cost trade-off", fontsize=12, pad=10)
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(133)
    sc = ax3.scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], cmap="viridis", s=28, alpha=0.75)
    ax3.scatter(pareto_pts[:, 1], pareto_pts[:, 2], c="red", s=18, alpha=0.6)
    for idx, label, color, marker in special:
        ax3.scatter([pts[idx, 1]], [pts[idx, 2]], c=color, s=90, marker=marker, edgecolors="black", linewidth=1.1)
    ax3.set_xlabel("Total cost (trillion USD)", fontsize=10)
    ax3.set_ylabel("Rocket env (0.1B tCO₂e)", fontsize=10)
    ax3.set_title("Cost–environment (color = time)", fontsize=12, pad=10)
    fig.colorbar(sc, ax=ax3, label="Build time (periods)")
    ax3.grid(True, alpha=0.25)

    fig.tight_layout(w_pad=4.0)
    return [_save_matplotlib(fig, f"{prefix}_pareto.png", out_dir)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["auto", "gurobi", "pulp"], default="auto")
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--mip-gap", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-saa", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--T", type=int, default=160)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--plot-dir", type=str, default=".")
    ap.add_argument("--make-sensitivity", action="store_true")
    ap.add_argument("--make-robustness", action="store_true")
    ap.add_argument("--n-saa-sens", type=int, default=30)
    ap.add_argument("--n-eval-sens", type=int, default=30)
    ap.add_argument("--n-eval-traj", type=int, default=60)
    ap.add_argument("--n-pareto", type=int, default=20)
    ap.add_argument("--n-pareto-eval", type=int, default=20)
    args = ap.parse_args()

    params = Params(T=int(args.T), eta_supply=float(args.eta))

    solver = str(args.solver)
    if solver == "auto":
        solver = "gurobi" if GUROBI_AVAILABLE else "pulp"

    print("=" * 70)
    print("Task2-New: SAA→(Benders式分解/解析子问题)→提炼规则→启发式调度")
    print("=" * 70)
    print(f"solver={solver}  T={params.T}  N_saa={int(args.n_saa)}  N_eval={int(args.n_eval)}  seed={int(args.seed)}")
    print(f"Gurobi可用={GUROBI_AVAILABLE}")

    sol, rules, heuristic_results, stats = run_policy(
        params=params,
        solver=solver,
        time_limit_s=int(args.time_limit),
        mip_gap=float(args.mip_gap),
        seed=int(args.seed),
        n_saa=int(args.n_saa),
        n_eval=int(args.n_eval),
    )

    print("-" * 70)
    print("[SAA最优解(用于提炼规则)]")
    print(f"objective={sol.objective:.6f}  E[shortfall]={sol.expected_shortfall/1e6:.3f}M MT  E[repair_cost]=${sol.expected_repair_cost/1e9:.3f}B")
    print(f"w(build ports)={rules['w'].tolist()}")
    print(f"build_mean_launches={rules['build_mean_launches']:.2f}  post_mean_launches={rules['post_mean_launches']:.2f}")

    print("-" * 70)
    print("[启发式调度评估(独立场景集)]")
    for k in [
        "E[completion_t]",
        "Std[completion_t]",
        "E[shortfall]",
        "P(shortfall>0)",
        "Max(shortfall)",
        "E[rocket_launches]",
        "E[elevator_mt]",
        "E[total_cost]",
        "E[objective]",
    ]:
        if k in stats:
            v = stats[k]
            if k == "P(shortfall>0)":
                print(f"{k} = {v:.3%}")
            elif "shortfall" in k:
                print(f"{k} = {v/1e6:.3f}M MT")
            elif "total_cost" in k:
                print(f"{k} = ${v/1e9:.3f}B")
            else:
                print(f"{k} = {v:.6f}")

    if bool(args.plots):
        out_dir = str(args.plot_dir)
        os.makedirs(out_dir, exist_ok=True)
        want_sens = bool(args.make_sensitivity) or (not bool(args.make_sensitivity) and not bool(args.make_robustness))
        want_rob = bool(args.make_robustness) or (not bool(args.make_sensitivity) and not bool(args.make_robustness))

        out_paths: List[str] = []
        if want_sens:
            out_paths.extend(
                make_sensitivity_plots(
                    params_base=params,
                    solver=solver,
                    time_limit_s=int(args.time_limit),
                    mip_gap=float(args.mip_gap),
                    seed=int(args.seed),
                    n_saa=int(args.n_saa_sens),
                    n_eval=int(args.n_eval_sens),
                    out_dir=out_dir,
                )
            )

        if want_rob:
            out_paths.extend(make_robustness_plots(params=params, sol=sol, eval_results=heuristic_results, out_dir=out_dir))

        traj_scenarios = sample_scenarios(params, n=int(args.n_eval_traj), seed=int(args.seed) + 20_000)
        out_paths.extend(make_trajectory_plots(params=params, rules=rules, eval_scenarios=traj_scenarios, out_dir=out_dir))
        out_paths.extend(make_dispatch_heatmap(params=params, rules=rules, eval_scenarios=traj_scenarios, out_dir=out_dir))
        out_paths.extend(make_tradeoff_plots(eval_results=heuristic_results, out_dir=out_dir))
        out_paths.extend(make_scurve_3d_projection_plot(params=params, sol=sol, rules=rules, out_dir=out_dir))
        out_paths.extend(
            make_pareto_frontier_plot(
                params_base=params,
                solver=solver,
                time_limit_s=int(args.time_limit),
                mip_gap=float(args.mip_gap),
                seed=int(args.seed),
                n_saa=int(args.n_pareto),
                n_eval=int(args.n_pareto_eval),
                out_dir=out_dir,
            )
        )

        print("-" * 70)
        print("[图表输出]")
        for p in out_paths:
            print(p)


if __name__ == "__main__":
    main()
