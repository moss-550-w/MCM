"""
=============================================================================
月球殖民地水资源动态补给优化模型 (Lunar Colony Water Resource Optimization Model)
=============================================================================
问题背景：10万人月球殖民地，520周(10年)规划周期
模型类型：混合整数线性规划 (MILP) + 动态库存平衡
求解器：PuLP (开源) 或 Gurobi/CPLEX (商业)
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pulp import *

# ==================== 1. 参数配置类 ====================

@dataclass
class ModelParameters:
    """模型核心参数配置"""
    
    # === 时间参数 ===
    T: int = 520                      # 总规划周期 (周)
    dt: float = 7.0                   # 每周天数
    
    # === 人口与需求参数 ===
    population: int = 100_000         # 殖民地人口 (10万人)
    water_per_person_day: float = 200.0   # 舒适标准: L/人/天
    survival_water_per_person_day: float = 2.7  # 生存底线: L/人/天
    
    # === 物流参数 ===
    rocket_capacity: float = 100.0    # 单枚火箭载荷 (ton)
    launch_cost_per_kg: float = 500.0 # $/kg 运输成本
    # 单次发射成本 = 500 * 100 * 1000 = $50,000,000
    
    # === 水资源循环参数 ===
    initial_eta: float = 0.85         # 初始回收率 (ISS水平 85%)
    max_eta: float = 0.99             # 理论上限回收率
    alpha: float = 0.001              # 设备转化系数: %/ton (每吨设备提升0.1%回收率)
    
    # === 库存参数 ===
    initial_stock: float = 500.0      # 初始库存 (ton)
    safety_days: float = 3.0          # 安全库存天数
    
    # === 惩罚参数 (软约束) ===
    penalty_comfort_cut: float = 1e9  # 削减舒适需求的惩罚系数
    
    def __post_init__(self):
        # 计算派生参数
        self.weekly_demand_comfort = (
            self.population * self.water_per_person_day * self.dt / 1000
        )  # ton/周 (舒适标准)
        
        self.weekly_demand_survival = (
            self.population * self.survival_water_per_person_day * self.dt / 1000
        )  # ton/周 (生存标准)
        
        self.safety_stock = (
            self.population * self.survival_water_per_person_day * self.safety_days / 1000
        )  # 安全库存底线 (ton)
        
        self.launch_cost_total = (
            self.launch_cost_per_kg * self.rocket_capacity * 1000
        )  # 单次发射总成本 ($)


# ==================== 2. 核心优化模型类 ====================

class WaterResourceOptimizer:
    """
    月球水资源动态补给优化器
    目标：最小化总物流成本，同时满足水资源动态平衡
    """
    
    def __init__(self, params: ModelParameters):
        self.params = params
        self.model = None
        self.variables = {}
        self.solution = None
        
    def build_model(self, use_soft_constraint: bool = True) -> LpProblem:
        """
        构建MILP模型
        
        Args:
            use_soft_constraint: 是否使用软约束（允许在必要时削减舒适需求）
        """
        T = self.params.T
        p = self.params
        
        # 创建问题实例
        prob = LpProblem("Lunar_Water_Optimization", LpMinimize)
        
        # ========== 决策变量 ==========
        
        # x[t]: 第t周发射的火箭数量 (整数, >=0)
        x = LpVariable.dicts("Rocket", range(T), lowBound=0, cat='Integer')
        
        # m_water[t]: 第t周运输的水质量 (连续, >=0)
        m_water = LpVariable.dicts("Water", range(T), lowBound=0, cat='Continuous')
        
        # m_equip[t]: 第t周运输的设备质量 (连续, >=0)
        m_equip = LpVariable.dicts("Equipment", range(T), lowBound=0, cat='Continuous')
        
        # S[t]: 第t周末的库存水平 (连续)
        S = LpVariable.dicts("Stock", range(-1, T), lowBound=0, cat='Continuous')
        
        # eta[t]: 第t周的回收率 (连续)
        eta_low = p.initial_eta if use_soft_constraint else 0
        eta = LpVariable.dicts("Efficiency", range(T), lowBound=eta_low, upBound=p.max_eta, cat='Continuous')
        
        # D[t]: 第t周的实际用水需求 (连续)
        # 如果使用软约束，允许在生存标准和舒适标准之间调整
        if use_soft_constraint:
            D = LpVariable.dicts("Demand", range(T), 
                               lowBound=p.weekly_demand_survival,
                               upBound=p.weekly_demand_comfort,
                               cat='Continuous')
            # 削减量（用于惩罚）
            comfort_cut = LpVariable.dicts("ComfortCut", range(T), lowBound=0, cat='Continuous')
        else:
            D = {t: p.weekly_demand_comfort for t in range(T)}
        
        # R[t]: 第t周的循环再生量
        R = LpVariable.dicts("Recycle", range(T), lowBound=0, cat='Continuous')
        
        eta_grid = None
        eta_level = None
        demand_at_eta = None
        if use_soft_constraint:
            eta_step = 0.005
            eta_grid = []
            k = 0
            while p.initial_eta + k * eta_step < p.max_eta - 1e-9:
                eta_grid.append(round(p.initial_eta + k * eta_step, 6))
                k += 1
            eta_grid.append(round(p.max_eta, 6))

            K = len(eta_grid)
            eta_level = LpVariable.dicts("EtaLevel", (range(T), range(K)), cat='Binary')
            demand_at_eta = LpVariable.dicts("DemandAtEta", (range(T), range(K)), lowBound=0, cat='Continuous')

            for t in range(T):
                prob += lpSum([eta_level[t][kk] for kk in range(K)]) == 1, f"EtaLevel_OneHot_{t}"
                prob += eta[t] == lpSum([eta_grid[kk] * eta_level[t][kk] for kk in range(K)]), f"EtaLevel_Value_{t}"

                prob += lpSum([demand_at_eta[t][kk] for kk in range(K)]) == D[t], f"DemandAtEta_Decomp_{t}"
                for kk in range(K):
                    prob += (
                        demand_at_eta[t][kk] <= p.weekly_demand_comfort * eta_level[t][kk]
                    ), f"DemandAtEta_UB_{t}_{kk}"

        # 存储变量引用
        self.variables = {
            'x': x, 'm_water': m_water, 'm_equip': m_equip,
            'S': S, 'eta': eta, 'D': D, 'R': R
        }
        if use_soft_constraint:
            self.variables['comfort_cut'] = comfort_cut
            self.variables['eta_level'] = eta_level
            self.variables['demand_at_eta'] = demand_at_eta
            
        # ========== 目标函数 ==========
        # 最小化总成本 = 发射成本 + 舒适需求削减惩罚
        total_launch_cost = lpSum([x[t] * p.launch_cost_total for t in range(T)])
        
        if use_soft_constraint:
            total_penalty = lpSum([comfort_cut[t] * p.penalty_comfort_cut 
                                  for t in range(T)])
            prob += total_launch_cost + total_penalty, "Total_Cost"
        else:
            prob += total_launch_cost, "Total_Launch_Cost"
        
        # ========== 约束条件 ==========
        
        # 初始条件
        prob += S[-1] == p.initial_stock, "Initial_Stock"
        
        for t in range(T):
            # Eq(4): 回收率动态演化
            if t == 0:
                prob += eta[t] == p.initial_eta + p.alpha * m_equip[t], f"Eta_Evolution_{t}"
            else:
                prob += eta[t] == eta[t-1] + p.alpha * m_equip[t], f"Eta_Evolution_{t}"
            
            # Eq(3): 循环再生量 = 上周消耗 * 上周回收率
            if t == 0:
                prob += R[t] == 0, f"Recycle_Init_{t}"  # 第0周无循环
            else:
                if use_soft_constraint:
                    prob += R[t] == lpSum(
                        [eta_grid[kk] * demand_at_eta[t-1][kk] for kk in range(len(eta_grid))]
                    ), f"Recycle_Calc_{t}"
                else:
                    prob += R[t] == D[t-1] * eta[t-1], f"Recycle_Calc_{t}"
            
            # Eq(2): 库存平衡方程
            # S[t] = S[t-1] + 地球补给(m_water) + 循环再生(R) - 总消耗(D)
            prob += S[t] == S[t-1] + m_water[t] + R[t] - D[t], f"Stock_Balance_{t}"
            
            # Eq(6): 运力约束
            prob += m_water[t] + m_equip[t] <= x[t] * p.rocket_capacity, f"Capacity_{t}"
            
            # Eq(8): 安全库存约束
            prob += S[t] >= p.safety_stock, f"Safety_Stock_{t}"
            
            # 软约束定义
            if use_soft_constraint:
                prob += D[t] + comfort_cut[t] == p.weekly_demand_comfort, f"Demand_Def_{t}"
        
        self.model = prob
        return prob
    
    def solve(self, solver: Optional = None, msg: bool = True) -> Dict:
        """
        求解模型
        
        Returns:
            包含求解结果的字典
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # 求解
        if solver is None:
            self.model.solve(PULP_CBC_CMD(msg=msg, timeLimit=300))
        else:
            self.model.solve(solver)
        
        # 提取结果
        status = LpStatus[self.model.status]
        
        if status not in ['Optimal', 'Not Solved']:
            print(f"Warning: Solver status is {status}")
            
        # 收集解
        solution = {
            'status': status,
            'objective_value': value(self.model.objective),
            'total_cost': value(self.model.objective),
            'rockets': [value(self.variables['x'][t]) for t in range(self.params.T)],
            'water_shipped': [value(self.variables['m_water'][t]) for t in range(self.params.T)],
            'equipment_shipped': [value(self.variables['m_equip'][t]) for t in range(self.params.T)],
            'stock_levels': [value(self.variables['S'][t]) for t in range(-1, self.params.T)],
            'efficiency': [value(self.variables['eta'][t]) for t in range(self.params.T)],
            'recycle_amount': [value(self.variables['R'][t]) for t in range(self.params.T)],
        }
        
        if 'comfort_cut' in self.variables:
            solution['demand'] = [value(self.variables['D'][t]) for t in range(self.params.T)]
            solution['comfort_cuts'] = [value(self.variables['comfort_cut'][t]) for t in range(self.params.T)]
        else:
            solution['demand'] = [self.params.weekly_demand_comfort] * self.params.T
            
        self.solution = solution
        return solution
    
    def analyze_solution(self) -> pd.DataFrame:
        """生成详细的解决方案分析报告"""
        if self.solution is None:
            raise ValueError("No solution available. Call solve() first.")
        
        p = self.params
        sol = self.solution
        
        # 创建时间序列DataFrame
        df = pd.DataFrame({
            'Week': range(p.T),
            'Rockets': sol['rockets'],
            'Water_Shipped_ton': sol['water_shipped'],
            'Equipment_Shipped_ton': sol['equipment_shipped'],
            'Total_Shipped_ton': [w + e for w, e in zip(sol['water_shipped'], sol['equipment_shipped'])],
            'Stock_Level_ton': sol['stock_levels'][1:],  # 去掉初始值
            'Recycling_Rate': sol['efficiency'],
            'Recycled_Water_ton': sol['recycle_amount'],
            'Water_Demand_ton': sol['demand'],
        })
        
        # 计算累计指标
        df['Cumulative_Rockets'] = df['Rockets'].cumsum()
        df['Cumulative_Cost_M'] = df['Cumulative_Rockets'] * p.launch_cost_total / 1e6
        df['Cumulative_Equipment'] = df['Equipment_Shipped_ton'].cumsum()
        
        # 计算自给率
        df['Self_Sufficiency'] = df['Recycled_Water_ton'] / df['Water_Demand_ton']
        df['Self_Sufficiency'] = df['Self_Sufficiency'].replace([np.inf, -np.inf], 0)
        
        return df
    
    def print_summary(self):
        """打印求解结果摘要"""
        if self.solution is None:
            return
            
        p = self.params
        sol = self.solution
        
        print("=" * 60)
        print("月球水资源优化模型 - 求解结果摘要")
        print("=" * 60)
        print(f"求解状态: {sol['status']}")
        print(f"总成本: ${sol['total_cost']:,.2f}")
        print(f"总发射次数: {sum(sol['rockets'])} 枚")
        print(f"平均每2周发射: {sum(sol['rockets']) / (p.T/2):.2f} 枚")
        print(f"总运水质量: {sum(sol['water_shipped']):,.2f} ton")
        print(f"总运设备质量: {sum(sol['equipment_shipped']):,.2f} ton")
        print(f"最终回收率: {sol['efficiency'][-1]*100:.2f}%")
        print(f"最终库存水平: {sol['stock_levels'][-1]:,.2f} ton")
        print(f"安全库存底线: {p.safety_stock:.2f} ton")
        
        # 检查是否有舒适需求削减
        if 'comfort_cuts' in sol and sum(sol['comfort_cuts']) > 0:
            cut_weeks = sum(1 for c in sol['comfort_cuts'] if c > 0.01)
            print(f"\n警告: {cut_weeks} 周出现舒适需求削减")
        
        print("=" * 60)


# ==================== 3. 可视化模块 ====================

class ResultVisualizer:
    """结果可视化工具"""
    
    def __init__(self, optimizer: WaterResourceOptimizer):
        self.opt = optimizer
        self.params = optimizer.params
        self.df = optimizer.analyze_solution()
        
    def plot_comprehensive_dashboard(self, save_path: Optional[str] = None):
        """绘制综合仪表板"""
        df = self.df
        p = self.params
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Lunar Colony Water Resource Optimization Dashboard', fontsize=16, fontweight='bold')
        
        # 1. 库存水平 vs 安全线
        ax = axes[0, 0]
        ax.fill_between(df['Week'], df['Stock_Level_ton'], alpha=0.3, label='Water Stock')
        ax.axhline(y=p.safety_stock, color='r', linestyle='--', label=f'Safety Stock ({p.safety_stock:.1f}t)')
        ax.set_xlabel('Week')
        ax.set_ylabel('Stock (ton)')
        ax.set_title('Water Inventory Level Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 回收率演化
        ax = axes[0, 1]
        ax.plot(df['Week'], df['Recycling_Rate'] * 100, 'g-', linewidth=2, label='Recycling Rate')
        ax.axhline(y=p.max_eta * 100, color='r', linestyle='--', label=f'Theoretical Max ({p.max_eta*100}%)')
        ax.axhline(y=p.initial_eta * 100, color='orange', linestyle=':', label=f'Initial ({p.initial_eta*100}%)')
        ax.set_xlabel('Week')
        ax.set_ylabel('Recycling Rate (%)')
        ax.set_title('Water Recycling Efficiency Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 发射计划 (瀑布图)
        ax = axes[1, 0]
        launch_weeks = df[df['Rockets'] > 0]['Week']
        launch_counts = df[df['Rockets'] > 0]['Rockets']
        ax.bar(launch_weeks, launch_counts, width=2, color='steelblue', alpha=0.7)
        ax.set_xlabel('Week')
        ax.set_ylabel('Rockets Launched')
        ax.set_title('Rocket Launch Schedule')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 累计成本曲线
        ax = axes[1, 1]
        ax.plot(df['Week'], df['Cumulative_Cost_M'], 'purple', linewidth=2)
        ax.fill_between(df['Week'], df['Cumulative_Cost_M'], alpha=0.2, color='purple')
        ax.set_xlabel('Week')
        ax.set_ylabel('Cumulative Cost (Million $)')
        ax.set_title(f'Total Logistics Cost: ${df["Cumulative_Cost_M"].iloc[-1]:.1f}M')
        ax.grid(True, alpha=0.3)
        
        # 5. 供需平衡 (水循环)
        ax = axes[2, 0]
        width = 0.35
        weeks_sample = range(0, p.T, 52)  # 每年采样
        x_pos = np.arange(len(weeks_sample))
        
        shipped = [df.iloc[w]['Water_Shipped_ton'] if w < len(df) else 0 for w in weeks_sample]
        recycled = [df.iloc[w]['Recycled_Water_ton'] if w < len(df) else 0 for w in weeks_sample]
        demand = [df.iloc[w]['Water_Demand_ton'] if w < len(df) else 0 for w in weeks_sample]
        
        ax.bar(x_pos - width/2, shipped, width, label='Earth Supply', color='skyblue')
        ax.bar(x_pos + width/2, recycled, width, label='Recycled', color='lightgreen')
        ax.plot(x_pos, demand, 'ro-', label='Demand', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Water Mass (ton/week)')
        ax.set_title('Supply vs Demand Balance (Yearly Samples)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Y{i+1}' for i in range(len(weeks_sample))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. 设备累计投入 vs 自给率
        ax = axes[2, 1]
        ax2 = ax.twinx()
        
        line1 = ax.plot(df['Week'], df['Cumulative_Equipment'], 'b-', linewidth=2, label='Cumulative Equipment')
        line2 = ax2.plot(df['Week'], df['Self_Sufficiency'] * 100, 'r-', linewidth=2, label='Self-Sufficiency Rate')
        
        ax.set_xlabel('Week')
        ax.set_ylabel('Cumulative Equipment (ton)', color='b')
        ax2.set_ylabel('Self-Sufficiency (%)', color='r')
        ax.set_title('Equipment Investment vs Self-Sufficiency')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_phase_analysis(self):
        """分阶段策略分析"""
        df = self.df
        
        # 定义阶段
        phases = {
            'Initial (1-52)': (0, 52),
            'Growth (53-156)': (52, 156),
            'Stabilization (157-312)': (156, 312),
            'Mature (313-520)': (312, 520)
        }
        
        phase_data = []
        for name, (start, end) in phases.items():
            segment = df.iloc[start:end]
            phase_data.append({
                'Phase': name,
                'Avg_Rockets_per_Week': segment['Rockets'].mean(),
                'Total_Water_Shipped': segment['Water_Shipped_ton'].sum(),
                'Total_Equipment_Shipped': segment['Equipment_Shipped_ton'].sum(),
                'Final_Efficiency': segment['Recycling_Rate'].iloc[-1] * 100,
                'Avg_Self_Sufficiency': segment['Self_Sufficiency'].mean() * 100
            })
        
        phase_df = pd.DataFrame(phase_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase-by-Phase Strategy Analysis', fontsize=14, fontweight='bold')
        
        # 发射频率
        axes[0, 0].bar(phase_df['Phase'], phase_df['Avg_Rockets_per_Week'], color='steelblue')
        axes[0, 0].set_ylabel('Avg Rockets/Week')
        axes[0, 0].set_title('Launch Frequency by Phase')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 物资构成
        x = np.arange(len(phase_df))
        width = 0.35
        axes[0, 1].bar(x - width/2, phase_df['Total_Water_Shipped'], width, label='Water', color='skyblue')
        axes[0, 1].bar(x + width/2, phase_df['Total_Equipment_Shipped'], width, label='Equipment', color='orange')
        axes[0, 1].set_ylabel('Total Mass (ton)')
        axes[0, 1].set_title('Cargo Composition by Phase')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(phase_df['Phase'], rotation=45)
        axes[0, 1].legend()
        
        # 回收率提升
        axes[1, 0].plot(phase_df['Phase'], phase_df['Final_Efficiency'], 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_ylabel('Recycling Efficiency (%)')
        axes[1, 0].set_title('Efficiency Progression')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 自给率
        axes[1, 1].bar(phase_df['Phase'], phase_df['Avg_Self_Sufficiency'], color='green', alpha=0.6)
        axes[1, 1].set_ylabel('Self-Sufficiency (%)')
        axes[1, 1].set_title('Water Self-Sufficiency Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=100, color='r', linestyle='--', label='Full Self-Sufficiency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return phase_df


# ==================== 4. 敏感性分析模块 ====================

class SensitivityAnalyzer:
    """敏感性分析工具"""
    
    def __init__(self, base_params: ModelParameters):
        self.base_params = base_params
        
    def analyze_alpha_sensitivity(self, alpha_range: np.ndarray) -> pd.DataFrame:
        """
        分析设备转化系数 alpha 的敏感性
        alpha: 每吨设备提升的回收率
        """
        results = []
        
        for alpha in alpha_range:
            params = ModelParameters()
            params.alpha = alpha
            
            try:
                opt = WaterResourceOptimizer(params)
                opt.build_model(use_soft_constraint=True)
                sol = opt.solve(msg=False)
                
                results.append({
                    'alpha': alpha,
                    'total_cost': sol['total_cost'],
                    'total_rockets': sum(sol['rockets']),
                    'final_efficiency': sol['efficiency'][-1],
                    'total_equipment': sum(sol['equipment_shipped']),
                    'total_water': sum(sol['water_shipped']),
                    'status': sol['status']
                })
            except Exception as e:
                results.append({
                    'alpha': alpha,
                    'error': str(e)
                })
                
        return pd.DataFrame(results)
    
    def analyze_capacity_sensitivity(self, capacity_range: np.ndarray) -> pd.DataFrame:
        """分析火箭载荷能力的敏感性"""
        results = []
        
        for cap in capacity_range:
            params = ModelParameters()
            params.rocket_capacity = cap
            
            try:
                opt = WaterResourceOptimizer(params)
                opt.build_model(use_soft_constraint=True)
                sol = opt.solve(msg=False)
                
                results.append({
                    'capacity': cap,
                    'total_cost': sol['total_cost'],
                    'total_rockets': sum(sol['rockets']),
                    'final_efficiency': sol['efficiency'][-1],
                })
            except Exception as e:
                results.append({
                    'capacity': cap,
                    'error': str(e)
                })
                
        return pd.DataFrame(results)


# ==================== 5. 主执行流程 ====================

def main():
    """主执行函数 - 完整的美赛Q3求解流程"""
    
    print("=" * 70)
    print("月球殖民地水资源动态补给优化模型 (MCM/ICM Q3 Solution)")
    print("=" * 70)
    
    # Step 1: 初始化参数
    print("\n[Step 1] 初始化模型参数...")
    params = ModelParameters()
    
    print(f"规划周期: {params.T} 周 ({params.T/52:.1f} 年)")
    print(f"殖民地人口: {params.population:,} 人")
    print(f"舒适用水标准: {params.water_per_person_day} L/人/天")
    print(f"生存用水底线: {params.survival_water_per_person_day} L/人/天")
    print(f"单枚火箭载荷: {params.rocket_capacity} ton")
    print(f"单次发射成本: ${params.launch_cost_total:,.0f}")
    print(f"设备转化系数 α: {params.alpha} (每吨设备提升{params.alpha*100}%回收率)")
    
    # Step 2: 构建并求解模型
    print("\n[Step 2] 构建MILP模型...")
    optimizer = WaterResourceOptimizer(params)
    optimizer.build_model(use_soft_constraint=True)
    
    print("[Step 3] 求解优化问题 (使用CBC求解器)...")
    solution = optimizer.solve(msg=True)
    
    # Step 3: 输出结果
    optimizer.print_summary()
    
    # Step 4: 详细分析
    print("\n[Step 4] 生成详细分析报告...")
    df = optimizer.analyze_solution()
    
    # 显示关键周数据
    print("\n关键时间节点数据:")
    key_weeks = [0, 52, 104, 156, 260, 520-1]
    print(df.iloc[key_weeks][['Week', 'Rockets', 'Stock_Level_ton', 
                              'Recycling_Rate', 'Self_Sufficiency']].to_string())
    
    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    visualizer = ResultVisualizer(optimizer)
    visualizer.plot_comprehensive_dashboard(save_path="lunar_water_optimization.png")
    phase_df = visualizer.plot_phase_analysis()
    print("\n阶段分析摘要:")
    print(phase_df.to_string())
    
    # Step 6: 敏感性分析 (可选，计算量大)
    print("\n[Step 6] 执行敏感性分析...")
    analyzer = SensitivityAnalyzer(params)
    
    print("分析设备转化系数 α 的影响...")
    alpha_range = np.linspace(0.0005, 0.002, 5)
    alpha_results = analyzer.analyze_alpha_sensitivity(alpha_range)
    print(alpha_results.to_string())
    
    # Step 7: 策略建议输出
    print("\n" + "=" * 70)
    print("策略建议 (Strategic Recommendations)")
    print("=" * 70)
    
    total_rockets = sum(solution['rockets'])
    total_equip = sum(solution['equipment_shipped'])
    final_eta = solution['efficiency'][-1]
    
    print(f"1. 发射策略: 10年共需发射 {total_rockets} 枚火箭")
    print(f"   - 前2年高频发射 (建设期)，建立初始库存")
    print(f"   - 第3-5年重点运送设备，将回收率从{params.initial_eta*100}%提升至{final_eta*100:.1f}%")
    print(f"   - 第6年后进入维持阶段，发射频率降低")
    
    print(f"\n2. 设备投资策略: 累计运送 {total_equip:.1f} 吨水循环设备")
    print(f"   - 早期高投入换取后期自给自足")
    print(f"   - 最终自给率可达 {df['Self_Sufficiency'].iloc[-1]*100:.1f}%")
    
    print(f"\n3. 成本估算: 总物流成本约 ${solution['total_cost']/1e9:.2f}B")
    print(f"   - 人均成本: ${solution['total_cost']/params.population:,.0f}")
    
    print("\n" + "=" * 70)
    print("模型求解完成!")
    print("=" * 70)
    
    return optimizer, df


if __name__ == "__main__":
    # 执行主流程
    optimizer, results_df = main()
    
    # 导出结果到CSV (可选)
    results_df.to_csv("lunar_water_optimization_results.csv", index=False)
    print("\n结果已保存到 lunar_water_optimization_results.csv")
