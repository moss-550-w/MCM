# ================================
# 导入所需库
# ================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
import matplotlib
import textwrap

# 设置中文字体和样式
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ================================
# 图1：基础模型年际调度示意图
# ================================
def plot_basic_model_schedule():
    """基础模型年际调度示意图"""
    # 创建数据
    years = np.arange(1, 131)
    np.random.seed(42)  # 固定随机种子
    rocket_launches = np.zeros(130)
    carbon_emissions = np.zeros(130)
    carbon_limit = 180000  # 碳排放上限
    
    # 模拟数据：高发射年和低发射年交替
    for i in range(130):
        if i % 10 < 5:  # 高发射年
            rocket_launches[i] = 2000
            carbon_emissions[i] = 1005370  # 超过上限
        else:  # 低发射年
            rocket_launches[i] = np.random.randint(350, 500)
            carbon_emissions[i] = np.random.randint(181370, 194870)  # 接近上限
    
    fig, ax1 = plt.subplots(figsize=(14, 6), constrained_layout=True)
    
    # 火箭发射柱状图
    ax1.bar(years, rocket_launches, alpha=0.7, color='steelblue', label='Rocket launches')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Rocket launches (per year)', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xlim(0, 131)
    
    # 碳排放折线图（第二个y轴）
    ax2 = ax1.twinx()
    ax2.plot(years, carbon_emissions, 'r-', linewidth=2.5, label='Carbon emissions')
    ax2.axhline(y=carbon_limit, color='green', linestyle='--', linewidth=2, 
                label='Emission cap', alpha=0.7)
    ax2.fill_between(years, carbon_limit, carbon_emissions, 
                     where=(carbon_emissions > carbon_limit), 
                     color='red', alpha=0.3, label='Above cap')
    ax2.set_ylabel('Carbon emissions (tCO2e)', color='darkred', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # 标注高发射年和低发射年区域
    ax1.axvspan(1, 50, alpha=0.1, color='gray', label='Infrastructure')
    ax1.axvspan(51, 100, alpha=0.1, color='lightblue', label='Operations')
    ax1.axvspan(101, 130, alpha=0.1, color='lightgreen', label='Stabilization')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        fontsize=10,
        frameon=True,
    )

    fig.savefig('basic_model_schedule.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 图2：三方案雷达图对比
# ================================
def plot_three_scenarios_radar():
    """三种最优方案雷达图对比"""
    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    
    def radar_factory(num_vars, frame='circle'):
        """创建一个雷达图坐标系"""
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        
        class RadarAxes(PolarAxes):
            name = 'radar'
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')
            
            def fill(self, *args, closed=True, **kwargs):
                return super().fill(closed=closed, *args, **kwargs)
            
            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)
                return lines
            
            def _close_line(self, line):
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)
        
        register_projection(RadarAxes)
        return theta
    
    # 定义指标（逆指标，越小越好）
    categories = ['Carbon\n(10k tCO2e)', 'Total cost\n($100M)', 'Duration\n(years)']
    N = len(categories)
    
    # 三种方案的数据（已归一化，1为最优）
    # 注意：这里需要将实际值转换为评分（越小越好）
    data = {
        'Eco-first': [0.95, 0.85, 0.80],
        'Cost-first': [0.70, 0.95, 0.95],
        'Balanced': [0.90, 0.88, 0.82],
    }
    
    # 创建雷达图
    theta = radar_factory(N, frame='polygon')
    
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75])
    ax = fig.add_subplot(gs[0, 0], projection='radar')
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    
    colors = ['#2E8B57', '#FF6347', '#4682B4']
    
    for i, (scenario, values) in enumerate(data.items()):
        ax.plot(theta, values, color=colors[i], linewidth=2.5, label=scenario)
        ax.fill(theta, values, color=colors[i], alpha=0.2)
    
    # 设置角度标签
    ax.set_xticks(theta)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # 设置径向标签
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 添加实际数值标注
    actual_values = {
        'Eco-first': [10150.5, 209992.9, 130],
        'Cost-first': [10500.8, 208579.8, 129],
        'Balanced': [10165.1, 210297.1, 130]
    }
    
    handles, labels = ax.get_legend_handles_labels()
    ax_info.legend(handles, labels, loc='upper left', frameon=True, fontsize=11)

    info_lines = []
    for scenario, values in actual_values.items():
        info_lines.append(
            f"{scenario}\n"
            f"Carbon: {values[0]:.1f} (10k t)\n"
            f"Cost: {values[1]:.1f} ($100M)\n"
            f"Duration: {values[2]} yr"
        )
    info_text = "\n\n".join(info_lines)
    ax_info.text(
        0,
        0.72,
        info_text,
        ha='left',
        va='top',
        fontsize=10,
        linespacing=1.25,
        wrap=True,
    )

    fig.savefig('three_scenarios_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 图3：阶段化权重与运输结构堆叠图
# ================================
def plot_stage_weight_transport():
    """阶段化权重与运输结构堆叠图"""
    # 创建数据
    stages = ['Infrastructure\n(0-50 yr)', 'Operations\n(51-100 yr)', 'Stabilization\n(101-130 yr)']
    x = np.arange(len(stages))
    
    # 运输结构数据（火箭vs电梯，单位：万公吨/年）
    rocket_transport = [80, 60, 20]  # 逐年减少
    elevator_transport = [20, 40, 80]  # 逐年增加
    
    # AHP权重数据（环境权重）
    env_weights = [0.3, 0.5, 0.7]  # 环境权重逐年增加
    
    fig, ax1 = plt.subplots(figsize=(12, 7), constrained_layout=True)
    
    # 堆叠柱状图（运输结构）
    bar_width = 0.6
    bar1 = ax1.bar(x, rocket_transport, bar_width,
                   color='#FF6B6B', alpha=0.8, label='Rocket transport')
    bar2 = ax1.bar(x, elevator_transport, bar_width,
                   bottom=rocket_transport, color='#4ECDC4', alpha=0.8, label='Space elevator transport')
    
    ax1.set_ylabel('Annual throughput (10k tonnes)', fontsize=12)
    ax1.set_xlabel('Project stage', fontsize=12)
    ax1.set_ylim(0, 120)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=11)
    
    # 在柱状图上标注百分比
    for i in range(len(stages)):
        total = rocket_transport[i] + elevator_transport[i]
        rocket_pct = rocket_transport[i] / total * 100
        elevator_pct = elevator_transport[i] / total * 100
        
        center_x = bar1[i].get_x() + bar1[i].get_width() / 2
        ax1.text(center_x, rocket_transport[i]/2, f'{rocket_pct:.0f}%',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax1.text(center_x, rocket_transport[i] + elevator_transport[i]/2, f'{elevator_pct:.0f}%',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 第二个y轴（AHP环境权重）
    ax2 = ax1.twinx()
    line = ax2.plot(x, env_weights, 's-', color='#6A0572',
                   linewidth=3, markersize=10, label='Environmental weight')
    ax2.set_ylabel('AHP environmental weight', color='#6A0572', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#6A0572')
    ax2.set_ylim(0, 1)
    ax2.grid(False)
    
    # 在折线上标注权重值
    for i, weight in enumerate(env_weights):
        ax2.text(x[i], weight+0.03, f'{weight:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#6A0572')
    
    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        fontsize=10,
        framealpha=0.9,
    )
    
    # 添加网格线
    ax1.yaxis.grid(True, alpha=0.3)
    
    fig.savefig('stage_weight_transport.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 图4：关键参数敏感性折线图
# ================================
def plot_sensitivity_analysis():
    """关键参数敏感性折线图"""
    # 创建参数变化范围
    param_change = np.arange(-20, 25, 5)  # -20% 到 +20%，步长5%
    
    # 模拟敏感性数据（百分比变化）
    # 这里使用线性模型模拟
    np.random.seed(42)
    
    # E_t_limit 对碳排放的影响
    emission_impact = -1.02 * param_change + np.random.normal(0, 1, len(param_change))
    # e1 对碳排放的影响
    e1_impact = 0.97 * param_change + np.random.normal(0, 0.5, len(param_change))
    # M 对总成本的影响
    M_cost_impact = 0.6 * param_change + np.random.normal(0, 0.8, len(param_change))
    # ω_{s,1} 对碳排放的影响
    omega_impact = 0.035 * param_change + np.random.normal(0, 0.1, len(param_change))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    label_fs = 14 +4
    title_fs = 16 +4
    legend_fs = 12 +4
    tick_fs = 12 +4
    
    # 子图1: E_t_limit 的影响
    ax1 = axes[0]
    ax1.plot(param_change, emission_impact, 'o-', linewidth=2.5, color='#E74C3C', markersize=8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.fill_between(param_change, 0, emission_impact, where=(emission_impact>0), 
                     color='#E74C3C', alpha=0.2)
    ax1.fill_between(param_change, 0, emission_impact, where=(emission_impact<0), 
                     color='#2ECC71', alpha=0.2)
    ax1.set_xlabel('Emission cap change (%)', fontsize=label_fs)
    ax1.set_ylabel('Total emissions change (%)', fontsize=label_fs)
    ax1.set_title(r'Sensitivity: $E_t^{\mathrm{limit}}$', fontsize=title_fs, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=tick_fs)
    ax1.grid(True, alpha=0.3)
    # 添加回归线
    z = np.polyfit(param_change, emission_impact, 1)
    p = np.poly1d(z)
    ax1.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'Slope: {z[0]:.2f}')
    ax1.legend(loc='upper left', fontsize=legend_fs)
    
    # 子图2: e1 的影响
    ax2 = axes[1]
    ax2.plot(param_change, e1_impact, 's-', linewidth=2.5, color='#3498DB', markersize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.fill_between(param_change, 0, e1_impact, where=(e1_impact>0), 
                     color='#3498DB', alpha=0.2)
    ax2.set_xlabel('Rocket emission factor change (%)', fontsize=label_fs)
    ax2.set_ylabel('Total emissions change (%)', fontsize=label_fs)
    ax2.set_title(r'Sensitivity: $e_1$', fontsize=title_fs, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=tick_fs)
    ax2.grid(True, alpha=0.3)
    z = np.polyfit(param_change, e1_impact, 1)
    p = np.poly1d(z)
    ax2.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'Slope: {z[0]:.2f}')
    ax2.legend(loc='upper left', fontsize=legend_fs)
    
    # 子图3: M 的影响
    ax3 = axes[2]
    ax3.plot(param_change, M_cost_impact, '^-', linewidth=2.5, color='#9B59B6', markersize=8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.fill_between(param_change, 0, M_cost_impact, where=(M_cost_impact>0), 
                     color='#9B59B6', alpha=0.2)
    ax3.set_xlabel('Penalty factor change (%)', fontsize=label_fs)
    ax3.set_ylabel('Total cost change (%)', fontsize=label_fs)
    ax3.set_title(r'Sensitivity: $M$', fontsize=title_fs, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=tick_fs)
    ax3.grid(True, alpha=0.3)
    z = np.polyfit(param_change, M_cost_impact, 1)
    p = np.poly1d(z)
    ax3.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'Slope: {z[0]:.2f}')
    ax3.legend(loc='upper left', fontsize=legend_fs)
    
    # 子图4: ω_{s,1} 的影响
    ax4 = axes[3]
    ax4.plot(param_change, omega_impact, 'D-', linewidth=2.5, color='#F39C12', markersize=8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.fill_between(param_change, 0, omega_impact, where=(omega_impact>0), 
                     color='#F39C12', alpha=0.2)
    ax4.set_xlabel('Environmental weight change (%)', fontsize=label_fs)
    ax4.set_ylabel('Emissions change (%)', fontsize=label_fs)
    ax4.set_title(r'Sensitivity: $\omega_{s,1}$', fontsize=title_fs, fontweight='bold')
    ax4.tick_params(axis='both', labelsize=tick_fs)
    ax4.grid(True, alpha=0.3)
    z = np.polyfit(param_change, omega_impact, 1)
    p = np.poly1d(z)
    ax4.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'Slope: {z[0]:.3f}')
    ax4.legend(loc='upper left', fontsize=legend_fs)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 图5：模型框架演进示意图
# ================================
def plot_model_evolution():
    """模型框架演进示意图"""
    fig, ax = plt.subplots(figsize=(13.8, 5.4), constrained_layout=True)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 定义模型演进阶段
    stages = [
        {
            'name': 'Single-objective baseline',
            'desc': 'Integer programming + constraint relaxation\n• Objective: minimize emissions\n• Constraints: total demand, capacity limits\n• Key idea: slack variables for hard constraints',
            'color': '#3498DB'
        },
        {
            'name': 'Multi-objective extension',
            'desc': 'Multi-objective IP + GA + AHP\n• Tradeoff: environment / cost / time\n• Key idea: stage-wise dynamic AHP weights\n• Solver: GA-based Pareto front',
            'color': '#2ECC71'
        },
        {
            'name': 'Learning-curve dynamics',
            'desc': 'Time-varying parameters + learning curve\n• Update: $e(t)=e_0·exp(-λt)$\n• Captures: technological progress and efficiency gains\n• Benefit: closer to real engineering evolution',
            'color': '#F39C12'
        },
        {
            'name': 'Lifecycle assessment (LCA)',
            'desc': 'LCA + multi-pollutant optimization\n• Scope: "launch-to-arrival" -> "cradle-to-grave"\n• Metrics: CO2, NOx, black carbon with separate caps\n• Output: full environmental footprint',
            'color': '#9B59B6'
        }
    ]

    def wrap_multiline(s: str, width: int) -> str:
        return "\n".join(
            textwrap.fill(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            for line in s.split("\n")
        )
    
    stage_y = 0.60
    box_w, box_h = 0.12, 0.305
    x_centers = np.linspace(0.26, 0.86, len(stages))

    for i in range(len(stages) - 1):
        x0 = x_centers[i] + box_w / 2 + 0.015
        x1 = x_centers[i + 1] - box_w / 2 - 0.015
        ax.annotate(
            '',
            xy=(x1, stage_y),
            xytext=(x0, stage_y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle='->', lw=1.8, color='gray', alpha=0.7),
        )
    
    # 绘制每个阶段的框
    for i, stage in enumerate(stages):
        # 绘制主框
        box = mpatches.FancyBboxPatch((x_centers[i] - box_w / 2, stage_y - box_h / 2), box_w, box_h,
                                      boxstyle="round,pad=0.011",
                                      facecolor=stage['color'], alpha=0.9,
                                      edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # 添加阶段名称
        ax.text(x_centers[i], stage_y + box_h * 0.34, stage['name'],
                ha='center', va='center', fontsize=10.0, fontweight='bold', color='white', wrap=True)
        
        # 添加阶段描述
        desc = wrap_multiline(stage['desc'], 25)
        ax.text(x_centers[i], stage_y - box_h * 0.05, desc,
                ha='center', va='center', fontsize=8.0, color='white', linespacing=1.10, wrap=True,
                bbox=dict(boxstyle="round,pad=0.15", facecolor='black', alpha=0.22))
    
    # 添加底部说明
    ax.text(0.5, stage_y - box_h / 2 - 0.08, 
            '                                              ▲ Each stage extends capabilities and improves fidelity, forming a complete environmental decision framework.',
            ha='center', va='center', fontsize=9.5, style='italic',
            transform=ax.transAxes)
    
    fig.savefig('model_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 主程序：生成所有图表
# ================================
def generate_all_figures():
    """生成所有5幅图表"""
    print("生成图1: 基础模型年际调度示意图...")
    plot_basic_model_schedule()
    
    print("生成图2: 三方案雷达图对比...")
    plot_three_scenarios_radar()
    
    print("生成图3: 阶段化权重与运输结构堆叠图...")
    plot_stage_weight_transport()
    
    print("生成图4: 关键参数敏感性折线图...")
    plot_sensitivity_analysis()
    
    print("生成图5: 模型框架演进示意图...")
    plot_model_evolution()
    
    print("所有图表已生成完成！")

# 执行生成所有图表
if __name__ == "__main__":
    generate_all_figures()
