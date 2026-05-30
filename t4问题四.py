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
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # 火箭发射柱状图
    ax1.bar(years, rocket_launches, alpha=0.7, color='steelblue', label='火箭发射数')
    ax1.set_xlabel('年份', fontsize=12)
    ax1.set_ylabel('火箭发射数（枚/年）', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xlim(0, 131)
    
    # 碳排放折线图（第二个y轴）
    ax2 = ax1.twinx()
    ax2.plot(years, carbon_emissions, 'r-', linewidth=2.5, label='碳排放量')
    ax2.axhline(y=carbon_limit, color='green', linestyle='--', linewidth=2, 
                label='碳排放上限', alpha=0.7)
    ax2.fill_between(years, carbon_limit, carbon_emissions, 
                     where=(carbon_emissions > carbon_limit), 
                     color='red', alpha=0.3, label='超额排放')
    ax2.set_ylabel('碳排放量（吨CO₂e）', color='darkred', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # 标注高发射年和低发射年区域
    ax1.axvspan(1, 50, alpha=0.1, color='gray', label='基建期')
    ax1.axvspan(51, 100, alpha=0.1, color='lightblue', label='运营期')
    ax1.axvspan(101, 130, alpha=0.1, color='lightgreen', label='稳定期')
    
    # 设置标题和图例
    plt.title('基础模型年际调度示意图\n火箭发射数与碳排放量动态变化 (130年周期)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('basic_model_schedule.png', dpi=300, bbox_inches='tight')
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
    categories = ['碳排放\n(万吨CO₂e)', '总成本\n(亿美元)', '总周期\n(年)']
    N = len(categories)
    
    # 三种方案的数据（已归一化，1为最优）
    # 注意：这里需要将实际值转换为评分（越小越好）
    data = {
        '环保优先': [0.95, 0.85, 0.80],  # 碳排放低，成本中等，周期长
        '成本优先': [0.70, 0.95, 0.95],  # 碳排放高，成本低，周期短
        '均衡优化': [0.90, 0.88, 0.82],  # 各项均衡
    }
    
    # 创建雷达图
    theta = radar_factory(N, frame='polygon')
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='radar'))
    
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
    
    # 添加图例和标题
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    plt.title('三种最优方案三维对比雷达图\n(数值越小表示性能越好)', 
              fontsize=14, fontweight='bold', pad=30)
    
    # 添加实际数值标注
    actual_values = {
        '环保优先': [10150.5, 209992.9, 130],
        '成本优先': [10500.8, 208579.8, 129],
        '均衡优化': [10165.1, 210297.1, 130]
    }
    
    # 在图中添加实际数值
    for i, (scenario, values) in enumerate(actual_values.items()):
        text = f"{scenario}:\n碳排放: {values[0]:.1f}万吨\n成本: {values[1]:.1f}亿$\n周期: {values[2]}年"
        ax.text(0.5, 0.9 - i*0.15, text, transform=ax.transAxes, 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor=colors[i], alpha=0.1))
    
    plt.tight_layout()
    plt.savefig('three_scenarios_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 图3：阶段化权重与运输结构堆叠图
# ================================
def plot_stage_weight_transport():
    """阶段化权重与运输结构堆叠图"""
    # 创建数据
    stages = ['基建期\n(0-50年)', '运营期\n(51-100年)', '稳定期\n(101-130年)']
    
    # 运输结构数据（火箭vs电梯，单位：万公吨/年）
    rocket_transport = [80, 60, 20]  # 逐年减少
    elevator_transport = [20, 40, 80]  # 逐年增加
    
    # AHP权重数据（环境权重）
    env_weights = [0.3, 0.5, 0.7]  # 环境权重逐年增加
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 堆叠柱状图（运输结构）
    bar_width = 0.6
    bar1 = ax1.bar(stages, rocket_transport, bar_width, 
                   color='#FF6B6B', alpha=0.8, label='火箭运输')
    bar2 = ax1.bar(stages, elevator_transport, bar_width, 
                   bottom=rocket_transport, color='#4ECDC4', alpha=0.8, label='电梯运输')
    
    ax1.set_ylabel('年运输量（万公吨）', fontsize=12)
    ax1.set_xlabel('建设阶段', fontsize=12)
    ax1.set_ylim(0, 110)
    
    # 在柱状图上标注百分比
    for i in range(len(stages)):
        total = rocket_transport[i] + elevator_transport[i]
        rocket_pct = rocket_transport[i] / total * 100
        elevator_pct = elevator_transport[i] / total * 100
        
        ax1.text(i, rocket_transport[i]/2, f'{rocket_pct:.0f}%', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax1.text(i, rocket_transport[i] + elevator_transport[i]/2, f'{elevator_pct:.0f}%', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 第二个y轴（AHP环境权重）
    ax2 = ax1.twinx()
    line = ax2.plot(stages, env_weights, 's-', color='#6A0572', 
                   linewidth=3, markersize=10, label='环境权重')
    ax2.set_ylabel('AHP环境权重', color='#6A0572', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#6A0572')
    ax2.set_ylim(0, 1)
    ax2.grid(False)
    
    # 在折线上标注权重值
    for i, weight in enumerate(env_weights):
        ax2.text(i, weight+0.03, f'{weight:.2f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='#6A0572')
    
    # 设置标题
    plt.title('阶段化运输结构与AHP权重变化趋势\n（运输占比与环境优先级的协同演变）', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, 
               loc='upper left', fontsize=10, framealpha=0.9)
    
    # 添加网格线
    ax1.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stage_weight_transport.png', dpi=300, bbox_inches='tight')
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
    
    # 子图1: E_t_limit 的影响
    ax1 = axes[0]
    ax1.plot(param_change, emission_impact, 'o-', linewidth=2.5, color='#E74C3C', markersize=8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.fill_between(param_change, 0, emission_impact, where=(emission_impact>0), 
                     color='#E74C3C', alpha=0.2)
    ax1.fill_between(param_change, 0, emission_impact, where=(emission_impact<0), 
                     color='#2ECC71', alpha=0.2)
    ax1.set_xlabel('碳排放约束变化率 (%)', fontsize=11)
    ax1.set_ylabel('总碳排放变化率 (%)', fontsize=11)
    ax1.set_title(r'参数 $E_t^{\mathrm{limit}}$ 敏感性', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # 添加回归线
    z = np.polyfit(param_change, emission_impact, 1)
    p = np.poly1d(z)
    ax1.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'斜率: {z[0]:.2f}')
    ax1.legend(loc='upper left', fontsize=9)
    
    # 子图2: e1 的影响
    ax2 = axes[1]
    ax2.plot(param_change, e1_impact, 's-', linewidth=2.5, color='#3498DB', markersize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.fill_between(param_change, 0, e1_impact, where=(e1_impact>0), 
                     color='#3498DB', alpha=0.2)
    ax2.set_xlabel('火箭碳排放系数变化率 (%)', fontsize=11)
    ax2.set_ylabel('总碳排放变化率 (%)', fontsize=11)
    ax2.set_title(r'参数 $e_1$ 敏感性', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    z = np.polyfit(param_change, e1_impact, 1)
    p = np.poly1d(z)
    ax2.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'斜率: {z[0]:.2f}')
    ax2.legend(loc='upper left', fontsize=9)
    
    # 子图3: M 的影响
    ax3 = axes[2]
    ax3.plot(param_change, M_cost_impact, '^-', linewidth=2.5, color='#9B59B6', markersize=8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.fill_between(param_change, 0, M_cost_impact, where=(M_cost_impact>0), 
                     color='#9B59B6', alpha=0.2)
    ax3.set_xlabel('惩罚系数变化率 (%)', fontsize=11)
    ax3.set_ylabel('总成本变化率 (%)', fontsize=11)
    ax3.set_title(r'参数 $M$ 敏感性', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    z = np.polyfit(param_change, M_cost_impact, 1)
    p = np.poly1d(z)
    ax3.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'斜率: {z[0]:.2f}')
    ax3.legend(loc='upper left', fontsize=9)
    
    # 子图4: ω_{s,1} 的影响
    ax4 = axes[3]
    ax4.plot(param_change, omega_impact, 'D-', linewidth=2.5, color='#F39C12', markersize=8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.fill_between(param_change, 0, omega_impact, where=(omega_impact>0), 
                     color='#F39C12', alpha=0.2)
    ax4.set_xlabel('环境权重变化率 (%)', fontsize=11)
    ax4.set_ylabel('碳排放变化率 (%)', fontsize=11)
    ax4.set_title(r'参数 $\omega_{s,1}$ 敏感性', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    z = np.polyfit(param_change, omega_impact, 1)
    p = np.poly1d(z)
    ax4.plot(param_change, p(param_change), 'k--', linewidth=1, alpha=0.7, 
             label=f'斜率: {z[0]:.3f}')
    ax4.legend(loc='upper left', fontsize=9)
    
    # 添加总标题
    fig.suptitle('关键参数敏感性分析：参数变化对输出指标的影响', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 图5：模型框架演进示意图
# ================================
def plot_model_evolution():
    """模型框架演进示意图"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    # 定义模型演进阶段
    stages = [
        {
            'name': '基础单目标模型',
            'desc': '整数规划+约束松弛\n• 目标：最小化碳排放\n• 约束：物资总量、运力上限\n• 创新：松弛变量处理硬约束',
            'color': '#3498DB'
        },
        {
            'name': '创新多目标模型',
            'desc': '多目标整数规划+遗传算法+AHP\n• 目标：环保/成本/时间三维权衡\n• 创新：阶段化AHP动态权重\n• 求解：遗传算法帕累托最优',
            'color': '#2ECC71'
        },
        {
            'name': '动态学习曲线扩展',
            'desc': '时变参数+学习曲线\n• 改进：$e(t)=e_0·exp(-λt)$\n• 适应：技术进步与效率提升\n• 优势：更贴近实际工程演进',
            'color': '#F39C12'
        },
        {
            'name': '全生命周期评估',
            'desc': 'LCA框架+多污染物优化\n• 扩展：从"发射到抵达"到"采矿到报废"\n• 多指标：CO₂、NOₓ、黑碳独立约束\n• 评估：完整的环境足迹',
            'color': '#9B59B6'
        }
    ]
    
    # 绘制演进箭头
    arrow_x = np.linspace(0.1, 0.9, len(stages))
    arrow_y = [0.5] * len(stages)
    
    # 绘制连接箭头
    for i in range(len(stages)-1):
        ax.annotate('', xy=(arrow_x[i+1], arrow_y[i]), xytext=(arrow_x[i], arrow_y[i]),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.7))
    
    # 绘制每个阶段的框
    for i, stage in enumerate(stages):
        # 绘制主框
        box = mpatches.FancyBboxPatch((arrow_x[i]-0.1, arrow_y[i]-0.2), 0.2, 0.4,
                                      boxstyle="round,pad=0.02",
                                      facecolor=stage['color'], alpha=0.9,
                                      edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # 添加阶段名称
        ax.text(arrow_x[i], arrow_y[i]+0.15, stage['name'], 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # 添加阶段描述
        ax.text(arrow_x[i], arrow_y[i]-0.05, stage['desc'], 
                ha='center', va='center', fontsize=9, color='white',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.3))
    
    # 添加标题
    plt.title('模型框架演进路径：从单目标优化到全生命周期评估', 
              fontsize=14, fontweight='bold', pad=30)
    
    # 添加底部说明
    ax.text(0.5, 0.02, 
            '▲ 每个阶段都在前一阶段基础上扩展功能和提升精度，形成完整的环境决策框架',
            ha='center', va='center', fontsize=10, style='italic',
            transform=ax.transAxes)
    
    # 添加演进方向标注
    ax.text(0.05, 0.7, '模型复杂度与精度', 
            rotation=90, ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(0.07, 0.3), xytext=(0.07, 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig('model_evolution.png', dpi=300, bbox_inches='tight')
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