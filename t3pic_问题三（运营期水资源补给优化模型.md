根据您的问题三（运营期水资源补给优化模型），我设计了几种高级且有价值的图表，并提供Python代码。这些图表既紧密相关又具有创新性：

## 1. **动态库存平衡甘特图（3D）**
展示库存、补给、消耗的动态平衡关系

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_3d_inventory_gantt():
    """创建3D甘特图展示库存动态"""
    fig = plt.figure(figsize=(16, 8))
    
    # 3D视图1：库存、补给、消耗
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 生成模拟数据
    weeks = np.arange(1, 101)
    
    # 模拟数据
    supply_water = np.where(weeks <= 20, 
                           np.random.uniform(80, 120, 100) * (1 - 0.02*weeks), 
                           np.random.uniform(10, 30, 100))
    supply_equipment = np.where(weeks <= 20, 
                               np.random.uniform(5, 15, 100), 
                               np.random.uniform(1, 3, 100))
    consumption = np.random.uniform(90, 110, 100)
    inventory = np.cumsum(supply_water - consumption) + 1000
    recycling_rate = np.minimum(0.8 + 0.01*weeks, 0.99)
    
    # 3D柱状图
    xpos, ypos = np.meshgrid(weeks[::2], [0, 1, 2])
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    
    dx = 0.8 * np.ones_like(zpos)
    dy = 0.8 * np.ones_like(zpos)
    dz = []
    
    for i, week in enumerate(weeks[::2]):
        dz.extend([supply_water[i*2], supply_equipment[i*2], consumption[i*2]])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['补给水', '补给设备', '消耗']
    
    for i in range(3):
        idx = np.arange(i, len(dz), 3)
        ax1.bar3d(xpos[idx], ypos[idx], zpos[idx], 
                 dx[idx], dy[idx], np.array(dz)[idx],
                 color=colors[i], alpha=0.8, label=labels[i])
    
    ax1.set_xlabel('周数')
    ax1.set_ylabel('物资类型')
    ax1.set_zlabel('吨')
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['补给水', '补给设备', '消耗'])
    ax1.set_title('3D动态补给甘特图', fontsize=14)
    ax1.legend()
    
    # 2D视图2：库存与回收率关系
    ax2 = fig.add_subplot(122)
    
    # 双Y轴图
    ax2_secondary = ax2.twinx()
    
    line1 = ax2.plot(weeks, inventory, 'b-', linewidth=3, label='库存量', alpha=0.7)
    ax2.fill_between(weeks, 810, inventory, alpha=0.2, color='blue')
    ax2.axhline(y=810, color='r', linestyle='--', alpha=0.5, label='安全库存')
    
    line2 = ax2_secondary.plot(weeks, recycling_rate*100, 'g-', linewidth=3, 
                              label='回收率', alpha=0.7)
    ax2_secondary.set_ylim(75, 100)
    
    # 组合图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ['安全库存']
    ax2.legend(lines + [plt.Line2D([0], [0], color='red', linestyle='--')], 
              labels, loc='upper left')
    
    ax2.set_xlabel('周数')
    ax2.set_ylabel('库存量（吨）', color='b')
    ax2_secondary.set_ylabel('回收率（%）', color='g')
    ax2.set_title('库存量与回收率动态关系', fontsize=14)
    
    plt.tight_layout()
    return fig

fig = create_3d_inventory_gantt()
plt.show()
```

## 2. **两阶段策略雷达流图**
展示战略投资期和稳定运营期的对比

```python
def create_radar_stream():
    """创建雷达流图展示两阶段策略对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection='polar'))
    
    # 阶段数据
    categories = ['设备投入', '水补给', '回收率', '发射频率', '库存安全', '成本效率']
    N = len(categories)
    
    # 角度设置
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # 阶段1：战略投资期
    values1 = [85, 35, 45, 90, 60, 40]  # 高设备投入，高发射频率
    values1 += values1[:1]
    
    # 阶段2：稳定运营期
    values2 = [15, 25, 95, 10, 95, 90]  # 高回收率，低成本
    values2 += values2[:1]
    
    ax1 = axes[0]
    ax1.plot(angles, values1, 'o-', linewidth=2, color='#FF6B6B')
    ax1.fill(angles, values1, alpha=0.25, color='#FF6B6B')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_title('战略投资期\n（第1-20周）', fontsize=12, fontweight='bold', color='#FF6B6B')
    ax1.grid(True, alpha=0.3)
    
    # 添加流线
    for i in range(N):
        ax1.annotate('', xy=(angles[i], values1[i]), xytext=(angles[i], 0),
                    arrowprops=dict(arrowstyle='->', color='#FF6B6B', alpha=0.5, lw=1))
    
    ax2 = axes[1]
    ax2.plot(angles, values2, 'o-', linewidth=2, color='#4ECDC4')
    ax2.fill(angles, values2, alpha=0.25, color='#4ECDC4')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title('稳定运营期\n（第21-520周）', fontsize=12, fontweight='bold', color='#4ECDC4')
    ax2.grid(True, alpha=0.3)
    
    # 添加流线
    for i in range(N):
        ax2.annotate('', xy=(angles[i], values2[i]), xytext=(angles[i], 0),
                    arrowprops=dict(arrowstyle='->', color='#4ECDC4', alpha=0.5, lw=1))
    
    plt.suptitle('两阶段运营策略对比雷达流图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

fig = create_radar_stream()
plt.show()
```

## 3. **动态回收率演化图（热力图+3D）**
展示回收率随设备投入的动态变化

```python
def create_3d_recycling_evolution():
    """创建3D热力图展示回收率演化"""
    from matplotlib.colors import LinearSegmentedColormap
    
    fig = plt.figure(figsize=(16, 8))
    
    # 创建自定义色彩映射
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    cmap = LinearSegmentedColormap.from_list("custom", colors)
    
    # 生成3D表面数据
    ax1 = fig.add_subplot(121, projection='3d')
    
    weeks = np.arange(0, 101, 5)
    equipment_investment = np.linspace(0, 2, 21)  # 设备投入系数
    
    X, Y = np.meshgrid(weeks, equipment_investment)
    
    # 回收率函数：η = η_0 + α×设备投入 - β×时间衰减
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            week = X[i, j]
            equip = Y[i, j]
            base = 0.8
            improvement = 0.001 * equip * 1000  # α=0.001%/吨
            saturation = 0.99
            Z[i, j] = min(base + improvement * (1 - np.exp(-week/20)), saturation)
    
    # 3D表面图
    surf = ax1.plot_surface(X, Y, Z*100, cmap=cmap, alpha=0.9,
                           linewidth=0.1, antialiased=True)
    
    # 添加等高线投影
    ax1.contour(X, Y, Z*100, zdir='z', offset=75, cmap='viridis', alpha=0.3)
    ax1.contour(X, Y, Z*100, zdir='x', offset=0, cmap='viridis', alpha=0.3)
    ax1.contour(X, Y, Z*100, zdir='y', offset=0, cmap='viridis', alpha=0.3)
    
    ax1.set_xlabel('运营周数', labelpad=10)
    ax1.set_ylabel('设备投入（吨）', labelpad=10)
    ax1.set_zlabel('回收率（%）', labelpad=10)
    ax1.set_title('3D回收率演化表面图', fontsize=14)
    ax1.view_init(elev=25, azim=135)
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    cbar.set_label('回收率（%）')
    
    # 2D热力图
    ax2 = fig.add_subplot(122)
    
    # 创建更精细的热力图数据
    weeks_fine = np.arange(0, 101)
    equip_fine = np.linspace(0, 2, 101)
    X_fine, Y_fine = np.meshgrid(weeks_fine, equip_fine)
    
    Z_fine = np.zeros_like(X_fine)
    for i in range(X_fine.shape[0]):
        for j in range(X_fine.shape[1]):
            week = X_fine[i, j]
            equip = Y_fine[i, j]
            base = 0.8
            improvement = 0.001 * equip * 1000
            saturation = 0.99
            Z_fine[i, j] = min(base + improvement * (1 - np.exp(-week/20)), saturation)
    
    im = ax2.imshow(Z_fine*100, aspect='auto', cmap=cmap, 
                   extent=[0, 100, 0, 2], origin='lower')
    
    # 添加等高线
    contour = ax2.contour(weeks_fine, equip_fine, Z_fine*100, 
                         levels=[85, 90, 95, 98, 99], 
                         colors='white', alpha=0.7, linewidths=1)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.0f%%')
    
    # 标注最佳路径
    optimal_path = np.minimum(1.5 * np.exp(-weeks_fine/40), 0.2)
    ax2.plot(weeks_fine, optimal_path, 'r--', linewidth=3, label='最优投资路径')
    
    ax2.set_xlabel('运营周数')
    ax2.set_ylabel('设备投入（吨）')
    ax2.set_title('回收率演化热力图与最优路径', fontsize=14)
    ax2.legend(loc='upper right')
    
    # 添加颜色条
    cbar2 = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar2.set_label('回收率（%）')
    
    plt.suptitle('水资源回收率动态演化分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

fig = create_3d_recycling_evolution()
plt.show()
```

## 4. **敏感性分析交互式桑基图**
展示参数变化对系统的影响路径

```python
def create_sankey_sensitivity():
    """创建桑基图展示敏感性分析"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 定义节点
    labels = [
        # 输入参数节点
        "火箭载荷", "转化系数", "安全库存", 
        # 中间指标节点
        "单次运输成本", "设备效率", "库存周转率",
        # 最终指标节点
        "总运输成本", "发射次数", "水资源自给率"
    ]
    
    # 定义连接（source, target, value, color）
    sources = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 3, 4, 5]
    targets = [3, 6, 4, 7, 5, 8, 6, 7, 7, 8, 8, 7, 7, 8, 6]
    
    # 设置连接值（敏感性强度）
    values = [45, 25, 30, 20, 15, 10, 35, 30, 25, 20, 15, 10, 25, 20, 15]
    
    # 设置连接颜色（根据影响方向）
    colors = []
    for s, t in zip(sources, targets):
        if t in [6, 7]:  # 成本或发射次数增加
            colors.append("rgba(231, 111, 81, 0.6)")
        else:  # 效率或自给率提高
            colors.append("rgba(42, 157, 143, 0.6)")
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["#264653", "#2a9d8f", "#e9c46a", 
                   "#f4a261", "#e76f51", "#b5838d",
                   "#264653", "#2a9d8f", "#e9c46a"]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])
    
    fig.update_layout(
        title_text="敏感性分析：参数变化对运营指标的影响路径",
        font_size=12,
        height=600,
        annotations=[
            dict(
                x=0.1, y=1.1,
                xref="paper", yref="paper",
                text="输入参数",
                showarrow=False,
                font=dict(size=14, color="#264653")
            ),
            dict(
                x=0.5, y=1.1,
                xref="paper", yref="paper",
                text="中间指标",
                showarrow=False,
                font=dict(size=14, color="#f4a261")
            ),
            dict(
                x=0.9, y=1.1,
                xref="paper", yref="paper",
                text="最终指标",
                showarrow=False,
                font=dict(size=14, color="#e76f51")
            )
        ]
    )
    
    return fig

# 注意：Plotly图表需要保存为HTML或使用Jupyter Notebook显示
# fig = create_sankey_sensitivity()
# fig.show()
```

## 5. **鲁棒性检验网络图**
展示多场景下的系统稳定性

```python
def create_robustness_network():
    """创建网络图展示鲁棒性检验结果"""
    import networkx as nx
    
    fig = plt.figure(figsize=(14, 10))
    
    # 创建网络图
    G = nx.Graph()
    
    # 节点：场景和指标
    scenarios = ["基准场景", "设备短缺", "需求激增", "发射受限", "库存触底"]
    metrics = ["可行性", "成本增幅", "发射增幅", "安全率", "满足率"]
    all_nodes = scenarios + metrics
    
    # 添加节点
    for node in all_nodes:
        G.add_node(node)
    
    # 添加边（场景与指标的关系）
    edges_with_weights = [
        # 基准场景
        ("基准场景", "可行性", 1.0), ("基准场景", "成本增幅", 0.0),
        ("基准场景", "发射增幅", 0.0), ("基准场景", "安全率", 1.0),
        ("基准场景", "满足率", 1.0),
        
        # 设备短缺
        ("设备短缺", "可行性", 1.0), ("设备短缺", "成本增幅", 0.053),
        ("设备短缺", "发射增幅", 0.062), ("设备短缺", "安全率", 1.0),
        ("设备短缺", "满足率", 1.0),
        
        # 需求激增
        ("需求激增", "可行性", 1.0), ("需求激增", "成本增幅", 0.045),
        ("需求激增", "发射增幅", 0.051), ("需求激增", "安全率", 1.0),
        ("需求激增", "满足率", 1.0),
        
        # 发射受限
        ("发射受限", "可行性", 1.0), ("发射受限", "成本增幅", 0.070),
        ("发射受限", "发射增幅", 0.083), ("发射受限", "安全率", 1.0),
        ("发射受限", "满足率", 1.0),
        
        # 库存触底
        ("库存触底", "可行性", 1.0), ("库存触底", "成本增幅", 0.038),
        ("库存触底", "发射增幅", 0.045), ("库存触底", "安全率", 1.0),
        ("库存触底", "满足率", 1.0),
    ]
    
    for edge in edges_with_weights:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    # 节点位置
    pos = {}
    # 场景节点在左侧
    for i, scenario in enumerate(scenarios):
        pos[scenario] = (0.2, 0.8 - i*0.15)
    # 指标节点在右侧
    for i, metric in enumerate(metrics):
        pos[metric] = (0.8, 0.8 - i*0.15)
    
    # 绘制节点
    node_colors = ['#264653']*5 + ['#2a9d8f']*5
    node_sizes = [3000]*5 + [2000]*5
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9)
    
    # 绘制边（根据权重设置宽度和颜色）
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # 标准化权重用于边宽
    edge_widths = [1 + w*50 for w in weights]
    edge_colors = []
    for w in weights:
        if w < 0.02:  # 影响小
            edge_colors.append('#2a9d8f')
        elif w < 0.05:  # 影响中等
            edge_colors.append('#e9c46a')
        else:  # 影响大
            edge_colors.append('#e76f51')
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths,
                          edge_color=edge_colors, alpha=0.6)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    
    # 添加边权重标签
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1%}" 
                   for u, v in edges if G[u][v]['weight'] > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    # 添加图例
    plt.scatter([], [], c='#264653', s=300, label='场景节点')
    plt.scatter([], [], c='#2a9d8f', s=200, label='指标节点')
    plt.plot([], [], color='#2a9d8f', linewidth=3, label='影响小 (<2%)')
    plt.plot([], [], color='#e9c46a', linewidth=3, label='影响中 (2-5%)')
    plt.plot([], [], color='#e76f51', linewidth=3, label='影响大 (>5%)')
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.title('多场景鲁棒性检验网络图', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return fig

fig = create_robustness_network()
plt.show()
```

## 图表使用建议：

1. **3D动态补给甘特图** → 放在3.2模型建立部分，展示系统动态平衡
2. **两阶段策略雷达流图** → 放在3.3.1最优策略部分，直观对比两阶段特征
3. **3D回收率演化图** → 放在3.3.2敏感性分析部分，展示参数影响
4. **桑基图（如使用Plotly）** → 放在敏感性分析部分，展示影响路径
5. **鲁棒性网络图** → 放在3.3.3鲁棒性验证部分，展示多场景分析

这些图表都具有：
- **高度相关性**：直接对应模型三的核心内容
- **视觉新颖性**：采用3D、雷达、网络、热力图等高级形式
- **信息丰富性**：多层数据展示
- **专业美观性**：使用科学配色方案

您可以根据需要选择其中2-3个图表放入论文，建议优先选择前3个（甘特图、雷达流图、3D回收率图），因为它们最直观且容易理解。