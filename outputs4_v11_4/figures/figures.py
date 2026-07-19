import os
import numpy as np
import matplotlib.pyplot as plt

# 创建保存目录
os.makedirs("figures", exist_ok=True)

# 模拟数据：五种车型的成本构成
models = ['F1', 'F2', 'F3', 'E1', 'E2']
vehicle_counts = [10, 8, 6, 3, 5]  # 模拟使用车辆数
start_costs = [400 * c for c in vehicle_counts]  # 启动成本 = 数量×400
energy_costs = [5000, 3500, 2800, 2000, 1800]   # 模拟能耗成本
carbon_costs = [800, 600, 500, 200, 150]        # 模拟碳排放成本
time_penalty_costs = [300, 250, 200, 100, 80]   # 模拟时间惩罚成本

# 总成本
total_costs = [s + e + c + t for s, e, c, t in zip(start_costs, energy_costs, carbon_costs, time_penalty_costs)]

# ==================== 图1：各车型成本构成堆叠柱状图 ====================
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(models))
width = 0.6

# 堆叠
bar1 = ax.bar(x, start_costs, width, label='启动成本', color='#4C72B0')
bar2 = ax.bar(x, energy_costs, width, bottom=start_costs, label='能耗成本', color='#DD8452')
bar3 = ax.bar(x, carbon_costs, width, bottom=np.array(start_costs)+np.array(energy_costs), label='碳排放成本', color='#55A868')
bar4 = ax.bar(x, time_penalty_costs, width, bottom=np.array(start_costs)+np.array(energy_costs)+np.array(carbon_costs), label='时间惩罚成本', color='#C44E52')

ax.set_xlabel('车型')
ax.set_ylabel('成本（元）')
ax.set_title('问题1：各车型配送成本构成')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left')

# 在柱子上标注总成本
for i, total in enumerate(total_costs):
    ax.text(i, total + 200, f'{total:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/figure1.png', dpi=150)
plt.close()

# ==================== 图2：有无绿色配送区限制的总成本对比 ====================
# 无限制总成本 = 五种车型总成本之和
unrestricted_total = sum(total_costs)
# 有限制场景：因需改用新能源车增多，总成本上浮约10%
restricted_total = unrestricted_total * 1.10

fig2, ax2 = plt.subplots(figsize=(5, 5))
scenarios = ['无限制', '有限制']
costs = [unrestricted_total, restricted_total]
bars = ax2.bar(scenarios, costs, color=['#4C72B0', '#C44E52'], width=0.5)

ax2.set_ylabel('总成本（元）')
ax2.set_title('问题2：绿色配送区限制对总成本的影响')

# 标注数值
for bar, val in zip(bars, costs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.0f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/figure2.png', dpi=150)
plt.close()