import os, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 创建保存目录
os.makedirs("figures", exist_ok=True)

# 设置字体避免中文字符缺失警告
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 生成模拟数据（模拟男胎孕妇的Y染色体浓度数据）
np.random.seed(42)
n_subjects = 20  # 只显示部分个体曲线以免图过密
n_obs_per = 5
GA_all = np.linspace(10, 25, 200)

# 逻辑斯蒂函数
def logistic(t, alpha, beta, gamma):
    return alpha / (1 + np.exp(-beta*(t - gamma)))

# 模拟个体参数（真实场景下由模型估计）
subjects_params = []
for i in range(n_subjects):
    alpha = 1.0  # 归一化上限
    beta = np.random.uniform(0.3, 0.8)
    gamma = np.random.uniform(14, 20)
    subjects_params.append((alpha, beta, gamma))

# 生成观测数据（叠加噪声）
fig1, ax1 = plt.subplots(figsize=(8, 5))
for i, (a,b,g) in enumerate(subjects_params):
    # 每个个体取5个随机时间点
    t_obs = np.sort(np.random.uniform(10, 25, n_obs_per))
    y_true = logistic(t_obs, a, b, g)
    y_noise = y_true + np.random.normal(0, 0.03, n_obs_per)
    y_obs = np.clip(y_noise, 0, 1)
    ax1.scatter(t_obs, y_obs, s=20, alpha=0.6, label=f'Subj {i+1}' if i<5 else '')
    # 对每个个体拟合曲线（仅展示前5个的拟合线）
    if i < 5:
        try:
            popt, _ = curve_fit(logistic, t_obs, y_obs, p0=[0.5, 1, 15], bounds=([0,0,10],[1,5,25]))
            t_fine = np.linspace(10, 25, 100)
            y_fit = logistic(t_fine, *popt)
            ax1.plot(t_fine, y_fit, '--', linewidth=1, alpha=0.8)
        except:
            pass

# 阈值线
ax1.axhline(y=0.04, color='red', linestyle='-', linewidth=2, label='θ = 4.0%')
# 检测窗口
ax1.axvline(x=10, color='gray', linestyle=':', alpha=0.7)
ax1.axvline(x=25, color='gray', linestyle=':', alpha=0.7)
ax1.set_xlabel('Gestational Age (weeks)', fontsize=12)
ax1.set_ylabel('Y chromosome concentration', fontsize=12)
ax1.set_title('Individual Y-Conc growth and logistic fits (first 5 subjects)', fontsize=13)
ax1.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig("figures/figure1.png", dpi=150)
plt.close()

# 图2：分组后最优时点 vs 风险（模拟两组示例）
fig2, ax2 = plt.subplots(figsize=(8,5))
# 模拟两个组
group_labels = ['Low BMI group', 'High BMI group']
optimal_times = [16.5, 19.0]  # 最优时点（周）
risks_at_optimal = [0.12, 0.08]  # 组内平均风险

x_pos = np.arange(len(group_labels))
bars = ax2.bar(x_pos, optimal_times, width=0.4, yerr=0.5, capsize=5,
               color=['#4C72B0', '#DD8452'], alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(group_labels, fontsize=11)
ax2.set_ylabel('Optimal NIPT time (weeks)', fontsize=12)
ax2.set_title('Group-specific optimal testing window', fontsize=13)
ax2.set_ylim(14, 22)
# 标注风险值
for i, (bar, risk) in enumerate(zip(bars, risks_at_optimal)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.3,
             f'Risk={risk:.2f}', ha='center', va='bottom', fontsize=10)
ax2.axhline(y=13, color='gray', linestyle=':', alpha=0.5, label='Detect start')
ax2.axhline(y=22, color='gray', linestyle=':', alpha=0.5, label='Detect end')
ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig("figures/figure2.png", dpi=150)
plt.close()
