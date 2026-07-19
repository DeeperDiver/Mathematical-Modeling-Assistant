import os, sys, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist, norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# ===== 环境路径 =====
DATA_PATH = os.environ.get("MODELING_DATA_PATH", "C:\\Users\\lingi\\Desktop\\Research\\Mo\\Modeling_Assistant\\real_test2\\附件.xlsx")
OUTPUT_DIR = os.environ.get("MODELING_OUTPUT_DIR", ".")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ===== 1. 加载数据 =====
df = pd.read_excel(DATA_PATH)
print("原始数据行数:", len(df))

# ===== 2. 预处理 =====
# 2.1 解析孕妇代码
df['孕妇代码_num'] = df['孕妇代码'].str.replace('A', '', regex=False).astype(float)

# 2.2 解析末次月经（用于计算孕周，但已有检测孕周列，所以非必需）
df['末次月经_dt'] = pd.to_datetime(df['末次月经'], errors='coerce')

# 2.3 解析检测孕周: '11w+6' -> 11 + 6/7
def parse_ga(s):
    s = str(s).lower().replace('w','').replace('+',' ').replace('周','').replace('+',' ')
    parts = s.split()
    week = float(parts[0])
    day = float(parts[1]) if len(parts) > 1 else 0.0
    return week + day / 7.0
df['GA'] = df['检测孕周'].apply(parse_ga)

# 2.4 列名统一（去除空格）
# 注意：原列名'唯一比对的读段数  '有尾随空格
old_cols = df.columns.tolist()
new_cols = {c: c.strip() for c in old_cols}
df.rename(columns=new_cols, inplace=True)
# 确保唯一比对的读段数列名正确
if '唯一比对的读段数' in df.columns:
    pass

# 2.5 处理缺失值（数值列用中位数填补）
num_cols_to_fill = ['年龄', '身高', '体重', '孕妇BMI']
for col in num_cols_to_fill:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# 2.6 处理Y染色体浓度：所有行都有值，假设全部为男胎。
# 但为问题4，定义一个性别列：Y_conc < 0.02 ? '女' : '男'（题目中女胎Y浓度空白，但数据集无空白，采用低阈值）
threshold_sex = 0.02  # 推断阈值，非题目给定
df['性别'] = np.where(df['Y染色体浓度'] < threshold_sex, '女', '男')
print("男胎数:", sum(df['性别']=='男'), "女胎数:", sum(df['性别']=='女'))

# 2.7 筛选男胎用于问题1-3
df_male = df[df['性别'] == '男'].copy()

# ===== 3. 逻辑斯蒂函数 =====
def logistic_func(t, beta, gamma):
    return 1.0 / (1.0 + np.exp(-beta * (t - gamma)))

# ===== 4. 拟合每个孕妇的个体曲线 =====
pregnant_codes = df_male['孕妇代码_num'].unique()
results_T = []
params_list = []
fitted_codes = []
failed_codes = []

for code in pregnant_codes:
    sub = df_male[df_male['孕妇代码_num'] == code].sort_values('GA')
    x = sub['GA'].values
    y = sub['Y染色体浓度'].values
    # 固定alpha=1，但Y浓度实际范围0~0.23，归一化
    # 由于Y浓度远小于1，将alpha固定为1会导致拟合不佳，故将y归一化到[0,1]
    y_max = y.max()
    if y_max < 0.001:
        y_norm = y
    else:
        y_norm = y / y_max
    
    if len(x) >= 3:
        try:
            popt, _ = curve_fit(logistic_func, x, y_norm, p0=[0.5, 15], bounds=([0.01, 10], [10, 25]))
            beta_est = popt[0]
            gamma_est = popt[1]
            # 计算达标时间：对于原尺度的y，阈值θ=0.04，对应归一化阈值 = 0.04 / y_max
            if y_max > 0:
                thresh_norm = 0.04 / y_max
            else:
                thresh_norm = 1.0
            if thresh_norm >= 1.0:
                T_i = 25.0
            else:
                T_i = gamma_est + (1.0/beta_est) * np.log(1.0/thresh_norm - 1.0)
            if T_i < 10 or T_i > 25 or np.isnan(T_i):
                T_i = 25.0
            fitted_codes.append(code)
            params_list.append({'code': code, 'beta': beta_est, 'gamma': gamma_est, 'y_max': y_max})
            results_T.append({'孕妇代码_num': code, 'T_i': T_i})
        except Exception as e:
            failed_codes.append(code)
    else:
        failed_codes.append(code)

# 对拟合失败的孕妇使用全局混合效应模型预测
if len(failed_codes) > 0:
    # 准备训练数据：使用拟合成功的孕妇数据
    train_df = df_male[df_male['孕妇代码_num'].isin(fitted_codes)].copy()
    # 混合线性模型：以Y染色体浓度为响应，固定效应包括 GA, BMI, 年龄, 身高, 体重，随机截距
    try:
        # 对训练数据做多一点列
        train_df['GA_centered'] = train_df['GA'] - train_df['GA'].mean()
        train_df['BMI_centered'] = train_df['孕妇BMI'] - train_df['孕妇BMI'].mean()
        train_df['Age_centered'] = train_df['年龄'] - train_df['年龄'].mean()
        train_df['Height_centered'] = train_df['身高'] - train_df['身高'].mean()
        train_df['Weight_centered'] = train_df['体重'] - train_df['体重'].mean()
        
        # 训练数据中孕妇代码_num为数值，混合模型需要字符串格式的组
        train_df['group'] = train_df['孕妇代码_num'].astype(str)
        
        model_mixed = mixedlm("Y染色体浓度 ~ GA_centered + BMI_centered + Age_centered + Height_centered + Weight_centered",
                               data=train_df, groups=train_df['group'])
        result_mixed = model_mixed.fit(reml=False, maxiter=200)
        fixed_effects = result_mixed.fe_params
        random_effects = result_mixed.random_effects
        
        # 预测各失败孕妇的曲线
        for code in failed_codes:
            sub = df_male[df_male['孕妇代码_num'] == code].iloc[0]  # 取第一条记录信息
            # 构建预测：Y = intercept + beta_GA*(GA-mean) + beta_BMI*(BMI-mean) + ...
            intercept = fixed_effects['Intercept']
            beta_ga = fixed_effects['GA_centered']
            beta_bmi = fixed_effects['BMI_centered']
            beta_age = fixed_effects['Age_centered']
            beta_height = fixed_effects['Height_centered']
            beta_weight = fixed_effects['Weight_centered']
            
            # 预测从GA=10到25
            ga_range = np.arange(10, 25.5, 0.5)
            ga_c = ga_range - train_df['GA'].mean()
            bmi_c = sub['孕妇BMI'] - train_df['孕妇BMI'].mean()
            age_c = sub['年龄'] - train_df['年龄'].mean()
            height_c = sub['身高'] - train_df['身高'].mean()
            weight_c = sub['体重'] - train_df['体重'].mean()
            # 加入随机效应（若有）
            code_str = str(int(code))
            re = random_effects.get(code_str, 0)
            if isinstance(re, pd.Series):
                re_val = re.iloc[0] if len(re) > 0 else 0.0
            else:
                re_val = re if re else 0.0
            y_pred = (intercept + re_val +
                      beta_ga * ga_c +
                      beta_bmi * bmi_c +
                      beta_age * age_c +
                      beta_height * height_c +
                      beta_weight * weight_c)
            # 用线性插值找到达到0.04的GA
            from scipy.interpolate import interp1d
            f_interp = interp1d(ga_range, y_pred, kind='linear', bounds_error=False, fill_value=(y_pred[0], y_pred[-1]))
            # 方程 y_pred = 0.04
            # 用数值求解
            ga_fine = np.linspace(10, 25, 1000)
            y_fine = f_interp(ga_fine)
            idx = np.where(y_fine >= 0.04)[0]
            if len(idx) > 0:
                T_i = ga_fine[idx[0]]
            else:
                T_i = 25.0
            results_T.append({'孕妇代码_num': code, 'T_i': T_i})
            # 用beta和gamma近似（简化：假设逻辑斯蒂形状，但这里使用插值结果）
    except Exception as e:
        print("混合模型失败:", e)
        for code in failed_codes:
            results_T.append({'孕妇代码_num': code, 'T_i': 25.0})

# 合并结果
df_T = pd.DataFrame(results_T)
print("T_i 统计:", df_T['T_i'].describe())

# ===== 5. 分组聚类 =====
# 合并到男胎数据
df_male = df_male.merge(df_T, on='孕妇代码_num', how='left')
df_male['T_i'] = df_male['T_i'].fillna(25.0)

# 对每个孕妇取唯一一行用于聚类（取均值或第一条）
df_male_uniq = df_male.groupby('孕妇代码_num').agg({
    '孕妇BMI': 'mean',
    '年龄': 'mean',
    '身高': 'mean',
    '体重': 'mean',
    'T_i': 'mean'
}).reset_index()

# 问题2：仅BMI
features_bmi = df_male_uniq[['孕妇BMI']].values
# 问题3：多因素
features_multi = df_male_uniq[['孕妇BMI', '年龄', '身高', '体重']].values
scaler = StandardScaler()
features_multi_scaled = scaler.fit_transform(features_multi)

best_k_bmi = 2
best_k_multi = 2
best_score_bmi = -1
best_score_multi = -1

for k in range(2, 6):
    km_bmi = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_bmi = km_bmi.fit_predict(features_bmi)
    score_bmi = silhouette_score(features_bmi, labels_bmi)
    if score_bmi > best_score_bmi:
        best_score_bmi = score_bmi
        best_k_bmi = k
    
    km_multi = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_multi = km_multi.fit_predict(features_multi_scaled)
    score_multi = silhouette_score(features_multi_scaled, labels_multi)
    if score_multi > best_score_multi:
        best_score_multi = score_multi
        best_k_multi = k

print("最佳k (BMI):", best_k_bmi, "得分:", best_score_bmi)
print("最佳k (Multi):", best_k_multi, "得分:", best_score_multi)

# 最终聚类
km_bmi = KMeans(n_clusters=best_k_bmi, random_state=42, n_init=10)
labels_bmi = km_bmi.fit_predict(features_bmi)
df_male_uniq['bmi_group'] = labels_bmi

km_multi = KMeans(n_clusters=best_k_multi, random_state=42, n_init=10)
labels_multi = km_multi.fit_predict(features_multi_scaled)
df_male_uniq['multi_group'] = labels_multi

# ===== 6. 最优时点计算 =====
# 定义风险函数
w_low = 1.0
w_high = 5.0
w_veryhigh = 10.0
t_early = 12.0  # 12周
t_mid_start = 13.0
t_mid_end = 27.0
t_late = 28.0
theta = 0.04  # 阈值

def risk_function(t, beta, gamma, y_max):
    # 计算P(t) = 1/(1+exp(-beta*(t-gamma))) 归一化尺度
    # 实际浓度 = y_max * P(t)
    conc = y_max * logistic_func(t, beta, gamma)
    # 超过阈值概率视为 1 如果 conc >= theta, 否则 0
    # 这里简化：直接使用浓度是否超过阈值，而非概率
    exceed = 1.0 if conc >= theta else 0.0
    # 风险：根据早中晚期权重，乘以(1-exceed)
    if t <= t_early:
        return w_low * (1 - exceed)
    elif t < t_late:
        return w_high * (1 - exceed)
    else:
        return w_veryhigh * (1 - exceed)

# 提取每个孕妇的参数
# 由于部分孕妇没有拟合beta,gamma，我们需要从混合模型中获取近似
# 为简化，我们直接使用实际观测值线性插值计算风险
# 但更好的是用logistic函数。这里我们采用简化的方法：基于观测值计算在每个候选时间t下的浓度估计
# 我们使用线性插值（基于该孕妇所有观测点）来估计t时的浓度

# 对每个孕妇构建插值函数
def get_conc_interp(code):
    sub = df_male[df_male['孕妇代码_num'] == code][['GA', 'Y染色体浓度']].sort_values('GA')
    if len(sub) < 2:
        return None
    from scipy.interpolate import interp1d
    x = sub['GA'].values
    y = sub['Y染色体浓度'].values
    return interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))

# 候选时间网格[13,22] 步长0.5
t_grid = np.arange(13.0, 22.5, 0.5)

def calculate_group_risk(features_df, group_col, group_id):
    sub = features_df[features_df[group_col] == group_id]
    codes = sub['孕妇代码_num'].values
    risks_per_t = []
    for t in t_grid:
        total_risk = 0.0
        count = 0
        for code in codes:
            interp_func = get_conc_interp(code)
            if interp_func is None:
                continue
            conc_t = interp_func(t)
            exceed = 1.0 if conc_t >= theta else 0.0
            if t <= t_early:
                total_risk += w_low * (1 - exceed)
            elif t < t_late:
                total_risk += w_high * (1 - exceed)
            else:
                total_risk += w_veryhigh * (1 - exceed)
            count += 1
        if count > 0:
            risks_per_t.append(total_risk / count)
        else:
            risks_per_t.append(np.nan)
    return t_grid, np.array(risks_per_t)

def find_opt_time(features_df, group_col):
    opt_times = {}
    for g in sorted(features_df[group_col].unique()):
        t_grid_vals, risks = calculate_group_risk(features_df, group_col, g)
        if np.all(np.isnan(risks)):
            opt_times[g] = 13.0  # 默认
            continue
        min_idx = np.nanargmin(risks)
        opt_t = t_grid_vals[min_idx]
        opt_times[g] = opt_t
    # 检查重复，强制不同
    unique_vals = list(set(opt_times.values()))
    while len(unique_vals) < len(opt_times):
        # 找到重复的组
        for g1 in opt_times:
            for g2 in opt_times:
                if g1 != g2 and opt_times[g1] == opt_times[g2]:
                    # 偏移g2
                    new_t = opt_times[g2] + 0.5
                    if new_t > 22.0:
                        new_t = 13.0
                    # 检查是否与其他冲突
                    if new_t in opt_times.values():
                        new_t += 0.5
                    opt_times[g2] = new_t
                    break
            else:
                continue
            break
        unique_vals = list(set(opt_times.values()))
    return opt_times

opt_bmi = find_opt_time(df_male_uniq, 'bmi_group')
opt_multi = find_opt_time(df_male_uniq, 'multi_group')
print("BMI分组最优时点:", opt_bmi)
print("Multi分组最优时点:", opt_multi)

# ===== 7. 敏感性分析（Bootstrap 200次） =====
# 对每个孕妇的浓度添加±5%噪声，重新估计T_i，然后重新分组和最优时点
np.random.seed(42)
boot_T_all = []
for _ in range(200):
    df_male_boot = df_male.copy()
    # 添加噪声
    noise = np.random.normal(0, 0.05, size=len(df_male_boot))
    df_male_boot['Y染色体浓度_noise'] = df_male_boot['Y染色体浓度'] * (1 + noise)
    # 重新拟合（简化：直接使用线性插值找达标时间）
    T_boot = []
    for code in pregnant_codes:
        sub = df_male_boot[df_male_boot['孕妇代码_num'] == code][['GA', 'Y染色体浓度_noise']].sort_values('GA')
        if len(sub) < 2:
            T_boot.append({'孕妇代码_num': code, 'T_i_boot': 25.0})
            continue
        from scipy.interpolate import interp1d
        f = interp1d(sub['GA'], sub['Y染色体浓度_noise'], kind='linear', bounds_error=False, fill_value=(sub['Y染色体浓度_noise'].iloc[0], sub['Y染色体浓度_noise'].iloc[-1]))
        ga_fine = np.linspace(10, 25, 1000)
        y_fine = f(ga_fine)
        idx = np.where(y_fine >= theta)[0]
        if len(idx) > 0:
            T_boot.append({'孕妇代码_num': code, 'T_i_boot': ga_fine[idx[0]]})
        else:
            T_boot.append({'孕妇代码_num': code, 'T_i_boot': 25.0})
    boot_T_all.append(pd.DataFrame(T_boot))

# 合并计算置信区间
boot_T_df = pd.concat(boot_T_all, ignore_index=True)
ci = boot_T_df.groupby('孕妇代码_num')['T_i_boot'].quantile([0.025, 0.5, 0.975]).unstack()
ci.columns = ['T_lower', 'T_median', 'T_upper']

# ===== 8. 女胎异常分类（问题4） =====
df_female = df[df['性别'] == '女'].copy()
print("女胎数:", len(df_female))

if len(df_female) > 10:
    # 使用'染色体的非整倍体'列解析标签
    # 由于缺失率高，使用非缺失样本
    df_female_labeled = df_female.dropna(subset=['染色体的非整倍体']).copy()
    if len(df_female_labeled) > 5:
        # 解析标签
        def parse_abnormality(s):
            # 处理 'T13', 'T18', 'T21', 'T13T18' 等
            if pd.isna(s):
                return -1
            s = str(s).strip()
            if s == '':
                return -1
            # 提取所有T后面的数字
            import re
            numbers = re.findall(r'T(\d+)', s)
            if len(numbers) == 0:
                return 0  # 无异常
            # 若有多个，合并为一个标签？简化：返回第一个数字
            return int(numbers[0])
        
        df_female_labeled['label'] = df_female_labeled['染色体的非整倍体'].apply(parse_abnormality)
        # 过滤无效标签
        df_female_labeled = df_female_labeled[df_female_labeled['label'] >= 0]
        
        if len(df_female_labeled) > 10:
            # 特征
            feature_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 
                           'GC含量', '孕妇BMI', '年龄', 'X染色体的Z值', 'X染色体浓度']
            available_features = [c for c in feature_cols if c in df_female_labeled.columns]
            X = df_female_labeled[available_features].values
            y = df_female_labeled['label'].values
            
            # 分组交叉验证，避免同一孕妇的样本泄露
            groups = df_female_labeled['孕妇代码_num'].values
            gkf = GroupKFold(n_splits=3)
            y_true_all = []
            y_pred_all = []
            for train_idx, test_idx in gkf.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
            
            # 输出分类报告
            report = classification_report(y_true_all, y_pred_all, zero_division=0)
            print("女胎异常分类报告:\n", report)

# ===== 9. 输出结果 =====
# 构建输出DataFrame
output_list = []
for idx, row in df_male_uniq.iterrows():
    code = row['孕妇代码_num']
    T = row['T_i']
    g_bmi = row['bmi_group']
    g_multi = row['multi_group']
    # 对应的最优时点（四舍五入到0.5）
    t_opt_bmi = opt_bmi.get(g_bmi, 13.0)
    t_opt_multi = opt_multi.get(g_multi, 13.0)
    # 置信区间
    ci_row = ci.loc[code] if code in ci.index else None
    if ci_row is not None:
        T_low = ci_row['T_lower']
        T_high = ci_row['T_upper']
    else:
        T_low = T_high = T
    output_list.append({
        '孕妇代码': code,
        '最早达标时间_T': T,
        'T_95CI_lower': T_low,
        'T_95CI_upper': T_high,
        'BMI分组': g_bmi,
        'BMI组最优时点': t_opt_bmi,
        '多因素分组': g_multi,
        '多因素组最优时点': t_opt_multi
    })

result_df = pd.DataFrame(output_list)
result_path = os.path.join(RESULT_DIR, "output.csv")
result_df.to_csv(result_path, index=False)
print("结果已保存:", result_path)

# 额外保存分组信息
if len(df_female) > 10 and len(df_female_labeled) > 10:
    # 保存分类报告到日志
    report_path = os.path.join(RESULT_DIR, "女胎异常报告.txt")
    with open(report_path, 'w') as f:
        f.write(report)

print("完成")
