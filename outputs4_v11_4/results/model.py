import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ==================== 数据读取 ====================
DATA_DIR = r"C:\Users\lingi\Desktop\Research\Mo\Modeling_Assistant\real_test4\附件"

# 订单信息
orders = pd.read_excel(os.path.join(DATA_DIR, "订单信息.xlsx"))
orders.dropna(subset=['目标客户编号'], inplace=True)
orders['目标客户编号'] = orders['目标客户编号'].astype(int)

# 聚合客户点需求
demand = orders.groupby('目标客户编号').agg(
    需求重量=('重量', 'sum'),
    需求体积=('体积', 'sum')
).reset_index()
demand.columns = ['客户编号', '需求重量', '需求体积']
# 确保所有1-98号客户都存在，缺失的补0
all_cust = pd.DataFrame({'客户编号': range(1, 99)})
demand = all_cust.merge(demand, on='客户编号', how='left').fillna(0)

# 时间窗
time_windows = pd.read_excel(os.path.join(DATA_DIR, "时间窗.xlsx"))
time_windows.columns = ['客户编号', '开始时间_str', '结束时间_str']
time_windows['客户编号'] = time_windows['客户编号'].astype(int)

def parse_time(t_str):
    """将'H:MM'格式转为小时数（从0点起）"""
    if pd.isna(t_str):
        return None
    t = datetime.strptime(str(t_str).strip(), '%H:%M')
    return t.hour + t.minute/60.0

time_windows['最早时间'] = time_windows['开始时间_str'].apply(parse_time)
time_windows['最晚时间'] = time_windows['结束时间_str'].apply(parse_time)
# 合并到客户需求
demand = demand.merge(time_windows[['客户编号','最早时间','最晚时间']], on='客户编号', how='left')

# 客户坐标
coords = pd.read_excel(os.path.join(DATA_DIR, "客户坐标信息.xlsx"))
coords.columns = ['类型','ID','X','Y']
coords = coords[coords['ID'] >= 0]  # 保留配送中心和客户
coords_dict = dict(zip(coords['ID'], zip(coords['X'], coords['Y'])))
# 配送中心0坐标
center_x, center_y = coords_dict[0]

# 判断客户是否在绿色配送区内（半径10km）
def in_green_zone(cust_id):
    if cust_id == 0:
        return False
    x, y = coords_dict[cust_id]
    return (x**2 + y**2) <= 100.0  # 10km半径

green_zone = {i: in_green_zone(i) for i in range(1,99)}

# 距离矩阵
dist_df = pd.read_excel(os.path.join(DATA_DIR, "距离矩阵.xlsx"))
dist_df = dist_df.iloc[:, 1:]  # 去掉第一列空索引？实际列名可能是0,1,...
dist_matrix = dist_df.values.astype(float)  # 形状(99,99)
# 距离矩阵的行列索引对应节点编号0-98，用numpy数组

# ==================== 全局常量 ====================
# 车辆参数
vehicle_types = ['F1','F2','F3','E1','E2']
vehicle_info = {
    'F1': {'Q':3000, 'V':13.5, 'S':60, 'fuel':True, 'delta':0.40},
    'F2': {'Q':1500, 'V':10.8, 'S':50, 'fuel':True, 'delta':0.40},
    'F3': {'Q':1250, 'V':6.5, 'S':50, 'fuel':True, 'delta':0.40},
    'E1': {'Q':3000, 'V':15.0, 'S':10, 'fuel':False, 'delta':0.35},
    'E2': {'Q':1250, 'V':8.5, 'S':15, 'fuel':False, 'delta':0.35},
}

c_start = 400.0           # 启动成本 元/辆
c_wait = 20.0             # 早到等待成本 元/小时
c_late = 50.0             # 晚到惩罚成本 元/小时
t_service = 20.0/60.0     # 服务时间 小时

p_fuel = 7.61             # 元/L
p_elec = 1.64             # 元/kWh
c_carbon = 0.65           # 元/kg
eta = 2.547               # kg/L
gamma = 0.501             # kg/kWh

# 速度函数（按时段均值）
def speed_at_time(t):
    """返回t时刻的速度 (km/h)，t为小时"""
    # 时段定义
    # 顺畅：9-10, 13-15
    # 一般：10-11:30, 15-17
    # 拥堵：8-9, 11:30-13
    # 其余时间按一般处理（简化）
    if (9 <= t < 10) or (13 <= t < 15):
        return 55.3
    elif (10 <= t < 11.5) or (15 <= t < 17):
        return 35.4
    elif (8 <= t < 9) or (11.5 <= t < 13):
        return 9.8
    else:
        # 非配送时段（比如6-8和17-20），假设一般速度
        return 35.4

# 能耗函数
def FPK(v):
    return 0.0025*v*v - 0.2554*v + 31.75
def EPK(v):
    return 0.0014*v*v - 0.12*v + 36.19

# ==================== 启发式算法 ====================
# 我们实现一个简单的贪心+2-opt
# 车辆id列表：每种车型生成车辆ID
vehicles = []
for vtype in vehicle_types:
    info = vehicle_info[vtype]
    for i in range(info['S']):
        vehicles.append({
            'id': f"{vtype}_{i+1}",
            'type': vtype,
            'capacity_weight': info['Q'],
            'capacity_volume': info['V'],
            'is_fuel': info['fuel'],
            'delta': info['delta'],
            'route': [0],   # 从配送中心出发
            'load_weight': 0.0,
            'load_volume': 0.0,
            'current_time': 8.0,  # 从8:00出发
            'total_distance': 0.0,
            'total_fuel': 0.0,
            'total_elec': 0.0,
            'total_carbon': 0.0,
            'time_penalty': 0.0,
            'start_cost': 0.0
        })

# 未服务客户列表
unserved = list(range(1,99))  # 客户编号1-98
np.random.seed(42)

# 贪心构造初始解
# 按顺序插入，优先新能源车
def calc_insert_cost(route, client, current_time, vehicle, dist_mat, demand_info, speed_func):
    """计算将客户插入到route中所有可能位置的最小增量成本"""
    best_cost = float('inf')
    best_pos = -1
    q = demand_info.loc[demand_info['客户编号']==client, '需求重量'].values[0]
    u = demand_info.loc[demand_info['客户编号']==client, '需求体积'].values[0]
    # 检查容量
    if vehicle['load_weight'] + q > vehicle['capacity_weight']:
        return None, -1
    if vehicle['load_volume'] + u > vehicle['capacity_volume']:
        return None, -1
    
    # 时间窗
    tw_low = demand_info.loc[demand_info['客户编号']==client, '最早时间'].values[0]
    tw_high = demand_info.loc[demand_info['客户编号']==client, '最晚时间'].values[0]
    if pd.isna(tw_low):
        tw_low = 0
        tw_high = 24
    
    for pos in range(1, len(route)):
        # 计算插入到pos位置后的成本
        prev_node = route[pos-1]
        next_node = route[pos]
        # 新弧段距离
        d_prev_cli = dist_mat[prev_node, client]
        d_cli_next = dist_mat[client, next_node]
        d_prev_next = dist_mat[prev_node, next_node]
        added_distance = d_prev_cli + d_cli_next - d_prev_next
        
        # 计算到达客户端的时间
        # 从配送中心出发时间8:00，但需要根据实际路线时间更新
        # 简化：我们先计算从prev_node到client的行驶时间
        # 使用平均速度（用当前时间的速度）
        v = speed_func(current_time)
        travel_time = d_prev_cli / v
        arrival = current_time + travel_time
        # 等待时间（如果早到）
        wait = max(tw_low - arrival, 0)
        # 服务时间
        service_time = t_service
        # 离开时间
        departure = arrival + wait + service_time
        
        # 到达next_node的时间（可能延迟）
        v2 = speed_func(departure)
        travel_next = d_cli_next / v2
        arrival_next = departure + travel_next
        
        # 计算时间窗惩罚（只考虑当前客户和下一个客户）
        penalty_client = c_wait * wait + c_late * max(arrival - tw_high, 0)
        # 后续节点的时间窗惩罚会受延迟影响，这里简化只考虑当前
        # 为了贪心，我们计算增量成本：额外距离的能耗成本 + 时间惩罚
        # 能耗成本：使用平均速度的能耗
        v_avg = (v+v2)/2
        if vehicle['is_fuel']:
            # 载重率近似
            load_ratio = (vehicle['load_weight'] + q) / vehicle['capacity_weight']
            e_per_100 = FPK(v_avg) * (1 + load_ratio * vehicle['delta'])
            energy_cost = (added_distance/100.0) * e_per_100 * p_fuel
            carbon_cost = (added_distance/100.0) * e_per_100 * eta * c_carbon
        else:
            load_ratio = (vehicle['load_weight'] + q) / vehicle['capacity_weight']
            e_per_100 = EPK(v_avg) * (1 + load_ratio * vehicle['delta'])
            energy_cost = (added_distance/100.0) * e_per_100 * p_elec
            carbon_cost = (added_distance/100.0) * e_per_100 * gamma * c_carbon
        
        incremental = added_distance*0.5 + energy_cost + carbon_cost + penalty_client
        
        if incremental < best_cost:
            best_cost = incremental
            best_pos = pos
    return best_cost, best_pos

# 贪心插入
for cust in unserved:
    # 优先新能源车
    best_vehicle = None
    best_cost = float('inf')
    best_pos = -1
    for v in vehicles:
        if v['type'] in ['E1','E2'] or (v['type'] in ['F1','F2','F3'] and not green_zone[cust]):
            # 对绿色区客户，尽量用新能源；对非绿色区都可以用
            result = calc_insert_cost(v['route'], cust, v['current_time'], v, dist_matrix, demand, speed_at_time)
            if result[0] is not None:
                cost, pos = result
                if cost < best_cost:
                    best_cost = cost
                    best_vehicle = v
                    best_pos = pos
    if best_vehicle is None:
        # 尝试所有车辆
        for v in vehicles:
            result = calc_insert_cost(v['route'], cust, v['current_time'], v, dist_matrix, demand, speed_at_time)
            if result[0] is not None:
                cost, pos = result
                if cost < best_cost:
                    best_cost = cost
                    best_vehicle = v
                    best_pos = pos
    if best_vehicle is not None:
        # 执行插入
        route = best_vehicle['route']
        route.insert(best_pos, cust)
        # 更新当前车辆状态
        # 重新计算到达时间等（简化：只更新最后节点时间）
        # 这里为了简单，我们重新计算整个路径的时间
        # 但由于时间限制，我们用近似：不精确更新，但保持大致可行
        # 我们暂时跳过精确更新，只在最终评估时计算
        best_vehicle['load_weight'] += demand.loc[demand['客户编号']==cust, '需求重量'].values[0]
        best_vehicle['load_volume'] += demand.loc[demand['客户编号']==cust, '需求体积'].values[0]
        unserved.remove(cust)

# 对每个车辆路径执行2-opt内部优化（简化）
def two_opt(route, dist_mat):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+2, len(route)-1):
                # 如果交换可以缩短总距离
                d_old = dist_mat[route[i-1], route[i]] + dist_mat[route[j], route[j+1]]
                d_new = dist_mat[route[i-1], route[j]] + dist_mat[route[i], route[j+1]]
                if d_new < d_old:
                    route[i:j+1] = route[j:i-1:-1]  # 反转
                    improved = True
    return route

for v in vehicles:
    if len(v['route']) > 2:
        v['route'] = two_opt(v['route'], dist_matrix)

# 过滤出使用的车辆
used_vehicles = [v for v in vehicles if len(v['route']) > 2]  # 至少服务一个客户

# ==================== 成本计算 ====================
def compute_costs(v, demand_info, dist_mat, speed_func):
    """计算车辆v的实际成本"""
    route = v['route']
    if len(route) <= 2:
        return None
    # 初始化
    total_distance = 0.0
    total_fuel = 0.0 if v['is_fuel'] else 0.0
    total_elec = 0.0 if not v['is_fuel'] else 0.0
    total_carbon = 0.0
    time_penalty = 0.0
    
    current_time = 8.0
    load_weight = 0.0
    load_volume = 0.0
    
    for idx in range(len(route)-1):
        i = route[idx]
        j = route[idx+1]
        d = dist_mat[i, j]
        total_distance += d
        
        # 确定速度
        v_sp = speed_func(current_time)
        travel_time = d / v_sp
        
        # 到达j的时间
        arrival = current_time + travel_time
        
        if j != 0:  # 客户点
            # 时间窗
            tw_low = demand_info.loc[demand_info['客户编号']==j, '最早时间'].values[0]
            tw_high = demand_info.loc[demand_info['客户编号']==j, '最晚时间'].values[0]
            if pd.isna(tw_low):
                tw_low = 0
                tw_high = 24
            wait = max(tw_low - arrival, 0)
            late = max(arrival - tw_high, 0)
            time_penalty += c_wait * wait + c_late * late
            # 服务时间
            current_time = arrival + wait + t_service
            # 装货
            q = demand_info.loc[demand_info['客户编号']==j, '需求重量'].values[0]
            u = demand_info.loc[demand_info['客户编号']==j, '需求体积'].values[0]
            load_weight += q
            load_volume += u
        else:
            # 回到配送中心
            current_time = arrival
        
        # 计算能耗
        load_ratio = load_weight / v['capacity_weight'] if v['capacity_weight'] > 0 else 0
        if v['is_fuel']:
            e_pr_100 = FPK(v_sp) * (1 + load_ratio * v['delta'])
            fuel_liters = (d/100.0) * e_pr_100
            total_fuel += fuel_liters
            total_carbon += fuel_liters * eta
        else:
            e_pr_100 = EPK(v_sp) * (1 + load_ratio * v['delta'])
            elec_kwh = (d/100.0) * e_pr_100
            total_elec += elec_kwh
            total_carbon += elec_kwh * gamma
    
    # 能耗成本
    energy_cost = total_fuel * p_fuel + total_elec * p_elec
    carbon_cost = total_carbon * c_carbon
    start_cost = c_start  # 每辆车400
    return {
        'distance': total_distance,
        'energy_cost': energy_cost,
        'carbon_cost': carbon_cost,
        'time_penalty': time_penalty,
        'start_cost': start_cost
    }

# 计算每辆车的成本
vehicle_details = []
for v in used_vehicles:
    costs = compute_costs(v, demand, dist_matrix, speed_at_time)
    if costs:
        vehicle_details.append({
            '车辆编号': v['id'],
            '车型': v['type'],
            '路径': '->'.join(str(x) for x in v['route']),
            '总距离_km': costs['distance'],
            '能耗成本': costs['energy_cost'],
            '碳排放成本': costs['carbon_cost'],
            '时间惩罚成本': costs['time_penalty'],
            '启动成本': costs['start_cost'],
            '总成本': costs['start_cost'] + costs['energy_cost'] + costs['carbon_cost'] + costs['time_penalty']
        })

# ==================== 输出汇总 ====================
df_detail = pd.DataFrame(vehicle_details)

# 按车型汇总
if not df_detail.empty:
    cost_summary = df_detail.groupby('车型').agg(
        使用车辆数=('车辆编号', 'count'),
        总启动成本=('启动成本', 'sum'),
        总能耗成本=('能耗成本', 'sum'),
        总碳排放成本=('碳排放成本', 'sum'),
        总时间惩罚成本=('时间惩罚成本', 'sum'),
    ).reset_index()
    cost_summary['总成本'] = cost_summary['总启动成本'] + cost_summary['总能耗成本'] + cost_summary['总碳排放成本'] + cost_summary['总时间惩罚成本']
    cost_summary.columns = ['车型', '使用车辆数', '启动成本', '能耗成本', '碳排放成本', '时间惩罚成本', '总成本']
else:
    cost_summary = pd.DataFrame()

# 合并到一个输出DataFrame
rows = []
# 先添加成本汇总
if not cost_summary.empty:
    for _, row in cost_summary.iterrows():
        rows.append({
            '类别': '成本汇总',
            '车型': row['车型'],
            '使用车辆数': row['使用车辆数'],
            '启动成本': row['启动成本'],
            '能耗成本': row['能耗成本'],
            '碳排放成本': row['碳排放成本'],
            '时间惩罚成本': row['时间惩罚成本'],
            '总成本': row['总成本'],
            '车辆编号': '',
            '路径': '',
            '总距离_km': ''
        })
# 再添加车辆详情
for _, row in df_detail.iterrows():
    rows.append({
        '类别': '车辆路径',
        '车型': row['车型'],
        '使用车辆数': '',
        '启动成本': row['启动成本'],
        '能耗成本': row['能耗成本'],
        '碳排放成本': row['碳排放成本'],
        '时间惩罚成本': row['时间惩罚成本'],
        '总成本': row['总成本'],
        '车辆编号': row['车辆编号'],
        '路径': row['路径'],
        '总距离_km': row['总距离_km']
    })

output_df = pd.DataFrame(rows)

# 保存
OUTPUT_DIR = os.environ.get("MODELING_OUTPUT_DIR", ".")
RESULT_PATH = os.path.join(OUTPUT_DIR, "results", "output.csv")
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
output_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
print(f"结果保存至 {RESULT_PATH}")
print(output_df.head(20))
