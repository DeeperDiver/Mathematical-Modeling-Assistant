"""V11 三层防线第一层：纯机器提取"事实表"。

从题目原文用正则提取所有 (数值, 单位, 原文上下文)，
从数据附件样例值推断字符串列的解析代码建议。

这两类产物都是"机器事实"，不经过 LLM 改写，作为后续节点的真理基准：
- Clarifier 写入 dynamic_ltm 时，必须引用这些常量值（第二层校验）
- Coder 生成代码后，代码里的数值字面量必须与这些值匹配（第三层校验）
"""

from __future__ import annotations

import re
from typing import Any

from modeling_assistant.schemas.state import ColumnProfile, ProblemFact


# ─────────────────────────────────────────────────────────────────────
# V11.4：fact 语义类型分类
# ─────────────────────────────────────────────────────────────────────

# 强信号词：明确描述数据分布/合理范围的词（不含"范围/区间/介于"等弱词）
# 只有同时满足"强信号词 + 列名匹配"才标为 data_range
_DATA_RANGE_STRONG_KEYWORDS = (
    "正常范围",
    "参考区间",
    "合理范围",
    "含量范围",
    "浓度范围",
    "比例范围",
    "数据范围",
    "取值范围",
)

# 计数单位（不参与代码常量校验）
_COUNT_UNITS = frozenset({
    "倍", "次", "枚", "架", "个", "秒", "分", "时", "天", "周", "月", "年",
})


def classify_fact(
    fact: ProblemFact,
    columns: list[ColumnProfile] | None = None,
) -> str:
    """V11.4：识别 fact 的语义类型（physical_param / data_range / count）。

    双重判据避免误判真物理参数为 data_range：
    1. 计数单位（次/枚/架/个等）→ count
    2. 同时满足"强信号词 + 列名匹配"→ data_range
    3. 其他 → physical_param（保守默认，保留校验）

    示例：
    - "GC 含量正常范围 40%-60%" → 强信号词"正常范围" + 列名"GC含量" → data_range
    - "Y 染色体浓度达到 4%" → 无强信号词 → physical_param（保留校验）
    - "无人机速度范围 70~140 m/s" → "速度范围"不在强信号列表 → physical_param（保留校验）
    - "3 枚烟幕弹" → 单位"枚"是计数 → count
    """
    # 1. 计数单位
    if fact.unit in _COUNT_UNITS:
        return "count"

    # 2. data_range 双重判据：强信号词 + 列名匹配
    context = fact.context or ""
    # V11.4 修复：PDF 提取的中文文本可能在字间插入空格（如"范 围"拆开），
    # 需要先去空格再做匹配，否则"含量范围"无法匹配"含量范 围"。
    context_no_space = re.sub(r"\s+", "", context)
    has_strong_keyword = any(kw in context_no_space for kw in _DATA_RANGE_STRONG_KEYWORDS)
    if has_strong_keyword and columns:
        # 检查 context 里是否出现某个数据列名
        # 容忍空格差异：去掉所有空格后做子串匹配
        # 例如 context="GC 含量正常范围..." 去空格后="GC含量正常范围..."
        # 列名"GC含量" 即可匹配
        col_name_in_context = any(
            col.name and re.sub(r"\s+", "", col.name) in context_no_space
            for col in columns
            if col.name
        )
        if col_name_in_context:
            return "data_range"
        # 有强信号词但没匹配到列名 → 保守判为 physical_param
        return "physical_param"

    return "physical_param"


# ─────────────────────────────────────────────────────────────────────
# 数值常量提取
# ─────────────────────────────────────────────────────────────────────

# 匹配带单位的数值：整数或小数 + 常见单位
# 单位清单覆盖数学建模常见物理量
_NUM_UNIT_PATTERN = re.compile(
    r"("
    r"\d+\.?\d*"               # 数值：3 / 3.0 / 0.5 / 100
    r")\s*"
    r"("
    r"m/s²|m/s2|"
    r"m/s|km/s|cm/s|"
    r"m²|m2|"
    r"m³|m3|"
    r"km²|km2|"
    r"kg/m³|kg/m3|"
    r"mg/m³|mg/m3|"
    r"m|km|cm|mm|"
    r"kg|g|mg|t|"
    r"s|min|hour|h|ms|"
    r"Hz|kHz|MHz|"
    r"rad|°|度|"
    r"Pa|kPa|MPa|"
    r"N|kN|"
    r"J|kJ|"
    r"W|kW|MW|"
    r"V|kV|"
    r"A|mA|"
    r"℃|°C|°F|"
    r"L|mL|"
    r"mol|mmol|"
    r"px|"
    r"元|万|亿|"
    r"%|‰|"
    r"倍|次|枚|架|个|秒|分|时|天|周|月|年"
    r")"
)

# 常见量纲前缀上下文关键词（用于 LLM 标注 role_hint）
_CONTEXT_KEYWORDS = [
    "速度", "velocity", "speed",
    "加速度", "acceleration",
    "高度", "altitude", "height",
    "长度", "length", "distance",
    "半径", "radius",
    "直径", "diameter",
    "质量", "mass", "weight",
    "时间", "time", "duration",
    "频率", "frequency",
    "角度", "angle",
    "压力", "pressure",
    "力", "force",
    "能量", "energy",
    "功率", "power",
    "电压", "voltage",
    "电流", "current",
    "温度", "temperature",
    "体积", "volume",
    "物质的量", "mole",
    "面积", "area",
    "速度下沉", "下沉速度",
    "飞行速度",
    "投放", "起爆",
    "遮蔽", "有效",
    "间隔",
    "速度范围",
]


def extract_facts_from_problem(
    raw_problem: str,
    columns: list[ColumnProfile] | None = None,
) -> list[ProblemFact]:
    """从题目原文用正则提取所有数值常量。

    纯机器操作，不调用 LLM。提取 (value, unit, context) 三元组：
    - value: 浮点数
    - unit: 单位字符串
    - context: 提取该数值的原文片段（前后 20 字），便于 LLM 消歧

    V11.4：新增 columns 参数，用于 classify_fact 双重判据识别 data_range。
    若不传 columns，则所有 fact 默认为 physical_param（保守默认）。
    """
    if not raw_problem:
        return []

    facts: list[ProblemFact] = []
    seen_values: set[tuple[float, str]] = set()

    for match in _NUM_UNIT_PATTERN.finditer(raw_problem):
        value_str = match.group(1)
        unit = match.group(2)
        try:
            value = float(value_str)
        except ValueError:
            continue

        # 去重：相同 (value, unit) 只保留第一次出现
        key = (value, unit)
        if key in seen_values:
            continue
        seen_values.add(key)

        # 提取上下文：前后 20 字
        start = max(0, match.start() - 20)
        end = min(len(raw_problem), match.end() + 20)
        context = raw_problem[start:end].replace("\n", " ").strip()
        # 去除首尾可能残留的标点
        context = context.strip("，。、；：""''()[]{}")

        # 尝试用关键词推断 role_hint（机器不强制，仅作辅助）
        role_hint = ""
        for keyword in _CONTEXT_KEYWORDS:
            if keyword in context:
                role_hint = keyword
                break

        fact = ProblemFact(
            value=value,
            unit=unit,
            context=context,
            role_hint=role_hint,
        )
        # V11.4：填充 category 字段
        fact.category = classify_fact(fact, columns)
        facts.append(fact)

    return facts


# ─────────────────────────────────────────────────────────────────────
# 字符串列解析建议
# ─────────────────────────────────────────────────────────────────────

def infer_parse_hint(col: ColumnProfile) -> str:
    """根据列的样例值推断字符串解析代码建议。

    纯机器推断，不调用 LLM。覆盖常见格式：
    - '16W' → df['x'].str.replace('W','').astype(float)
    - '2023-01-15' → pd.to_datetime(df['x'])
    - '95%' → df['x'].str.rstrip('%').astype(float) / 100
    - '1,234' → df['x'].str.replace(',','').astype(float)
    """
    if not col.sample_values:
        return ""

    # 取第一个非空样例值
    samples = [str(v) for v in col.sample_values if v is not None and str(v).strip()]
    if not samples:
        return ""

    sample = samples[0]
    col_ref = f"df['{col.name}']"

    # 1. 数字+单位（如 '16W', '10kg', '3m'）
    # 匹配：整数/小数 + 字母单位
    m = re.match(r"^-?\d+\.?\d*([a-zA-Z°]+)$", sample)
    if m:
        unit = m.group(1)
        return f"{col_ref}.str.replace('{unit}', '', regex=False).astype(float)"

    # 2. 日期格式（如 '2023-01-15', '2023/01/15'）
    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", sample):
        return f"pd.to_datetime({col_ref})"

    # 3. 百分比（如 '95%', '0.5%'）
    if sample.endswith("%") and re.match(r"^-?\d+\.?\d*%$", sample):
        return f"{col_ref}.str.rstrip('%').astype(float) / 100"

    # 4. 千分位逗号（如 '1,234.5'）
    if "," in sample and re.match(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$", sample):
        return f"{col_ref}.str.replace(',', '', regex=False).astype(float)"

    # 5. 中文字符+数字（如 '第3组', '周12'）
    m = re.match(r"^[^\d]+(-?\d+\.?\d*)$", sample)
    if m:
        prefix = sample[:m.start(1)]
        return f"{col_ref}.str.replace('{prefix}', '', regex=False).astype(float)"

    # 6. 混合类型（如 'M1', 'FY1'）
    m = re.match(r"^[A-Za-z]+(\d+)$", sample)
    if m:
        return f"# 混合标识符，可能需要保留原值或提取数字: {col_ref}.str.extract(r'(\\d+)').astype(int)"

    return ""


def annotate_parse_hints(columns: list[ColumnProfile]) -> None:
    """原地填充每列的 parse_hint 字段。"""
    for col in columns:
        if col.dtype in ("int", "float", "bool", "datetime"):
            continue
        if col.parse_hint:
            continue  # 已有 hint 不覆盖
        col.parse_hint = infer_parse_hint(col)
