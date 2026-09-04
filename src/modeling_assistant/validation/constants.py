"""V11 三层防线第二层 + 第三层：常量一致性校验。

第二层：Clarifier 写入 dynamic_ltm 前校验 assumptions/equations 中的数值
是否与 problem_facts 一致，防止 LLM 把 3 m/s 记成 1 m/s。

第三层：Coder 生成代码后扫描代码中的数值字面量和列名，与 problem_facts
和 data_profile 比对，防止 Coder 手滑写错常量或臆造列名。
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Any

from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ColumnProfile,
    ControlState,
    DynamicLTM,
    GraphState,
    ProblemFact,
    StaticLTM,
)


# ─────────────────────────────────────────────────────────────────────
# 第二层：LTM 写入前校验
# ─────────────────────────────────────────────────────────────────────

# 从字符串中提取所有数字（用于扫描 assumptions/equations 中的数值）
_NUM_PATTERN = re.compile(r"-?\d+\.?\d*")

# 物理单位集合（用于判断 fact 是否是物理量）
_PHYSICAL_UNITS = frozenset({
    "m/s", "m/s²", "km/s", "cm/s", "m/s2",
    "m", "km", "cm", "mm",
    "s", "min", "h", "ms",
    "kg", "g", "mg", "t",
    "Hz", "kHz", "MHz",
    "rad", "°", "度",
    "Pa", "kPa", "MPa",
    "N", "kN",
    "J", "kJ",
    "W", "kW", "MW",
    "V", "kV",
    "A", "mA",
    "℃", "°C", "°F",
    "L", "mL",
    "mol", "mmol",
    "%", "‰",
    "m²", "m2", "km²", "km2", "m³", "m3",
})

# 在 LTM 文本中匹配 "<数值> <单位>" 模式（带空格或紧邻）
# 用于主防线：扫描 LTM 里带单位的数值，与 problem_facts 比对
_NUM_UNIT_IN_TEXT_PATTERN = re.compile(
    r"(-?\d+\.?\d*)\s*"
    r"(m/s²|m/s2|m/s|km/s|cm/s|"
    r"m²|m2|km²|km2|m³|m3|"
    r"m|km|cm|mm|"
    r"s|min|h|ms|"
    r"kg|g|mg|t|"
    r"Hz|kHz|MHz|"
    r"rad|°|度|"
    r"Pa|kPa|MPa|"
    r"N|kN|J|kJ|W|kW|MW|V|kV|A|mA|"
    r"℃|°C|°F|"
    r"L|mL|mol|mmol|"
    r"%|‰)"
)


def extract_numbers_from_text(text: str) -> list[float]:
    """从文本中提取所有数字，返回浮点数列表。"""
    if not text:
        return []
    numbers: list[float] = []
    for match in _NUM_PATTERN.finditer(text):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            continue
    return numbers


def _values_close(a: float, b: float, rel_tol: float = 0.05) -> bool:
    """判断两个数值是否在相对误差范围内相等。"""
    if abs(b) < 1e-9:
        return abs(a) < 1e-6
    return abs(a - b) < max(rel_tol * abs(b), 0.01)


def _fact_marked_irrelevant(fact: ProblemFact, dynamic_ltm: DynamicLTM) -> bool:
    """判断 Clarifier 是否明确记录该常量与当前小题无关。"""
    for key, decision in (dynamic_ltm.constant_relevance or {}).items():
        key_text = str(key)
        fact_tokens = (fact.context.strip(), f"{fact.value:g} {fact.unit}")
        if any(token and (token in key_text or key_text in token) for token in fact_tokens):
            return str(decision).strip().startswith("无关")
    return False


def check_ltm_against_facts(
    dynamic_ltm: DynamicLTM,
    static_ltm: StaticLTM,
) -> list[str]:
    """第二层校验：检查 dynamic_ltm 中的数值是否与 problem_facts 一致。

    返回校验问题列表。空列表表示通过。

    V11.1 修复：重写比对逻辑，消除跨类型冲突噪音。

    检查规则：
    1. 主防线（数值+单位模式匹配）：扫描 LTM 文本里所有 "<数值> <单位>" 模式，
       与 problem_facts 比对。如果 LTM 里出现 "1.0 m/s" 但 problem_facts 里
       只有 "3.0 m/s"，告警。这能直接拦住"3 m/s 被写成 1 m/s"的错误。
       不带单位的数字（如数学系数 0.5、迭代次数 100）不参与校验。

    2. 辅助防线（唯一值引用检查）：problem_facts 里值唯一的物理量
       必须在 LTM 中出现至少一次。如果 LLM 在压缩时丢了这个常量，告警。
       重复值（如多个 10.0 m）不强制要求，避免误报。
    """
    issues: list[str] = []
    facts = static_ltm.problem_facts
    if not facts:
        return issues

    # 合并 LTM 中的所有文本
    ltm_text_parts: list[str] = []
    ltm_text_parts.extend(dynamic_ltm.assumptions or [])
    ltm_text_parts.extend(dynamic_ltm.equations or [])
    ltm_text_parts.append(dynamic_ltm.objective or "")
    ltm_text_parts.append(dynamic_ltm.solution_outline or "")
    ltm_text = "\n".join(ltm_text_parts)

    # 按单位分组 problem_facts，便于主防线比对
    facts_by_unit: dict[str, list[ProblemFact]] = {}
    for fact in facts:
        if _fact_marked_irrelevant(fact, dynamic_ltm):
            continue
        if fact.unit not in _PHYSICAL_UNITS:
            continue
        # V11.4 修复：data_range 和 count 类 fact 跳过 LTM 数值比对
        # - data_range：数据列范围描述（如"GC 含量正常范围 40%-60%"），
        #   LLM 在 LTM 里可能写数据中观察到的具体值（如 88%），属于数据特性而非记错常量
        # - count：纯计数单位（如"3 枚"），LTM 里出现其他计数属于建模上下文，不该告警
        if fact.category in ("data_range", "count"):
            continue
        facts_by_unit.setdefault(fact.unit, []).append(fact)

    # ── 主防线：扫描 LTM 里的 "<数值> <单位>" 模式 ──
    # 只比对 LTM 里带单位的数值，不带单位的数字不参与校验
    for match in _NUM_UNIT_IN_TEXT_PATTERN.finditer(ltm_text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        unit = match.group(2)
        # 该单位在 problem_facts 里没有，不校验（可能是 LTM 自创的单位表述）
        if unit not in facts_by_unit:
            continue
        # 检查该 value 是否与 facts_by_unit[unit] 里的某个值匹配（允许 5% 误差）
        matching_facts = facts_by_unit[unit]
        if any(_values_close(value, f.value) for f in matching_facts):
            continue
        # 不匹配 → 告警
        expected_values = [f.value for f in matching_facts]
        issues.append(
            f"LTM 中出现 '{value} {unit}'，但题目常量为 {expected_values} {unit}"
            f"（可能 LLM 记错了数值）"
        )

    # ── 辅助防线：唯一值物理量必须被引用 ──
    # 统计每个 (value, unit) 在 problem_facts 里出现的次数
    unit_value_count: Counter[tuple[float, str]] = Counter()
    for fact in facts:
        if _fact_marked_irrelevant(fact, dynamic_ltm):
            continue
        if fact.unit not in _PHYSICAL_UNITS:
            continue
        # V11.4 修复：data_range 和 count 类 fact 跳过"必须被引用"检查
        if fact.category in ("data_range", "count"):
            continue
        unit_value_count[(fact.value, fact.unit)] += 1

    ltm_numbers = extract_numbers_from_text(ltm_text)
    for (value, unit), count in unit_value_count.items():
        # 只对"值唯一"的物理量强制要求 LTM 引用
        # 重复值（如两个 10.0 m）不强制，因为 LLM 可能只引用其中一个
        if count != 1:
            continue
        # 检查 LTM 里是否出现该值（允许 5% 误差）
        if not any(_values_close(n, value) for n in ltm_numbers):
            # 找到该 fact 的 context 用于告警
            matching_fact = next(
                (f for f in facts if f.value == value and f.unit == unit),
                None,
            )
            context = matching_fact.context[:50] if matching_fact else ""
            issues.append(
                f"关键常量未在 LTM 中引用：{value} {unit}"
                f"（原文：{context}）"
            )

    return issues


# ─────────────────────────────────────────────────────────────────────
# 第三层：Coder 代码校验
# ─────────────────────────────────────────────────────────────────────

def _extract_float_literals_from_ast(node: ast.AST) -> list[float]:
    """从 AST 中提取所有浮点字面量和整数字面量。"""
    numbers: list[float] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)):
            numbers.append(float(child.value))
    return numbers


def _extract_column_accesses_from_ast(node: ast.AST) -> list[str]:
    """从 AST 中提取 df['xxx'] 形式的列**读取**访问（不含写入新列）。

    V11.2 修复（Bug 2）：
    - 跳过 Assign.targets 中的 Subscript（写入新列，如 df['新列'] = ...），
      避免 Coder 自创中间列（'反射率平滑'、'权重' 等）被误判为臆造列名。
    - 扩大变量名识别：df/data/X/y 及其变体（df1, df_renamed, df_SiC 等），
      避免 Coder 用 df1_renamed['wn'] 时因变量名不在白名单而漏校验。
    - 识别 rename 后的列访问：保守策略，不校验 rename 目标列（新名），
      只校验 rename 源列（旧名）。
    """
    import re

    # 1. 收集所有 Assign.targets 中的 Subscript 节点 id（写入位置）
    #    df['新列'] = ... 中的 df['新列'] 是写入，应跳过
    write_subscript_ids: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Subscript):
                        write_subscript_ids.add(id(sub))

    # 2. 收集所有 rename 目标列名（保守：不校验 rename 后的新名）
    #    df.rename(columns={'旧名': '新名'}) 中的 '新名' 不校验，'旧名' 仍校验
    rename_target_names = _extract_rename_targets_from_ast(node)

    # 3. 识别 df/data/X/y 及常见变体（df1, df2, df_xxx, data_xxx 等）
    df_var_pattern = re.compile(r"^(df|data|X|y)\w*$", re.IGNORECASE)

    columns: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            # 跳过写入位置
            if id(child) in write_subscript_ids:
                continue
            value = child.value
            slice_node = child.slice
            if isinstance(value, ast.Name) and df_var_pattern.match(value.id):
                col_name: str | None = None
                if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                    col_name = slice_node.value
                # Python 3.9+ 可能直接是 Constant
                elif hasattr(slice_node, "value") and isinstance(getattr(slice_node, "value", None), str):
                    col_name = slice_node.value
                if col_name is not None and col_name not in rename_target_names:
                    columns.append(col_name)
    return columns


def _extract_rename_targets_from_ast(node: ast.AST) -> set[str]:
    """V11.4：从 AST 中提取 df.rename(columns={'旧名': '新名'}) 调用的目标列名。

    覆盖两种调用形式：
    - df = df.rename(columns={'旧名': '新名'})  # 赋值
    - df.rename(columns={'旧名': '新名'}, inplace=True)  # inplace

    返回所有 '新名'（即 rename 后的新列名）。
    """
    rename_target_names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            # df.rename(...) 形式（不区分赋值/inplace）
            if isinstance(func, ast.Attribute) and func.attr == "rename":
                for kw in child.keywords:
                    if kw.arg == "columns" and isinstance(kw.value, ast.Dict):
                        # values 是新名列表
                        for v in kw.value.values:
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                rename_target_names.add(v.value)
    return rename_target_names


def check_code_against_facts(
    code: str,
    static_ltm: StaticLTM,
    artifacts: ArtifactBundle | None = None,
    dynamic_ltm: DynamicLTM | None = None,
) -> list[str]:
    """第三层校验：扫描 Coder 代码中的数值和列名，与 problem_facts 比对。

    返回校验问题列表。空列表表示通过。

    检查规则：
    1. 关键常量缺失：problem_facts 中的核心物理量（如 3.0 m/s）必须以字面量
       形式出现在代码里。如果出现缺失，Coder 可能把 3.0 写成了 1.0。

    2. 列名校验：代码里访问的 df['xxx'] 必须在 data_profile.columns 中存在。
       如果出现不存在的列名，Coder 可能臆造了列名。
    """
    issues: list[str] = []
    if not code:
        return ["代码为空，无法校验。"]

    # 解析 AST
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"代码语法错误，无法进行常量校验：{exc}"]

    # 1. 数值字面量提取
    code_numbers = _extract_float_literals_from_ast(tree)

    # 关键常量覆盖：检查 problem_facts 中的物理量是否出现在代码里
    facts = static_ltm.problem_facts
    if facts:
        for fact in facts:
            if dynamic_ltm is not None and _fact_marked_irrelevant(fact, dynamic_ltm):
                continue
            if fact.unit not in _PHYSICAL_UNITS:
                continue
            # V11.4 修复：data_range 和 count 类 fact 跳过字面量检查
            # - data_range：数据列范围描述（如"GC 含量正常范围 40%-60%"），
            #   是数据筛选阈值而非建模参数，代码可不写字面量
            # - count：纯计数单位（如"3 枚"），不参与代码常量校验
            if fact.category in ("data_range", "count"):
                continue
            # V11.3 修复（百分比等价）：百分比/千分比单位接受小数等价值
            # 题目说"4%"，LLM 合理地写成 0.04（科学计算标准写法），应通过校验
            acceptable_values = {fact.value}
            if fact.unit == "%":
                acceptable_values.add(fact.value / 100.0)
            elif fact.unit == "‰":
                acceptable_values.add(fact.value / 1000.0)

            found = any(
                _values_close(n, v)
                for n in code_numbers
                for v in acceptable_values
            )
            if not found:
                issues.append(
                    f"代码常量缺失：题目要求 {fact.value} {fact.unit}"
                    f"（原文：{fact.context[:50]}），但代码中未找到该数值字面量"
                    f"（已接受小数等价值，如 {fact.value}/100={fact.value/100.0:g}）。"
                )

    # 2. 列名校验
    if static_ltm.data_profile and static_ltm.data_profile.columns:
        # V11.3 修复（派生列识别）：收集代码中通过 df['新列'] = ... 创建的派生列名
        # 这些列虽不在数据画像里，但代码前面已创建，读取是合法的
        created_columns: set[str] = set()
        for child in ast.walk(tree):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Subscript):
                        slice_node = target.slice
                        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                            created_columns.add(slice_node.value)
        # V11.4 修复：valid_columns 纳入 rename 目标列
        # 之前只在读取侧跳过 rename 目标列，但没主动加入 valid_columns，
        # 导致 LLM 写 df.rename(columns={'旧名': '新名'}) 后访问 df['新名'] 仍被告警。
        # 现在主动把 rename 目标列加入合法列集合，逻辑更一致。
        rename_target_names = _extract_rename_targets_from_ast(tree)
        # 有效列 = 数据画像列 + 代码创建的派生列 + rename 目标列
        valid_columns = (
            {col.name for col in static_ltm.data_profile.columns}
            | created_columns
            | rename_target_names
        )
        accessed_columns = _extract_column_accesses_from_ast(tree)
        for col_name in accessed_columns:
            if col_name not in valid_columns:
                issues.append(
                    f"代码访问的列名 '{col_name}' 不在数据画像中，也不是代码中创建的派生列或 rename 目标列。"
                    f"有效列名：{sorted(valid_columns)[:10]}"
                )

    return issues
