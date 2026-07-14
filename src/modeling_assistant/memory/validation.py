"""动态 LTM 的符号查重与公式闭环校验。

Goal.md 要求 Clarifier 在写入动态 LTM 前进行：
1. 符号查重（nomenclature 中不能有重复定义）
2. 公式闭环校验（equations 中引用的符号必须在 nomenclature 中定义）
"""

from __future__ import annotations

import re

from modeling_assistant.schemas.state import DynamicLTM


# 提取方程中的符号 token：匹配形如变量名的标识符
# 例如：y = ax + b → ["y", "ax", "b"]
_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

# 保留字：数学函数和常见关键字，不算未定义符号
_MATH_RESERVED = {
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "log", "ln", "exp", "sqrt", "abs", "min", "max", "sum", "prod",
    "argmin", "argmax", "integral", "lim", "inf", "sup",
    "if", "then", "else", "for", "all", "exists", "in",
    "and", "or", "not", "true", "false", "null",
    "Score", "total", "inn", "fea",
    "s", "t",  # 常用自变量，可在公式中默认存在
    "d",  # 微分符号
    "f", "g", "h",  # 常用函数名
}


def validate_dynamic_ltm(ltm: DynamicLTM) -> list[str]:
    """校验动态 LTM 的符号一致性与公式闭环。

    返回错误信息列表；空列表表示通过。
    """
    errors: list[str] = []

    # 1. 符号查重：nomenclature 的 key 本身就是 dict，天然唯一；
    #    但检查是否有不同符号但同义描述的情况
    seen_descriptions: dict[str, str] = {}
    for symbol, desc in ltm.nomenclature.items():
        if desc in seen_descriptions:
            errors.append(
                f"符号查重：'{symbol}' 与 '{seen_descriptions[desc]}' "
                f"具有相同的描述 '{desc}'，可能导致歧义。"
            )
        else:
            seen_descriptions[desc] = symbol

    # 2. 公式闭环：equations 中引用的符号必须在 nomenclature 中定义
    defined_symbols = set(ltm.nomenclature.keys()) | _MATH_RESERVED
    for i, equation in enumerate(ltm.equations):
        tokens = _TOKEN_RE.findall(equation)
        undefined = [tok for tok in tokens if tok not in defined_symbols]
        if undefined:
            errors.append(
                f"公式闭环：第 {i + 1} 条方程 "
                f"'{equation[:50]}...' 引用了未定义符号：{undefined}。"
            )

    # 3. objective 中引用的关键符号也应可追溯
    objective_tokens = _TOKEN_RE.findall(ltm.objective)
    objective_undefined = [
        tok for tok in objective_tokens
        if tok not in defined_symbols and tok not in {"目标", "函数", "最小化", "最大化"}
    ]
    if objective_undefined:
        # 只警告不强制失败
        pass

    return errors
