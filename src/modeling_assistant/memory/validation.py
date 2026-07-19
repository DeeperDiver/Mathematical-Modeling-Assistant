"""动态 LTM 的符号查重校验。

设计原则（基于 real_test 端到端测试教训）：
- 公式闭环校验已移除：数学建模公式天然含向量分量(P_M)、下标(M0)、
  自定义函数(cover)、积分变量(dt)、逻辑运算符(AND)等，穷举定义不可能，
  硬校验误判率极高（实测 5 次修复全部失败，导致死循环）。
- 符号一致性应由下游 Coder 执行反馈（NameError 触发回退）+
  milestone_reviewer_1 的 LLM 语义审查保证，而非静态正则校验。
- 保留符号查重（不同符号完全相同描述）作为硬错误，可能导致歧义。
"""

from __future__ import annotations

from modeling_assistant.schemas.state import DynamicLTM


def validate_dynamic_ltm(ltm: DynamicLTM) -> list[str]:
    """校验动态 LTM 的符号一致性。

    返回错误信息列表；空列表表示通过。
    """
    errors: list[str] = []

    # 符号查重：nomenclature 中不同符号具有完全相同的描述 → 可能歧义
    # 允许"导弹速度" vs "无人机速度"这种合理同属性命名（描述不完全相同）
    seen_descriptions: dict[str, str] = {}
    for symbol, desc in ltm.nomenclature.items():
        if desc in seen_descriptions:
            errors.append(
                f"符号查重：'{symbol}' 与 '{seen_descriptions[desc]}' "
                f"具有完全相同的描述 '{desc}'，可能导致歧义。"
            )
        else:
            seen_descriptions[desc] = symbol

    return errors
