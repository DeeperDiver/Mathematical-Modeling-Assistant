"""假设标签约定（V20）：【全文】/【问题N】放置标签 + 【关键】关键假设标签。

纯文本约定，不改动 assumptions 的 list[str] schema。本模块提供：
- 标签常量与正则，供 Prompt 规则、HITL 载荷分组、paper_check 机械检查复用；
- classify_assumptions() 按标签分组假设，供架构 HITL 展示。

标签规则：
- 每条假设以放置标签开头：【全文】（全文通用前提）或【问题N】
  （当前小题技术设定，N = 小题编号；单题模式 N=1）；
- 关键假设在放置标签后追加【关键】（如【问题3】【关键】），
  并保留「依据/风险/可验证性」标注；
- 历史遗留的【关键假设】前缀仅用于识别旧文本，不作为新约定。
"""

from __future__ import annotations

import re

# 放置标签：全文通用前提（可进入 3_assumptions.tex）
TAG_FULL = "【全文】"
# 关键假设标签：追加在放置标签之后
TAG_CRITICAL = "【关键】"
# 历史遗留的关键假设前缀（仅识别，不用于新输出）
TAG_CRITICAL_LEGACY = "【关键假设】"

# 【问题N】标签（宽容【问题 2】这类空白）
_QUESTION_TAG_RE = re.compile(r"【\s*问题\s*(\d+)\s*】")
# 关键标签（含历史前缀）
_CRITICAL_RE = re.compile(r"【\s*关键(?:假设)?\s*】")

# classify_assumptions() 的分组键
GROUP_FULL = "full"  # 【全文】假设
GROUP_QUESTION = "question"  # 【问题N】假设
GROUP_CRITICAL = "critical"  # 【关键】假设（与 full/question 有重叠）
GROUP_UNLABELED = "unlabeled"  # 无放置标签的假设（兜底，提示补标）


def has_question_tag(text: str) -> bool:
    """是否带【问题N】标签。"""
    return bool(_QUESTION_TAG_RE.search(text or ""))


def has_critical_tag(text: str) -> bool:
    """是否带【关键】或历史【关键假设】标签。"""
    return bool(_CRITICAL_RE.search(text or ""))


def question_index(text: str) -> int | None:
    """返回【问题N】的 N；无标签返回 None。"""
    match = _QUESTION_TAG_RE.search(text or "")
    return int(match.group(1)) if match else None


def classify_assumptions(assumptions: list[str]) -> dict[str, list[str]]:
    """按标签分组假设，供架构 HITL 展示。

    分组：
    - full：带【全文】的假设；
    - question：带【问题N】的假设；
    - critical：带【关键】的假设（与 full/question 有重叠，交叉项两处都列出）；
    - unlabeled：无放置标签的假设（提醒 Clarifier 补标）。
    """
    full: list[str] = []
    question: list[str] = []
    critical: list[str] = []
    unlabeled: list[str] = []
    for item in assumptions or []:
        text = str(item)
        if has_question_tag(text):
            question.append(text)
        elif TAG_FULL in text:
            full.append(text)
        else:
            unlabeled.append(text)
        if has_critical_tag(text):
            critical.append(text)
    return {
        GROUP_FULL: full,
        GROUP_QUESTION: question,
        GROUP_CRITICAL: critical,
        GROUP_UNLABELED: unlabeled,
    }
