"""方法知识库：从数学建模规范（method norms）按节点/题型切片，供 prompt 注入。

P0 优化：把 MathModelAgent 的 `math_modeling_norms.md` 规范知识库引入 Modeling_Assistant。

设计原则（与 Exemplar Learning System 一致）：
- 单一知识源：规范原文位于 `references/math_modeling_norms.md`，本模块运行时解析，
  不硬编码大段文本；规范更新后重启即生效。
- 按节点切片：每个下游节点只拿到与自身职责相关的小节（选型/假设/编码/图表）。
- 按题型切片：Mathematician/Realist/Coder 额外拿到当前题型（optimization/physics/
  forecasting/evaluation/data_mining）的专属指南与防错。
- 优雅降级：文件缺失或解析失败时返回空知识，prompt 渲染与旧行为完全一致。

本模块只影响「怎么建模/怎么写代码」的领域判断，不改变图结构、路由与验证逻辑。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 规范知识库路径（相对本模块：src/modeling_assistant/references/）
_NORMS_PATH = Path(__file__).resolve().parents[1] / "references" / "math_modeling_norms.md"

# `## 小节标题` 解析正则（保留 `###` 子标题在节内）
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


# ── 题型 → 该题型专属指南小节 ──────────────────────────────────────
# 与 memory/exemplar_search.PROBLEM_TYPE_KEYWORDS 的 5 类题型一一对应。
_TYPE_GUIDE_SECTIONS: dict[str, list[str]] = {
    "optimization": ["优化类模型详细指南"],
    "physics": ["机理/动力学类模型详细指南"],
    "forecasting": ["预测类模型详细指南"],
    "evaluation": ["评价类模型详细指南"],
    "data_mining": ["统计分析与机器学习"],
}

# ── 节点 → 通用小节（与题型无关的领域规范）─────────────────────────
_NODE_SECTIONS: dict[str, list[str]] = {
    "model_selection": ["模型大分类与选型速查", "题型防错速查"],
    "assumptions": ["假设与模型建立"],
    "coding": ["代码实现与结果", "编码阶段常见错误"],
    "chart": ["图表与可视化"],
}

# 无题型专属指南时的兜底小节
_FALLBACK_TYPE_SECTIONS = ["题型防错速查"]

# 解析结果缓存（模块级，避免每次渲染都重读文件）
_sections_cache: dict[str, str] | None = None


def parse_sections(text: str) -> dict[str, str]:
    """把规范 Markdown 按 `## ` 二级标题切分为 {标题: 正文}。"""
    sections: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = _SECTION_RE.match(line)
        if match:
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = match.group(1).strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def load_norm_sections() -> dict[str, str]:
    """读取规范文件并按小节解析；文件缺失/解析失败时返回空 dict。"""
    global _sections_cache
    if _sections_cache is None:
        try:
            text = _NORMS_PATH.read_text(encoding="utf-8")
            _sections_cache = parse_sections(text)
            logger.info("方法知识库加载成功：%d 个小节（%s）", len(_sections_cache), _NORMS_PATH)
        except Exception as exc:
            logger.warning("方法知识库加载失败（将降级为空知识）: %s", exc)
            _sections_cache = {}
    return _sections_cache


def _join_sections(sections: dict[str, str], names: list[str]) -> str:
    """按顺序拼接指定小节；缺失的小节跳过。"""
    parts = [sections[name] for name in names if sections.get(name)]
    return "\n\n".join(parts).strip()


def get_node_knowledge(node: str) -> str:
    """返回指定节点（model_selection / assumptions / coding / chart）的通用规范文本。"""
    sections = load_norm_sections()
    return _join_sections(sections, _NODE_SECTIONS.get(node, []))


def get_type_knowledge(problem_type: str) -> str:
    """返回当前题型的专属指南与防错文本；无专属指南时回退到通用题型防错速查。"""
    sections = load_norm_sections()
    names = _TYPE_GUIDE_SECTIONS.get(problem_type, [])
    text = _join_sections(sections, names)
    if not text:
        text = _join_sections(sections, _FALLBACK_TYPE_SECTIONS)
    return text


def build_knowledge_payload(problem_type: str) -> dict[str, str]:
    """组装 prompt 注入用的方法知识包。

    返回 dict 包含：
    - method_knowledge_active: "true"/"false"
    - problem_type: 判定出的题型（unknown 表示未能判定）
    - model_selection_knowledge: 选型决策树 + 通用题型防错（→ Mathematician）
    - type_knowledge: 当前题型专属指南与防错（→ Mathematician / Realist / Coder）
    - assumption_knowledge: 假设与模型建立规范（→ Clarifier / Milestone Reviewer 1）
    - coding_knowledge: 代码实现与编码防错（→ Coder）
    - chart_knowledge: 图表与可视化规范（→ Drawer）
    """
    sections = load_norm_sections()
    if not sections:
        return {
            "method_knowledge_active": "false",
            "problem_type": "unknown",
            "model_selection_knowledge": "",
            "type_knowledge": "",
            "assumption_knowledge": "",
            "coding_knowledge": "",
            "chart_knowledge": "",
        }

    resolved_type = problem_type if problem_type in _TYPE_GUIDE_SECTIONS else "unknown"
    # unknown 时题型专属指南为空（通用防错已含在 model_selection_knowledge 中），
    # 避免在题面缺失时误导 LLM 按某个具体题型建模。
    type_knowledge = (
        get_type_knowledge(resolved_type) if resolved_type != "unknown" else ""
    )
    return {
        "method_knowledge_active": "true",
        "problem_type": resolved_type,
        "model_selection_knowledge": _join_sections(sections, _NODE_SECTIONS["model_selection"]),
        "type_knowledge": type_knowledge,
        "assumption_knowledge": _join_sections(sections, _NODE_SECTIONS["assumptions"]),
        "coding_knowledge": _join_sections(sections, _NODE_SECTIONS["coding"]),
        "chart_knowledge": _join_sections(sections, _NODE_SECTIONS["chart"]),
    }
