"""承重结构分析规则层（V18）。

把「结论」与「结论所依赖的构造」显式连接成承重图，作为验证/论证/呈现要求
的附着点。本模块只做确定性规则：
- 符号注册表：从 LTM 提取构造候选与依赖引用；
- 验证状态：题面事实表 > 实证证据 > 仅假设 > 未验证；
- 物理锚点：与数据画像/数据情报/题面常量上下文的重合匹配；
- 承重度：依赖深度 + 验证缺失 + 锚点缺失 + 出错影响的加权归一化；
- 验证契约：按「承重度 × (1 − 验证完成度)」排序，根构造优先；
- 结论形态护栏：单向结论强制兜底、单链依赖强制交叉验证。

LLM 语义层（构造类型/风险/实验类型/结论形态）由 load_bearing_analyzer_node
调用，本模块负责规则兜底与后处理，保证 LLM 不可用时仍产出保守图。
"""

from __future__ import annotations

import re
from typing import Any

from modeling_assistant.schemas.responses import LoadBearingAnalysisResponse
from modeling_assistant.schemas.state import (
    ConclusionItem,
    ConstructItem,
    DynamicLTM,
    EmpiricalLayer,
    LoadBearingMap,
    StaticLTM,
    VerificationContract,
)

_VERIFICATION_SCORE = {
    "machine_verified": 1.0,
    "evidence_linked": 0.6,
    "self_set": 0.3,
    "unverified": 0.0,
}

# 实验类型 → 论文验收锚点小节（确定性映射，与题型无关）
_EXPERIMENT_SECTION = {
    "calibration": "8_sensitivity.tex",
    "perturbation": "8_sensitivity.tex",
    "cross_check": "8_sensitivity.tex",
    "contrast": "5_problemN.tex",
    "case_study": "5_problemN.tex",
    "artifact": "5_problemN.tex",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,11}")


def symbol_registry(dynamic_ltm: DynamicLTM) -> set[str]:
    """从 LTM 提取符号注册表：nomenclature 键 + 公式/假设/目标中出现的符号。"""
    symbols: set[str] = set()
    if dynamic_ltm is None:
        return symbols
    for sym in (getattr(dynamic_ltm, "nomenclature", None) or {}).keys():
        sym = str(sym).strip()
        if sym:
            symbols.add(sym)
    text = "\n".join(
        [
            *(getattr(dynamic_ltm, "equations", None) or []),
            *(getattr(dynamic_ltm, "assumptions", None) or []),
            getattr(dynamic_ltm, "objective", "") or "",
            getattr(dynamic_ltm, "solution_outline", "") or "",
        ]
    )
    for m in _TOKEN_RE.finditer(text):
        token = m.group(0)
        if len(token) >= 2:
            symbols.add(token)
    return {s for s in symbols if s}


def _facts_text(static_ltm: StaticLTM | None) -> str:
    if static_ltm is None:
        return ""
    parts: list[str] = []
    for fact in getattr(static_ltm, "problem_facts", None) or []:
        parts.append(fact.context or "")
        parts.append(fact.role_hint or "")
    return "\n".join(parts)


def _data_text(static_ltm: StaticLTM | None) -> str:
    if static_ltm is None:
        return ""
    parts: list[str] = []
    profile = getattr(static_ltm, "data_profile", None)
    if profile is not None:
        for col in getattr(profile, "columns", None) or []:
            parts.append(str(getattr(col, "name", "")))
        for fs in getattr(profile, "file_summaries", None) or []:
            for col in getattr(fs, "columns", None) or []:
                parts.append(str(getattr(col, "name", "")))
    parts.extend(getattr(static_ltm, "data_intelligence", None) or [])
    parts.extend(getattr(static_ltm, "data_findings", None) or [])
    return "\n".join(parts)


def _construct_verification(
    construct: str,
    registry: set[str],
    static_ltm: StaticLTM | None,
    dynamic_ltm: DynamicLTM | None,
    empirical: EmpiricalLayer | None,
) -> tuple[str, list[str]]:
    """规则判定构造的验证状态：题面事实表 > 实证证据 > 仅假设 > 未验证。"""
    facts = _facts_text(static_ltm)
    if construct and construct in facts:
        return "machine_verified", []
    for sym in registry:
        if len(sym) >= 2 and sym in construct and sym in facts:
            return "machine_verified", []

    run_ids: list[str] = []
    findings = getattr(empirical, "findings", None) or []
    for finding in findings:
        blob = " ".join(
            [
                getattr(finding, "assumption_tested", "") or "",
                getattr(finding, "evidence", "") or "",
                getattr(finding, "suggested_fix", "") or "",
            ]
        )
        if construct and construct in blob:
            run_ids.append(getattr(finding, "run_id", "") or "")
            continue
        for sym in registry:
            if len(sym) >= 2 and sym in construct and sym in blob:
                run_ids.append(getattr(finding, "run_id", "") or "")
                break
    if run_ids:
        return "evidence_linked", sorted({r for r in run_ids if r})

    assumptions_text = "\n".join(getattr(dynamic_ltm, "assumptions", None) or [])
    if construct and construct in assumptions_text:
        return "self_set", []
    return "unverified", []


def _physical_anchor(
    construct: str,
    registry: set[str],
    static_ltm: StaticLTM | None,
) -> str:
    """确定性物理锚点：与数据列/数据情报/题面常量上下文重合匹配。"""
    data = _data_text(static_ltm)
    facts = _facts_text(static_ltm)
    if construct:
        if construct in data:
            return f"数据对象:{construct}"
        if construct in facts:
            return f"题面常量上下文:{construct}"
        for sym in registry:
            if len(sym) >= 2 and sym in construct and sym in data:
                return f"数据对象:{sym}"
    return ""


def compute_load_bearing(
    is_root: bool,
    verification_status: str,
    has_anchor: bool,
    in_core: bool,
) -> float:
    """承重度 = 依赖深度 + 验证缺失 + 锚点缺失 + 出错影响，归一化到 0~1。"""
    score = 0.0
    score += 0.30 * (1.0 if is_root else 0.35)
    score += 0.30 * (1.0 - _VERIFICATION_SCORE.get(verification_status, 0.0))
    score += 0.20 * (0.0 if has_anchor else 1.0)
    score += 0.20 * (1.0 if in_core else 0.4)
    return round(min(1.0, max(0.0, score)), 2)


def _current_question(control: Any) -> str:
    questions = getattr(control, "sub_questions", None) or []
    idx = getattr(control, "current_sub_question_index", 0) or 0
    if questions and 0 <= idx < len(questions):
        return questions[idx]
    return ""


def _anchor_section(required_experiment: str, control: Any) -> str:
    section = _EXPERIMENT_SECTION.get(required_experiment, "8_sensitivity.tex")
    if section == "5_problemN.tex":
        idx = getattr(control, "current_sub_question_index", 0) or 0
        return f"{4 + idx}_problem{idx + 1}.tex"
    return section


def build_load_bearing_map(
    dynamic_ltm: DynamicLTM,
    static_ltm: StaticLTM | None,
    empirical: EmpiricalLayer | None,
    control: Any,
    response: LoadBearingAnalysisResponse | None = None,
    ltm_version: str = "",
) -> LoadBearingMap:
    """组装承重图：LLM 语义（可选） + 确定性规则后处理。"""
    registry = symbol_registry(dynamic_ltm)
    core_text = "\n".join(
        [
            *(getattr(dynamic_ltm, "equations", None) or []),
            getattr(dynamic_ltm, "objective", "") or "",
        ]
    )

    constructs: list[ConstructItem] = []
    if response is not None and response.constructs:
        for i, item in enumerate(response.constructs):
            name = (item.construct or "").strip()
            if not name:
                continue
            in_core = name in core_text or any(
                len(sym) >= 2 and sym in name and sym in core_text for sym in registry
            )
            status, run_ids = _construct_verification(
                name, registry, static_ltm, dynamic_ltm, empirical
            )
            anchor = _physical_anchor(name, registry, static_ltm) or (
                item.physical_anchor or ""
            ).strip()
            constructs.append(
                ConstructItem(
                    id=f"c{i + 1}",
                    construct=name,
                    construct_type=item.construct_type,
                    load_bearing=compute_load_bearing(
                        item.is_root, status, bool(anchor), in_core
                    ),
                    is_root=item.is_root,
                    verification_status=status,
                    evidence_run_ids=run_ids,
                    physical_anchor=anchor,
                    risk_if_wrong=item.risk_if_wrong,
                    required_experiment=item.required_experiment,
                )
            )
    else:
        # 规则兜底：nomenclature 键 + 长符号 token，保守构造清单
        names = [s for s in registry if s]
        for i, name in enumerate(sorted(names)):
            status, run_ids = _construct_verification(
                name, registry, static_ltm, dynamic_ltm, empirical
            )
            anchor = _physical_anchor(name, registry, static_ltm)
            constructs.append(
                ConstructItem(
                    id=f"c{i + 1}",
                    construct=name,
                    construct_type="parameter",
                    load_bearing=compute_load_bearing(
                        False, status, bool(anchor), name in core_text
                    ),
                    is_root=False,
                    verification_status=status,
                    evidence_run_ids=run_ids,
                    physical_anchor=anchor,
                    risk_if_wrong="",
                    required_experiment="perturbation",
                )
            )

    construct_by_name = {c.construct: c.id for c in constructs}
    conclusions: list[ConclusionItem] = []
    if response is not None and response.conclusions:
        for i, item in enumerate(response.conclusions):
            dep_ids = [
                construct_by_name[r]
                for r in (item.construct_refs or [])
                if r in construct_by_name
            ]
            shape = item.verdict_shape
            fallback = item.fallback_required
            spec = (item.fallback_spec or "").strip()
            if shape in ("all_negative", "all_positive"):
                fallback = True
                if not spec:
                    spec = "边界探测与反例搜索；给出结论翻转的条件与兜底表述"
            if len(dep_ids) == 1:
                fallback = True
                spec = f"{spec}；单链依赖：独立交叉验证".strip("；")
            conclusions.append(
                ConclusionItem(
                    id=f"v{i + 1}",
                    question_ref=item.question_ref,
                    answer_type=item.answer_type,
                    verdict_shape=shape,
                    load_bearing_construct_ids=dep_ids,
                    fallback_required=fallback,
                    fallback_spec=spec,
                )
            )
    else:
        question = _current_question(control) or (
            getattr(dynamic_ltm, "objective", "") or "全局结论"
        )
        conclusions.append(
            ConclusionItem(
                id="v1",
                question_ref=question,
                answer_type="verdict",
                verdict_shape="mixed",
                load_bearing_construct_ids=[c.id for c in constructs if c.is_root],
                fallback_required=False,
            )
        )

    root_gaps = [
        c.construct
        for c in constructs
        if c.is_root and _VERIFICATION_SCORE.get(c.verification_status, 0.0) < 0.6
    ]
    anchor_gaps = [c.construct for c in constructs if not c.physical_anchor]
    shape_risks = [
        v.id
        for v in conclusions
        if v.fallback_required or v.verdict_shape in ("all_negative", "all_positive")
    ]

    required_items: list[ConstructItem] = []
    seen: set[str] = set()
    for c in constructs:
        if c.id in seen:
            continue
        if c.is_root and _VERIFICATION_SCORE.get(c.verification_status, 0.0) < 0.6:
            required_items.append(c)
            seen.add(c.id)
    for c in constructs:
        if c.id in seen:
            continue
        if not c.physical_anchor:
            required_items.append(c)
            seen.add(c.id)

    def _priority_key(c: ConstructItem) -> tuple[float, float]:
        return (
            c.load_bearing * (1.0 - _VERIFICATION_SCORE.get(c.verification_status, 0.0)),
            c.load_bearing,
        )

    priority_order = [
        c.id
        for c in sorted(
            [c for c in constructs if c.id not in seen] + required_items,
            key=_priority_key,
            reverse=True,
        )
    ]
    acceptance_anchors = {
        c.id: _anchor_section(c.required_experiment, control) for c in constructs
    }

    return LoadBearingMap(
        conclusions=conclusions,
        constructs=constructs,
        contract=VerificationContract(
            priority_order=priority_order,
            required_items=required_items,
            acceptance_anchors=acceptance_anchors,
        ),
        root_gaps=root_gaps,
        anchor_gaps=anchor_gaps,
        shape_risks=shape_risks,
        ltm_version=ltm_version,
        analysis_incomplete=response is None,
        reasoning=(response.reasoning if response is not None else "规则兜底分析"),
    )


def reconcile_load_bearing_map(
    load_bearing_map: LoadBearingMap | None,
    empirical: EmpiricalLayer | None,
) -> LoadBearingMap | None:
    """执行证据回流：实证层新发现按构造名合并更新验证状态与证据链。"""
    if load_bearing_map is None:
        return None
    updated = load_bearing_map.model_copy(deep=True)
    findings = getattr(empirical, "findings", None) or []
    if not findings:
        return updated
    for item in updated.constructs:
        run_ids = set(item.evidence_run_ids or [])
        for finding in findings:
            blob = " ".join(
                [
                    getattr(finding, "assumption_tested", "") or "",
                    getattr(finding, "evidence", "") or "",
                    getattr(finding, "suggested_fix", "") or "",
                ]
            )
            if item.construct and item.construct in blob:
                run_ids.add(getattr(finding, "run_id", "") or "")
        if run_ids:
            item.evidence_run_ids = sorted(run_ids)
            if item.verification_status != "machine_verified":
                item.verification_status = "evidence_linked"
    return updated
