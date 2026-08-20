"""行文技艺题型级聚合：把同题型的多篇 WritingCraft 提炼为 CraftGuide。

规则（防过拟合）：每个模式至少被 min_occurrences 篇共有才进入共性；
范例句去重合并；章节名复用 exemplar_ingest 的归一化逻辑。
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

from modeling_assistant.data.exemplar_ingest import _normalize_section_name
from modeling_assistant.schemas.craft import (
    AlgorithmPattern,
    ArgumentFlow,
    CraftGuide,
    DerivationPattern,
    FigurePlacement,
    InterpretationPattern,
    SectionFocus,
    WritingCraft,
    WritingExample,
    WritingPattern,
)

logger = logging.getLogger(__name__)


def load_crafts(craft_dir: str | Path) -> list[WritingCraft]:
    crafts: list[WritingCraft] = []
    root = Path(craft_dir)
    if not root.exists():
        return crafts
    for path in sorted(root.glob("*.json")):
        try:
            crafts.append(WritingCraft.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("行文技艺卡加载失败 %s: %s", path, exc)
    return crafts


def load_craft_guides(guides_dir: str | Path) -> list[CraftGuide]:
    guides: list[CraftGuide] = []
    root = Path(guides_dir)
    if not root.exists():
        return guides
    for path in sorted(root.glob("*.json")):
        try:
            guides.append(CraftGuide.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("行文技艺指南加载失败 %s: %s", path, exc)
    return guides


def save_craft_guide(guide: CraftGuide, guides_dir: str | Path) -> Path:
    root = Path(guides_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{guide.problem_type}.json"
    path.write_text(guide.model_dump_json(indent=2), encoding="utf-8")
    return path


def _dedup_examples(examples: list[WritingExample]) -> list[WritingExample]:
    seen: set[str] = set()
    out: list[WritingExample] = []
    for ex in examples:
        key = ex.text
        if key and key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def _top_values(counter: Counter[str], limit: int = 3) -> list[str]:
    return [v for v, _c in counter.most_common(limit)]


def aggregate_craft_guides(
    crafts: list[WritingCraft],
    *,
    card_types: dict[str, str] | None = None,
    min_occurrences: int = 2,
) -> list[CraftGuide]:
    """按题型分组聚合。card_types 提供 card_id → problem_type 映射。"""
    card_types = card_types or {}
    groups: dict[str, list[WritingCraft]] = defaultdict(list)
    for craft in crafts:
        ptype = card_types.get(craft.card_id, "")
        if not ptype:
            continue
        groups[ptype].append(craft)

    guides: list[CraftGuide] = []
    for ptype, group in sorted(groups.items()):
        guides.append(_aggregate_group(ptype, group, min_occurrences))
    return guides


def _aggregate_group(ptype: str, group: list[WritingCraft], min_occ: int) -> CraftGuide:
    n = len(group)
    threshold = max(min_occ, round(n * 0.5))  # 至少 50% 论文共有

    # ── 数学推导：按章节聚类（触发条件并入内容，避免 4 篇样本下键过严）──
    deriv_groups: dict[str, list[DerivationPattern]] = defaultdict(list)
    for c in group:
        for d in c.derivation:
            key = _normalize_section_name(d.section) or "其他"
            deriv_groups[key].append(d)
    derivation_common: list[DerivationPattern] = []
    for _sec, items in deriv_groups.items():
        if len(items) < threshold:
            continue
        org_counter: Counter[str] = Counter()
        close_counter: Counter[str] = Counter()
        trg_counter: Counter[str] = Counter()
        examples: list[WritingExample] = []
        for d in items:
            org_counter.update(d.organization)
            close_counter.update(d.closing_moves)
            trg_counter.update([d.trigger] if d.trigger else [])
            examples.extend(d.examples)
        derivation_common.append(
            DerivationPattern(
                section=_sec,
                trigger=_top_values(trg_counter, 1)[0] if trg_counter else "",
                organization=_top_values(org_counter, 5),
                notation_usage=_top_values(Counter(d.notation_usage for d in items if d.notation_usage), 1)[0] if any(d.notation_usage for d in items) else "",
                depth_strategy=_top_values(Counter(d.depth_strategy for d in items if d.depth_strategy), 1)[0] if any(d.depth_strategy for d in items) else "",
                text_formula_ratio=_top_values(Counter(d.text_formula_ratio for d in items if d.text_formula_ratio), 1)[0] if any(d.text_formula_ratio for d in items) else "",
                closing_moves=_top_values(close_counter, 3),
                examples=_dedup_examples(examples)[:4],
            )
        )

    # ── 算法分析：按算法大类聚类（启发式/动态规划/精确/随机模拟等）──
    alg_groups: dict[str, list[AlgorithmPattern]] = defaultdict(list)
    for c in group:
        for a in c.algorithm:
            alg_groups[_normalize_algorithm_type(a.algorithm_type)].append(a)
    algorithm_common: list[AlgorithmPattern] = []
    for atype, items in alg_groups.items():
        if len(items) < threshold:
            continue
        pres_counter: Counter[str] = Counter()
        flow_counter: Counter[str] = Counter()
        fig_counter: Counter[str] = Counter()
        examples: list[WritingExample] = []
        for a in items:
            pres_counter.update(a.presentation)
            flow_counter.update(a.flow_organization)
            fig_counter.update(a.support_figures)
            examples.extend(a.examples)
        algorithm_common.append(
            AlgorithmPattern(
                algorithm_type=atype,
                presentation=_top_values(pres_counter, 3),
                flow_organization=_top_values(flow_counter, 6),
                complexity_analysis=_top_values(Counter(a.complexity_analysis for a in items if a.complexity_analysis), 1)[0] if any(a.complexity_analysis for a in items) else "",
                convergence_justification=_top_values(Counter(a.convergence_justification for a in items if a.convergence_justification), 1)[0] if any(a.convergence_justification for a in items) else "",
                result_reporting=_top_values(Counter(a.result_reporting for a in items if a.result_reporting), 1)[0] if any(a.result_reporting for a in items) else "",
                support_figures=_top_values(fig_counter, 4),
                examples=_dedup_examples(examples)[:4],
            )
        )

    # ── 模型解释：按解释对象聚类 ──
    int_groups: dict[str, list[InterpretationPattern]] = defaultdict(list)
    for c in group:
        for i in c.interpretation:
            int_groups[i.target or "其他"].append(i)
    interpretation_common: list[InterpretationPattern] = []
    for target, items in int_groups.items():
        if len(items) < threshold:
            continue
        org_counter: Counter[str] = Counter()
        move_counter: Counter[str] = Counter()
        examples: list[WritingExample] = []
        for i in items:
            org_counter.update(i.organization)
            move_counter.update(i.common_moves)
            examples.extend(i.examples)
        interpretation_common.append(
            InterpretationPattern(
                target=target,
                organization=_top_values(org_counter, 4),
                domain_linking=_top_values(Counter(i.domain_linking for i in items if i.domain_linking), 1)[0] if any(i.domain_linking for i in items) else "",
                parameter_meaning=_top_values(Counter(i.parameter_meaning for i in items if i.parameter_meaning), 1)[0] if any(i.parameter_meaning for i in items) else "",
                sensitivity_handling=_top_values(Counter(i.sensitivity_handling for i in items if i.sensitivity_handling), 1)[0] if any(i.sensitivity_handling for i in items) else "",
                common_moves=_top_values(move_counter, 4),
                examples=_dedup_examples(examples)[:4],
            )
        )

    # ── 行文语言：按句型功能聚类，范例合并去重 ──
    write_groups: dict[str, list[WritingPattern]] = defaultdict(list)
    for c in group:
        for w in c.writing:
            write_groups[w.function or "其他"].append(w)
    writing_common: list[WritingPattern] = []
    for function, items in write_groups.items():
        if len(items) < threshold:
            continue
        examples: list[WritingExample] = []
        for w in items:
            examples.extend(w.examples)
        writing_common.append(
            WritingPattern(
                function=function,
                examples=_dedup_examples(examples)[:5],
                usage_notes=_top_values(Counter(w.usage_notes for w in items if w.usage_notes), 1)[0] if any(w.usage_notes for w in items) else "",
            )
        )
    # 按固定功能顺序输出
    func_order = ["摘要句子", "假设铺垫句", "过渡衔接句", "结果解读句", "结论升华句", "局限说明句"]
    writing_common.sort(key=lambda w: func_order.index(w.function) if w.function in func_order else 99)

    # ── 图片位置规划：按 (图类型, 章节) 聚类 ──
    fig_groups: dict[tuple[str, str], list[FigurePlacement]] = defaultdict(list)
    for c in group:
        for f in c.figure_placements:
            key = (f.figure_type, _normalize_section_name(f.section))
            fig_groups[key].append(f)
    figure_placement_common: list[FigurePlacement] = []
    for (_ft, _sec), items in fig_groups.items():
        if len(items) < threshold:
            continue
        role_counter: Counter[str] = Counter(i.argument_role for i in items if i.argument_role)
        figure_placement_common.append(
            FigurePlacement(
                figure_type=_ft,
                section=_sec,
                argument_role=_top_values(role_counter, 1)[0] if role_counter else "",
                caption_style=_top_values(Counter(i.caption_style for i in items if i.caption_style), 1)[0] if any(i.caption_style for i in items) else "",
            )
        )

    # ── 正文侧重点：按章节归一化聚类 ──
    sec_groups: dict[str, list[SectionFocus]] = defaultdict(list)
    for c in group:
        for s in c.section_focuses:
            sec_groups[_normalize_section_name(s.section)].append(s)
    section_focus_common: list[SectionFocus] = []
    for section, items in sec_groups.items():
        if len(items) < threshold:
            continue
        focus_counter: Counter[str] = Counter(i.focus for i in items if i.focus)
        order_counter: Counter[str] = Counter()
        for i in items:
            order_counter.update(i.internal_order)
        section_focus_common.append(
            SectionFocus(
                section=section,
                focus=_top_values(focus_counter, 1)[0] if focus_counter else "",
                weight=round(sum(i.weight for i in items) / len(items), 3),
                internal_order=_top_values(order_counter, 6),
            )
        )

    # ── 论证链条：步骤取高频，衔接取高频 ──
    step_counter: Counter[str] = Counter()
    trans_counter: Counter[str] = Counter()
    for c in group:
        if c.argument_flow is None:
            continue
        step_counter.update(c.argument_flow.steps)
        trans_counter.update(c.argument_flow.transitions)
    argument_flow_common = None
    steps = [s for s, cnt in step_counter.items() if cnt >= threshold]
    transitions = [t for t, cnt in trans_counter.items() if cnt >= threshold]
    if steps:
        argument_flow_common = ArgumentFlow(
            steps=_top_values(Counter(steps), 10) or steps,
            transitions=_top_values(trans_counter, 6) or transitions,
        )

    return CraftGuide(
        problem_type=ptype,
        derivation_common=derivation_common,
        algorithm_common=algorithm_common,
        interpretation_common=interpretation_common,
        writing_common=writing_common,
        figure_placement_common=figure_placement_common,
        section_focus_common=section_focus_common,
        argument_flow_common=argument_flow_common,
        exemplar_ids=[c.card_id for c in group],
    )


def _normalize_algorithm_type(atype: str) -> str:
    """把具体算法名归并到大类，便于跨论文聚合。"""
    t = atype or ""
    if any(k in t for k in ("贪心", "启发", "遗传", "退火", "粒子群", "禁忌", "元启发")):
        return "启发式"
    if any(k in t for k in ("动态规划", "DP")):
        return "动态规划"
    if any(k in t for k in ("线性规划", "整数规划", "求解器", "LINGO", "CPLEX", "精确")):
        return "精确求解"
    if any(k in t for k in ("蒙特卡洛", "随机模拟", "仿真")):
        return "随机模拟"
    if any(k in t for k in ("机器学习", "回归", "聚类", "分类", "神经网络", "随机森林")):
        return "机器学习"
    return t[:12] or "其他"
