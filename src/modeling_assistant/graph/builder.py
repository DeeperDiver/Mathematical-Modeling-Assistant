from __future__ import annotations

from collections.abc import Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from modeling_assistant.agents.runtime import AgentRuntime, get_default_runtime
from modeling_assistant.agents.nodes import (
    analyst_node,
    arbiter_node,
    architect_node,
    clarifier_node,
    coder_node,
    drawer_node,
    exemplar_loader_node,
    fact_extractor_node,
    final_reviewer_node,
    hitl_arbitration_node,
    hitl_architecture_node,
    hitl_final_node,
    hitl_modeling_node,
    mathematician_node,
    milestone_reviewer_1_node,
    problem_node,
    realist_node,
    reflection_node,
    searcher_node,
    writer_node,
)
from modeling_assistant.data.loader import data_profile_node
from modeling_assistant.graph.routing import (
    route_after_architecture_hitl,
    route_after_arbiter,
    route_after_coder,
    route_after_final_review,
    route_after_hitl_modeling,
    route_after_milestone_reviewer_1,
    route_after_realist,
    route_after_reflection,
    route_after_result_reviewer,
    route_after_rollback,
)
from modeling_assistant.memory.archive import checkout_snapshot
from modeling_assistant.schemas.state import ControlState, GraphState
from modeling_assistant.validation.results import result_reviewer_node


NodeFn = Callable[[GraphState], GraphState]


def _bind_runtime(
    node: Callable[[GraphState, AgentRuntime | None, dict | None], GraphState],
    runtime: AgentRuntime,
) -> NodeFn:
    def wrapped(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
        return node(state, runtime=runtime, config=config)

    return wrapped


def rollback_node(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    control = state["control"].model_copy(deep=True)
    archive = state.get("ltm_archive", [])
    version = control.rollback_to_version
    # 未指定版本时，默认回滚到最近的 archive snapshot
    if not version and archive:
        version = archive[-1].version
    if not version:
        return {"control": control}

    dynamic_ltm = checkout_snapshot(archive, version)
    control.rollback_to_version = None
    control.hitl_required = False
    control.hitl_stage = "none"
    control.phase = f"rolled_back_to_{version}"
    # 用户主动 rollback 意味着"我要重新做"，重置建模预算
    control.modeling_revision_count = 0
    return {"dynamic_ltm": dynamic_ltm, "control": control}


def collect_artifacts_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """reflection 通过后触发 Writer 的缓冲节点。

    V5 修复：移除 fan-in 语义。原设计 collect_artifacts 是 fan-in 节点（入边：
    drawer + reflection），但当 result_reviewer 失败回退（不经过 reflection）时，
    LangGraph 的 fan-in 语义让 collect_artifacts 在 drawer 到达后立即执行，与
    result_reviewer 在同一 superstep 并行。collect_artifacts 看到的 state 是上一
    superstep 的（phase="code_executed_successfully"），看不到 result_reviewer 即将
    设置的 phase="result_review_failed"。两个 control 输出在 overwrite_reducer 合并时
    竞争，导致 writer 被错误调用，与 clarifier 回退路径并行产生多个 HITL 中断。

    修复：移除 drawer → collect_artifacts 边。collect_artifacts 只在 reflection 通过
    后执行（入边只有 reflection）。drawer 的产物通过 merge_artifacts_reducer 直接
    合并到 state.artifacts，writer 直接读取，不需要 collect_artifacts 来"收集"。

    回退路径（result_reviewer 失败 / coder 失败 / reflection 触发 clarifier 修正）
    均不经过 collect_artifacts，因此这里不需要检查回退标志。
    """
    control = state.get("control", ControlState()).model_copy(deep=True)
    control.phase = "artifacts_collected"
    return {"control": control}


def build_graph(runtime: AgentRuntime | None = None, *, checkpointer: InMemorySaver | None = None):
    resolved_runtime = runtime or get_default_runtime()
    graph = StateGraph(GraphState)

    graph.add_node("problem", _bind_runtime(problem_node, resolved_runtime))
    # V11 修复：插入 fact_extractor_node（纯机器提取题目常量），在 analyst 之前运行
    graph.add_node("fact_extractor", _bind_runtime(fact_extractor_node, resolved_runtime))
    graph.add_node("analyst", _bind_runtime(analyst_node, resolved_runtime))
    graph.add_node("data_profile", _bind_runtime(data_profile_node, resolved_runtime))
    graph.add_node("searcher", _bind_runtime(searcher_node, resolved_runtime))
    graph.add_node("exemplar_loader", _bind_runtime(exemplar_loader_node, resolved_runtime))
    graph.add_node("mathematician", _bind_runtime(mathematician_node, resolved_runtime))
    graph.add_node("realist", _bind_runtime(realist_node, resolved_runtime))
    graph.add_node("arbiter", _bind_runtime(arbiter_node, resolved_runtime))
    graph.add_node("hitl_arbitration", _bind_runtime(hitl_arbitration_node, resolved_runtime))
    graph.add_node("hitl_modeling", _bind_runtime(hitl_modeling_node, resolved_runtime))
    graph.add_node("clarifier", _bind_runtime(clarifier_node, resolved_runtime))
    graph.add_node("milestone_reviewer_1", _bind_runtime(milestone_reviewer_1_node, resolved_runtime))
    graph.add_node("hitl_architecture", _bind_runtime(hitl_architecture_node, resolved_runtime))
    graph.add_node("rollback", rollback_node)
    graph.add_node("architect", _bind_runtime(architect_node, resolved_runtime))
    graph.add_node("drawer", _bind_runtime(drawer_node, resolved_runtime))
    graph.add_node("coder", _bind_runtime(coder_node, resolved_runtime))
    graph.add_node("result_reviewer", _bind_runtime(result_reviewer_node, resolved_runtime))
    graph.add_node("reflection", _bind_runtime(reflection_node, resolved_runtime))
    graph.add_node("collect_artifacts", collect_artifacts_node)
    graph.add_node("writer", _bind_runtime(writer_node, resolved_runtime))
    graph.add_node("final_reviewer", _bind_runtime(final_reviewer_node, resolved_runtime))
    graph.add_node("hitl_final", _bind_runtime(hitl_final_node, resolved_runtime))

    graph.add_edge(START, "problem")
    # V11 修复：problem → fact_extractor → analyst，确保 Analyst 能看到机器提取的常量
    graph.add_edge("problem", "fact_extractor")
    graph.add_edge("fact_extractor", "analyst")
    graph.add_edge("analyst", "data_profile")
    graph.add_edge("data_profile", "searcher")
    # Exemplar Learning System：检索优秀论文表达知识后再进入建模阶段
    graph.add_edge("searcher", "exemplar_loader")
    graph.add_edge("exemplar_loader", "mathematician")
    graph.add_edge("mathematician", "realist")
    graph.add_conditional_edges(
        "realist",
        route_after_realist,
        {
            "mathematician": "mathematician",
            "arbiter": "arbiter",
            "clarifier": "clarifier",
        },
    )
    graph.add_conditional_edges(
        "arbiter",
        route_after_arbiter,
        {"clarifier": "clarifier", "rollback": "rollback", "hitl_arbitration": "hitl_arbitration"},
    )
    graph.add_conditional_edges(
        "hitl_arbitration",
        route_after_arbiter,
        {"clarifier": "clarifier", "rollback": "rollback", "hitl_arbitration": "hitl_arbitration"},
    )
    graph.add_edge("clarifier", "milestone_reviewer_1")
    graph.add_conditional_edges(
        "milestone_reviewer_1",
        route_after_milestone_reviewer_1,
        {"mathematician": "mathematician", "hitl_architecture": "hitl_architecture"},
    )
    graph.add_conditional_edges(
        "hitl_architecture",
        route_after_architecture_hitl,
        {"rollback": "rollback", "architect": "architect"},
    )
    graph.add_conditional_edges(
        "rollback",
        route_after_rollback,
        {"architect": "architect", "mathematician": "mathematician"},
    )
    graph.add_edge("architect", "drawer")
    graph.add_edge("architect", "coder")
    # V5 修复：移除 drawer → collect_artifacts 边。
    # 原设计 collect_artifacts 是 fan-in 节点（drawer + reflection），但当
    # result_reviewer 失败回退（不经过 reflection）时，LangGraph 的 fan-in 语义
    # 让 collect_artifacts 在 drawer 到达后立即执行，与 result_reviewer 在同一
    # superstep 并行，导致 phase 竞争和 multiple pending interrupts 错误。
    # 现在 collect_artifacts 只在 reflection 通过后执行，drawer 的产物通过
    # merge_artifacts_reducer 直接合并到 state.artifacts，writer 直接读取。
    graph.add_conditional_edges(
        "coder",
        route_after_coder,
        {
            "architect": "architect",
            "clarifier": "clarifier",
            "result_reviewer": "result_reviewer",
            "reflection": "reflection",
        },
    )
    # ResultReviewer 通过 → reflection（提取实证发现）；失败 → 回退
    graph.add_conditional_edges(
        "result_reviewer",
        route_after_result_reviewer,
        {"reflection": "reflection", "architect": "architect", "clarifier": "clarifier"},
    )
    # Reflection 后：Meta-Router 决策优先（mathematician/clarifier/architect/collect_artifacts）；
    # V6 修复：coder 失败（result_paths 空）+ budget 未耗尽 → 回 architect 重试
    # 否则 → collect_artifacts → writer
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "clarifier": "clarifier",
            "collect_artifacts": "collect_artifacts",
            "architect": "architect",
            "mathematician": "mathematician",
            "hitl_modeling": "hitl_modeling",
        },
    )
    # HITL modeling 节点后：根据人类决策路由
    # - accept → collect_artifacts（产出"待验证"论文）
    # - retry → architect（重置预算后回 Architect 重试当前方案）
    # - redirect → mathematician（重置预算后回 Mathematician 重新发散）
    graph.add_conditional_edges(
        "hitl_modeling",
        route_after_hitl_modeling,
        {"collect_artifacts": "collect_artifacts", "architect": "architect", "mathematician": "mathematician"},
    )
    # collect_artifacts 只在 reflection 通过后执行，直接前进到 writer
    # （回退路径不经过 collect_artifacts，无需条件路由）
    graph.add_edge("collect_artifacts", "writer")
    graph.add_edge("writer", "final_reviewer")
    graph.add_edge("final_reviewer", "hitl_final")
    graph.add_conditional_edges(
        "hitl_final",
        route_after_final_review,
        {"rollback": "rollback", "hitl_final": END},
    )

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
