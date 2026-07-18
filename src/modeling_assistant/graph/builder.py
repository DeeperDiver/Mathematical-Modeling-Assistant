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
    final_reviewer_node,
    hitl_arbitration_node,
    hitl_architecture_node,
    hitl_final_node,
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
    return {"dynamic_ltm": dynamic_ltm, "control": control}


def collect_artifacts_node(
    state: GraphState,
    runtime: AgentRuntime | None = None,
    config: dict | None = None,
) -> GraphState:
    """同步节点：等待 Drawer 与 Coder/ResultReviewer 两条并行路径都完成后触发 Writer。

    LangGraph 的 fan-in 语义保证该节点在两条入边都到达后只执行一次。
    """
    control = state.get("control", ControlState()).model_copy(deep=True)
    control.phase = "artifacts_collected"
    return {"control": control}


def build_graph(runtime: AgentRuntime | None = None, *, checkpointer: InMemorySaver | None = None):
    resolved_runtime = runtime or get_default_runtime()
    graph = StateGraph(GraphState)

    graph.add_node("problem", _bind_runtime(problem_node, resolved_runtime))
    graph.add_node("analyst", _bind_runtime(analyst_node, resolved_runtime))
    graph.add_node("data_profile", _bind_runtime(data_profile_node, resolved_runtime))
    graph.add_node("searcher", _bind_runtime(searcher_node, resolved_runtime))
    graph.add_node("mathematician", _bind_runtime(mathematician_node, resolved_runtime))
    graph.add_node("realist", _bind_runtime(realist_node, resolved_runtime))
    graph.add_node("arbiter", _bind_runtime(arbiter_node, resolved_runtime))
    graph.add_node("hitl_arbitration", _bind_runtime(hitl_arbitration_node, resolved_runtime))
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
    graph.add_edge("problem", "analyst")
    graph.add_edge("analyst", "data_profile")
    graph.add_edge("data_profile", "searcher")
    graph.add_edge("searcher", "mathematician")
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
    # 通过 collect_artifacts fan-in 节点同步 Drawer 与 Coder/ResultReviewer/Reflection 两条路径
    graph.add_edge("drawer", "collect_artifacts")
    graph.add_conditional_edges(
        "coder",
        route_after_coder,
        {
            "architect": "architect",
            "clarifier": "clarifier",
            "result_reviewer": "result_reviewer",
            "collect_artifacts": "collect_artifacts",
        },
    )
    # ResultReviewer 通过 → reflection（提取实证发现）；失败 → 回退
    graph.add_conditional_edges(
        "result_reviewer",
        route_after_result_reviewer,
        {"reflection": "reflection", "architect": "architect", "clarifier": "clarifier"},
    )
    # Reflection 后：有 refuted 发现 → 回 Clarifier 修正假设；否则 → collect_artifacts
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"clarifier": "clarifier", "collect_artifacts": "collect_artifacts"},
    )
    graph.add_edge("collect_artifacts", "writer")
    graph.add_edge("writer", "final_reviewer")
    graph.add_edge("final_reviewer", "hitl_final")
    graph.add_conditional_edges(
        "hitl_final",
        route_after_final_review,
        {"rollback": "rollback", "hitl_final": END},
    )

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
