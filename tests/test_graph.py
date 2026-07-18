from __future__ import annotations

from pathlib import Path

from langgraph.types import Command

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config.settings import AppSettings, load_settings
from modeling_assistant.graph.builder import build_graph
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ControlState,
    DynamicLTM,
    PlanCandidate,
    StaticLTM,
)


def _run_to_completion(app, state: dict, config: dict) -> dict:
    """运行图直到完成，自动批准所有 HITL 中断。"""
    current_input: dict | Command = state
    result = app.invoke(current_input, config)
    while result.get("control", ControlState()).hitl_required:
        result = app.invoke(Command(resume="approve"), config)
    return result


def test_graph_runs_minimal_flow():
    runtime = AgentRuntime.from_settings(load_settings(output_dir=Path("outputs")))
    app = build_graph(runtime=runtime)
    config = {"configurable": {"thread_id": "test-minimal"}}

    final_state = _run_to_completion(
        app,
        {
            "static_ltm": StaticLTM(raw_problem="预测交通拥堵并优化信号灯。"),
            "dynamic_ltm": DynamicLTM(),
            "ltm_archive": [],
            "control": ControlState(),
            "artifacts": ArtifactBundle(),
            "prompt_audit": {},
        },
        config,
    )

    assert final_state["control"].phase == "completed"
    assert final_state["static_ltm"].raw_problem == "预测交通拥堵并优化信号灯。"
    assert final_state["static_ltm"].problem_understanding
    assert isinstance(final_state["dynamic_ltm"].assumptions, list)
    assert isinstance(final_state["dynamic_ltm"].nomenclature, dict)
    assert isinstance(final_state["dynamic_ltm"].equations, list)
    assert final_state["dynamic_ltm"].objective
    assert final_state["dynamic_ltm"].solution_outline
    assert len(final_state["ltm_archive"]) >= 1
    assert final_state["artifacts"].figure_paths
    assert final_state["artifacts"].latex_path
    assert {"coder", "drawer", "writer"}.issubset(final_state["prompt_audit"])
    # result_paths 依赖 Coder 在真实数据上执行成功；降级模式（无有效 API key）下可能为空
    if final_state["artifacts"].result_paths:
        assert all(isinstance(p, str) for p in final_state["artifacts"].result_paths)


def test_runtime_settings_are_copied_into_control_state():
    runtime = AgentRuntime.from_settings(
        load_settings(
            max_debate_rounds=5,
            innovation_threshold=70,
            feasibility_threshold=65,
            innovation_weight=0.6,
            feasibility_weight=0.4,
            output_dir=Path("outputs"),
            llm_model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            api_base_url="https://api.deepseek.com",
        )
    )
    app = build_graph(runtime=runtime)
    config = {"configurable": {"thread_id": "test-settings"}}

    final_state = _run_to_completion(
        app,
        {
            "static_ltm": StaticLTM(raw_problem="测试：优化物流路径。"),
            "dynamic_ltm": DynamicLTM(),
            "ltm_archive": [],
            "control": ControlState(),
            "artifacts": ArtifactBundle(),
            "prompt_audit": {},
        },
        config,
    )

    assert final_state["control"].max_debate_rounds == 5
    assert final_state["control"].innovation_threshold == 70
    assert final_state["control"].feasibility_threshold == 65
    assert final_state["control"].innovation_weight == 0.6
    assert final_state["control"].feasibility_weight == 0.4
    assert final_state["control"].phase == "completed"
    # 同上，result_paths 在降级模式下可能为空
    if final_state["artifacts"].result_paths:
        assert all(isinstance(p, str) for p in final_state["artifacts"].result_paths)


def test_realist_pruning_filters_low_feasibility():
    """Realist 应将 feasibility < threshold 的方案标记为 kill。"""
    from modeling_assistant.agents.nodes import realist_node
    from modeling_assistant.schemas.state import GraphState

    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(
            innovation_threshold=60,
            feasibility_threshold=60,
            top_k_plans=[
                PlanCandidate(
                    id="p1",
                    title="好方案",
                    description="",
                    innovation_score=80,
                    feasibility_score=75,
                ),
                PlanCandidate(
                    id="p2",
                    title="不可行方案",
                    description="",
                    innovation_score=90,
                    feasibility_score=30,
                ),
            ],
        ),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    runtime = AgentRuntime.from_settings(
        AppSettings(output_dir=Path("outputs"), api_key_env="MISSING_KEY_FOR_TEST")
    )
    result = realist_node(state, runtime=runtime)
    plans = result["control"].top_k_plans
    assert plans[0].verdict == "keep"
    assert plans[1].verdict == "kill"
    assert result["control"].selected_plan_id == "p1"


def test_arbiter_routing_triggers_only_after_max_rounds():
    """route_after_realist 仅在 debate_round > max_debate_rounds 时进入 arbiter。"""
    from modeling_assistant.graph.routing import route_after_realist

    state = {
        "control": ControlState(
            innovation_score=80,
            feasibility_score=80,
            innovation_threshold=60,
            feasibility_threshold=60,
            debate_round=2,
            max_debate_rounds=3,
        )
    }
    assert route_after_realist(state) == "clarifier"

    # 分数不达标且未超轮数 → mathematician
    state["control"] = ControlState(
        innovation_score=40,
        feasibility_score=40,
        debate_round=2,
        max_debate_rounds=3,
    )
    assert route_after_realist(state) == "mathematician"

    # 刚好达到最大轮数且分数不达标 → 仍回 mathematician（Goal.md: >3 才介入）
    state["control"] = ControlState(
        innovation_score=40,
        feasibility_score=40,
        debate_round=3,
        max_debate_rounds=3,
    )
    assert route_after_realist(state) == "mathematician"

    # 超过最大轮数 → arbiter（无论分数是否达标）
    state["control"] = ControlState(
        innovation_score=80,
        feasibility_score=80,
        debate_round=4,
        max_debate_rounds=3,
    )
    assert route_after_realist(state) == "arbiter"


def test_coder_rollback_classification():
    """_classify_coder_error 应正确区分 architect 与 clarifier 回滚目标。"""
    from modeling_assistant.agents.nodes import _classify_coder_error

    assert _classify_coder_error("SyntaxError: invalid syntax") == "architect"
    assert _classify_coder_error("ModuleNotFoundError: No module named 'foo'") == "architect"
    assert _classify_coder_error("ValueError: optimization failed to converge") == "clarifier"
    assert _classify_coder_error("RuntimeError: solver infeasible") == "clarifier"


def test_route_after_coder_supports_clarifier():
    from modeling_assistant.graph.routing import route_after_coder

    state = {
        "control": ControlState(
            coder_error_count=3,
            coder_rollback_target="clarifier",
        )
    }
    assert route_after_coder(state) == "clarifier"

    state["control"] = ControlState(
        coder_error_count=3,
        coder_rollback_target="architect",
    )
    assert route_after_coder(state) == "architect"

    # 无结果文件路径时降级到 collect_artifacts，等待 Drawer 完成后统一触发 Writer
    state["control"] = ControlState(coder_error_count=0)
    assert route_after_coder(state) == "collect_artifacts"


def test_route_after_final_review_goes_to_rollback():
    """终稿 HITL 的 retry 应先进入 rollback 节点 checkout 版本。"""
    from modeling_assistant.graph.routing import route_after_final_review

    state = {"control": ControlState(rollback_to_version="v1.0")}
    assert route_after_final_review(state) == "rollback"

    state = {"control": ControlState()}
    assert route_after_final_review(state) == "hitl_final"


def test_route_after_rollback_respects_source():
    """根据 rollback_source 决定回滚去向。"""
    from modeling_assistant.graph.routing import route_after_rollback

    state = {"control": ControlState(rollback_source="final_hitl")}
    assert route_after_rollback(state) == "mathematician"

    state = {"control": ControlState(rollback_source="architecture_hitl")}
    assert route_after_rollback(state) == "architect"

    state = {"control": ControlState(rollback_source="arbitration")}
    assert route_after_rollback(state) == "architect"


def test_milestone_reviewer_1_hard_rejection():
    """Milestone Reviewer 1 对空动态 LTM 应直接打回 Mathematician。"""
    from modeling_assistant.agents.nodes import milestone_reviewer_1_node

    state = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(),  # 空
        "control": ControlState(),
        "ltm_archive": [],
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    result = milestone_reviewer_1_node(state)
    assert result["control"].need_rebrainstorm is True
    assert result["control"].phase == "milestone_review_1_rejected"
