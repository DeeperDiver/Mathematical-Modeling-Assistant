from __future__ import annotations

from pathlib import Path

from langgraph.types import Command

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config.settings import AppSettings, load_settings
from modeling_assistant.graph.builder import build_graph
from modeling_assistant.schemas.responses import (
    AnalystResponse,
    ArchitectResponse,
    ClarifierResponse,
    CoderResponse,
    DataIntelligenceResponse,
    DrawerResponse,
    LoadBearingAnalysisResponse,
    MathematicianResponse,
    MilestoneReviewer1Response,
    PlanEvaluation,
    RealistResponse,
    ReflectionResponse,
    ResultContract,
    WriterResponse,
)
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
    while True:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return result
        interrupt_value = (
            interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        )
        stage = (
            interrupt_value.get("stage")
            if isinstance(interrupt_value, dict)
            else str(interrupt_value)
        )
        result = app.invoke(
            Command(
                resume=(
                    "auto"
                    if stage == "implementation_human"
                    else "pass"
                    if stage == "sub_question_acceptance"
                    else "accept"
                    if stage == "cross_sub_question"
                    else "approve"
                )
            ),
            config,
        )


def _mock_sub_question_runtime(monkeypatch, output_dir: Path, mode: str):
    """V14：小题循环 e2e 的 mock runtime，避免真实 LLM 网络调用。"""
    solution = (
        "import os\nfrom pathlib import Path\nimport pandas as pd\n"
        "out = os.environ['MODELING_OUTPUT_DIR']\n"
        "p = Path(out) / 'results' / 'output.csv'\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "pd.DataFrame({'answer': [12.5]}).to_csv(p, index=False)\n"
    )
    figure = (
        "import os, matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "out = os.environ['MODELING_OUTPUT_DIR']\n"
        "d = os.path.join(out, 'figures')\n"
        "os.makedirs(d, exist_ok=True)\n"
        "plt.plot([1, 2, 3], [1, 4, 9])\n"
        "plt.savefig(os.path.join(d, 'figure1.png'))\n"
    )

    def mock_invoke_structured(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        if name == "analyst":
            return AnalystResponse(problem_understanding="x", data_schema={})
        if name == "mathematician":
            return MathematicianResponse(plans=[{"id": "p1", "title": "t", "description": "d", "innovation_score": 80, "feasibility_score": 80}])
        if name == "realist":
            return RealistResponse(plan_evaluations=[PlanEvaluation(plan_id="p1", innovation_score=80, feasibility_score=80, verdict="keep")])
        if name == "clarifier":
            return ClarifierResponse(assumptions=["a"], nomenclature={"x": "x"}, equations=["y=x"], objective="o", solution_outline="s", commit_summary="v")
        if name == "milestone_reviewer_1":
            return MilestoneReviewer1Response(approval=True, issues=[], feedback="ok")
        if name == "load_bearing_analyzer":
            return LoadBearingAnalysisResponse(constructs=[], conclusions=[], reasoning="ok")
        if name == "architect":
            return ArchitectResponse(
                outline={"摘要": "a", "问题重述": "b", "模型建立": "c", "模型求解": "d", "结果分析": "e"},
                pseudocode=["s1"],
                algorithms_summary="alg",
                result_contract=ResultContract(allow_single_row=True),
            )
        if name == "coder":
            return CoderResponse(code=solution, result_path="results/output.csv")
        if name == "drawer":
            return DrawerResponse(
                figure_code=figure,
                figure_paths=["figures/figure1.png"],
                observation="",
                observation_verdict="inconclusive",
                observation_confidence=0.3,
            )
        if name == "reflection":
            return ReflectionResponse(findings=[], run_summary="ok")
        if name == "data_analyst":
            return DataIntelligenceResponse(insights=["i"])
        if name == "writer":
            return WriterResponse(latex_content="\\documentclass{article}\n")
        raise AssertionError(f"unexpected prompt: {name}")

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke_structured)
    monkeypatch.setattr(
        AgentRuntime, "invoke", lambda self, name, state, system_prompt=None: "kw"
    )
    return AgentRuntime.from_settings(
        AppSettings(
            output_dir=output_dir,
            api_key_env="MISSING_KEY_FOR_TEST",
            coder_external_mode=mode,
            search_enabled=False,
        )
    )


def test_graph_runs_minimal_sub_question_flow(tmp_path, monkeypatch):
    """V14：小题循环 minimal flow——单小题 human 模式全链路。"""
    runtime = _mock_sub_question_runtime(monkeypatch, tmp_path / "outputs", "human")
    app = build_graph(runtime=runtime)
    config = {"configurable": {"thread_id": "test-minimal"}}

    final_state = _run_to_completion(
        app,
        {
            "static_ltm": StaticLTM(raw_problem="问题1 预测交通拥堵。"),
            "dynamic_ltm": DynamicLTM(),
            "ltm_archive": [],
            "control": ControlState(),
            "artifacts": ArtifactBundle(),
            "prompt_audit": {},
        },
        config,
    )

    assert final_state["control"].phase == "completed"
    assert final_state["static_ltm"].raw_problem == "问题1 预测交通拥堵。"
    ctrl = final_state["control"]
    assert ctrl.sub_questions == ["问题1 预测交通拥堵。"]
    assert len(ctrl.sub_results) == 1
    assert len(final_state["ltm_archive"]) >= 1
    assert any("q1.csv" in p for p in final_state["artifacts"].result_paths)
    assert final_state["artifacts"].latex_path
    assert {"coder", "writer"}.issubset(final_state["prompt_audit"])


def test_runtime_settings_are_copied_into_control_state(tmp_path, monkeypatch):
    """V14：配置仍应复制进 control，且 builtin 模式在小题循环中可用。"""
    runtime = _mock_sub_question_runtime(monkeypatch, tmp_path / "outputs", "builtin")
    runtime.settings.max_debate_rounds = 5
    runtime.settings.innovation_threshold = 70
    runtime.settings.feasibility_threshold = 65
    runtime.settings.innovation_weight = 0.6
    runtime.settings.feasibility_weight = 0.4
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
    assert len(final_state["control"].sub_results) == 1
    assert any("q1.csv" in p for p in final_state["artifacts"].result_paths)


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


def test_realist_plan_pool_keeps_top_three():
    """V23：Realist 剪枝后保留评分最高的前 3 个 keep 方案进方案池。"""
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
                PlanCandidate(id="p1", title="方案1", description="", innovation_score=90, feasibility_score=90),
                PlanCandidate(id="p2", title="方案2", description="", innovation_score=85, feasibility_score=85),
                PlanCandidate(id="p3", title="方案3", description="", innovation_score=80, feasibility_score=80),
                PlanCandidate(id="p4", title="方案4", description="", innovation_score=70, feasibility_score=70),
                PlanCandidate(id="p5", title="方案5", description="", innovation_score=60, feasibility_score=60),
            ],
        ),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    runtime = AgentRuntime.from_settings(
        AppSettings(output_dir=Path("outputs"), api_key_env="MISSING_KEY_FOR_TEST")
    )
    result = realist_node(state, runtime=runtime)
    assert result["control"].selected_plan_id == "p1"
    assert result["control"].plan_pool_ids == ["p1", "p2", "p3"]


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
    """V6.1 修改：所有 coder 失败（result_paths 空）都经过 reflection，
    由 reflection_node 消费 budget，route_after_reflection 决定回退到 coder_rollback_target。
    所以 route_after_coder 不再直接返回 clarifier/architect，而是返回 reflection。
    """
    from modeling_assistant.graph.routing import route_after_coder

    # coder 失败 3 次 + result_paths 空 → reflection（让 reflection 消费 budget）
    state = {
        "control": ControlState(
            coder_error_count=3,
            coder_rollback_target="clarifier",
        ),
        "artifacts": ArtifactBundle(result_paths=[]),
    }
    assert route_after_coder(state) == "reflection"

    # coder 失败 3 次 + result_paths 空 + rollback_target=architect → 仍然 reflection
    state["control"] = ControlState(
        coder_error_count=3,
        coder_rollback_target="architect",
    )
    assert route_after_coder(state) == "reflection"

    # 无结果文件路径时（Coder 失败/空代码）进入 reflection 节点，
    # 从失败日志中提炼"假设是否可数值化"等实证信号，而非直接丢弃
    state["control"] = ControlState(coder_error_count=0)
    assert route_after_coder(state) == "reflection"

    # 有结果文件路径 → result_reviewer（即使 coder_error_count >= 3）
    state = {
        "control": ControlState(
            coder_error_count=3,
            coder_rollback_target="architect",
        ),
        "artifacts": ArtifactBundle(result_paths=["outputs/results/output.csv"]),
    }
    assert route_after_coder(state) == "result_reviewer"


def test_route_after_coder_budget_exhausted_forces_reflection():
    """V6.1 修复：coder 失败 + budget 耗尽 + result_paths 空 → 仍前进到 reflection，
    让 route_after_reflection 处理 collect_artifacts，避免死循环。
    注意：route_after_coder 现在不区分 budget 是否耗尽，统一返回 reflection。
    budget 控制由 reflection_node + route_after_reflection 负责。
    """
    from modeling_assistant.graph.routing import route_after_coder

    # budget 耗尽 + 失败 3 次 + result_paths 空 → reflection
    state = {
        "control": ControlState(
            coder_error_count=3,
            coder_rollback_target="architect",
            modeling_revision_count=4,
            modeling_revision_budget=4,
        ),
        "artifacts": ArtifactBundle(result_paths=[]),
    }
    assert route_after_coder(state) == "reflection"

    # budget 未耗尽 + 失败 3 次 + result_paths 空 → 也 reflection
    state["control"] = ControlState(
        coder_error_count=3,
        coder_rollback_target="architect",
        modeling_revision_count=2,
        modeling_revision_budget=4,
    )
    assert route_after_coder(state) == "reflection"


def test_route_after_reflection_empty_results_returns_to_architect():
    """V6 修复（问题 B）：coder 失败（result_paths 空）+ budget 未耗尽 → 回退到 architect 重试，
    而非前进到 writer 生成不完整论文。
    """
    from modeling_assistant.graph.routing import route_after_reflection

    # coder 失败 + budget 未耗尽 → 回退到 coder_rollback_target
    state = {
        "control": ControlState(
            coder_rollback_target="architect",
            modeling_revision_count=1,
            modeling_revision_budget=4,
        ),
        "artifacts": ArtifactBundle(result_paths=[]),
    }
    assert route_after_reflection(state) == "architect"

    # coder 失败 + budget 未耗尽 + coder_rollback_target=clarifier → 回退到 clarifier
    state["control"] = ControlState(
        coder_rollback_target="clarifier",
        modeling_revision_count=1,
        modeling_revision_budget=4,
    )
    assert route_after_reflection(state) == "clarifier"


def test_route_after_reflection_empty_results_budget_exhausted_triggers_hitl():
    """budget 耗尽 + coder 失败（result_paths 空）→ 触发 HITL modeling，
    由人类决断（accept/retry/redirect），而非直接产出"待验证"论文。
    """
    from modeling_assistant.graph.routing import route_after_reflection

    state = {
        "control": ControlState(
            coder_rollback_target="architect",
            modeling_revision_count=4,
            modeling_revision_budget=4,
        ),
        "artifacts": ArtifactBundle(result_paths=[]),
    }
    assert route_after_reflection(state) == "hitl_modeling"


def test_route_after_reflection_trigger_clarifier_budget_exhausted_triggers_hitl():
    """budget 耗尽 + trigger_clarifier_revision=True → 触发 HITL modeling，
    而非回 clarifier 或直接产出"待验证"论文。
    """
    from modeling_assistant.graph.routing import route_after_reflection

    state = {
        "control": ControlState(
            trigger_clarifier_revision=True,
            modeling_revision_count=4,
            modeling_revision_budget=4,
        ),
        "artifacts": ArtifactBundle(result_paths=[]),
    }
    assert route_after_reflection(state) == "hitl_modeling"


def test_route_after_reflection_with_results_goes_to_collect_artifacts():
    """正常路径：coder 成功（result_paths 非空）→ collect_artifacts → writer。"""
    from modeling_assistant.graph.routing import route_after_reflection

    state = {
        "control": ControlState(),
        "artifacts": ArtifactBundle(result_paths=["outputs/results/output.csv"]),
    }
    assert route_after_reflection(state) == "collect_artifacts"


def test_route_after_reflection_trigger_clarifier_goes_to_clarifier():
    """trigger_clarifier_revision=True + budget 未耗尽 → clarifier（保留原逻辑）。"""
    from modeling_assistant.graph.routing import route_after_reflection

    state = {
        "control": ControlState(
            trigger_clarifier_revision=True,
            modeling_revision_count=1,
            modeling_revision_budget=4,
        ),
        "artifacts": ArtifactBundle(result_paths=[]),
    }
    assert route_after_reflection(state) == "clarifier"


def test_route_after_final_review_goes_to_rollback():
    """终稿 HITL 的 retry 应先进入 rollback 节点 checkout 版本。"""
    from modeling_assistant.graph.routing import route_after_final_review

    state = {"control": ControlState(rollback_to_version="v1.0")}
    assert route_after_final_review(state) == "rollback"

    state = {"control": ControlState()}
    assert route_after_final_review(state) == "hitl_final"

    # V15：rewrite 决策（paper_rewrite_requested）应回 Writer 重写论文
    state = {"control": ControlState(phase="paper_rewrite_requested")}
    assert route_after_final_review(state) == "writer"


def test_route_after_rollback_respects_source():
    """根据 rollback_source 决定回滚去向。"""
    from modeling_assistant.graph.routing import route_after_rollback

    state = {"control": ControlState(rollback_source="final_hitl")}
    assert route_after_rollback(state) == "mathematician"

    state = {"control": ControlState(rollback_source="architecture_hitl")}
    assert route_after_rollback(state) == "architect"

    state = {"control": ControlState(rollback_source="arbitration")}
    assert route_after_rollback(state) == "architect"


def test_route_after_architecture_hitl_revise_goes_to_clarifier():
    """V18：架构 HITL 人类打回假设（revise）→ 回 Clarifier 重新提炼。"""
    from modeling_assistant.graph.routing import route_after_architecture_hitl

    assert (
        route_after_architecture_hitl(
            {"control": ControlState(phase="architecture_revised")}
        )
        == "clarifier"
    )
    assert (
        route_after_architecture_hitl(
            {"control": ControlState(rollback_to_version="v1.0")}
        )
        == "rollback"
    )
    assert (
        route_after_architecture_hitl(
            {"control": ControlState(phase="architecture_approved")}
        )
        == "architect"
    )


def test_route_after_architecture_hitl_plan_switch_goes_to_clarifier():
    """V23：方案池中改选其他方案 → 回 Clarifier 按新方案重新提炼 LTM。"""
    from modeling_assistant.graph.routing import route_after_architecture_hitl

    assert (
        route_after_architecture_hitl(
            {"control": ControlState(phase="architecture_plan_switched")}
        )
        == "clarifier"
    )


def test_route_after_hitl_modeling_routes_by_decision():
    """HITL modeling 后根据人类决策路由：accept→collect_artifacts, retry→architect, redirect→mathematician。"""
    from modeling_assistant.graph.routing import route_after_hitl_modeling

    # accept → collect_artifacts
    state = {"control": ControlState(phase="hitl_modeling_accepted")}
    assert route_after_hitl_modeling(state) == "collect_artifacts"

    # retry → architect
    state = {"control": ControlState(phase="hitl_modeling_retry")}
    assert route_after_hitl_modeling(state) == "architect"

    # redirect → mathematician
    state = {"control": ControlState(phase="hitl_modeling_redirect")}
    assert route_after_hitl_modeling(state) == "mathematician"

    # 未知 phase → 兜底 collect_artifacts
    state = {"control": ControlState(phase="unknown")}
    assert route_after_hitl_modeling(state) == "collect_artifacts"


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


def test_combined_sub_ltms_dedupes_assumptions():
    """V20：合并多小题 LTM 时假设按原文去重（保序），避免全文假设重复。"""
    from modeling_assistant.agents.nodes import _combined_sub_ltms

    control = ControlState(
        sub_ltms=[
            DynamicLTM(assumptions=["【全文】数据真实可靠", "【问题1】浓度建模"]),
            DynamicLTM(assumptions=["【全文】数据真实可靠", "【问题2】时序建模"]),
        ]
    )
    combined = _combined_sub_ltms(control)
    assert combined.assumptions == [
        "【全文】数据真实可靠",
        "【问题1】浓度建模",
        "【问题2】时序建模",
    ]
