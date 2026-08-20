"""V18 承重结构分析测试：规则、契约、路由、注入与验收。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.agents.nodes import load_bearing_analyzer_node
from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.analysis.load_bearing import (
    build_load_bearing_map,
    reconcile_load_bearing_map,
    symbol_registry,
)
from modeling_assistant.config.settings import AppSettings
from modeling_assistant.graph.routing import route_after_milestone_reviewer_1
from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.responses import (
    LoadBearingAnalysisResponse,
    LoadBearingConclusion,
    LoadBearingConstruct,
)
from modeling_assistant.schemas.state import (
    ArtifactBundle,
    ConclusionItem,
    ConstructItem,
    ControlState,
    DataProfile,
    DynamicLTM,
    EmpiricalFinding,
    EmpiricalLayer,
    LoadBearingMap,
    ProblemFact,
    StaticLTM,
    VerificationContract,
    merge_artifacts_reducer,
)
from modeling_assistant.validation.paper_check import check_paper


def _ltm() -> DynamicLTM:
    return DynamicLTM(
        assumptions=["假设数据线性", "关键参数 M 决定判定"],
        nomenclature={"M": "有意义性度量", "K": "类别数"},
        equations=["M = IoU(T, S)", "y = f(M, K)"],
        objective="判断方案是否可行",
        solution_outline="用度量 M 判定可行域",
    )


def test_symbol_registry_extracts_nomenclature_and_equations():
    symbols = symbol_registry(_ltm())
    assert {"M", "K"}.issubset(symbols)
    assert "IoU" in symbols


def test_rule_only_fallback_builds_conservative_map():
    m = build_load_bearing_map(_ltm(), None, None, ControlState(), response=None)
    assert m.analysis_incomplete is True
    assert m.constructs
    assert {c.verification_status for c in m.constructs} <= {"unverified", "self_set"}
    assert m.anchor_gaps
    assert m.contract.required_items


def test_machine_verified_via_problem_facts():
    static = StaticLTM(
        problem_facts=[ProblemFact(value=3.0, unit="m/s", context="速度 M 为 3 m/s")]
    )
    m = build_load_bearing_map(_ltm(), static, None, ControlState(), response=None)
    m_item = next(c for c in m.constructs if c.construct == "M")
    assert m_item.verification_status == "machine_verified"


def test_evidence_linked_via_empirical():
    empirical = EmpiricalLayer(
        findings=[
            EmpiricalFinding(
                id="f1",
                run_id="run_3",
                source_node="reflection",
                assumption_tested="度量 M 的区分度",
                evidence="M 对案例区分度不足",
                verdict="refuted",
                confidence=0.8,
            )
        ]
    )
    m = build_load_bearing_map(_ltm(), None, empirical, ControlState(), response=None)
    m_item = next(c for c in m.constructs if c.construct == "M")
    assert m_item.verification_status == "evidence_linked"
    assert "run_3" in m_item.evidence_run_ids


def test_anchor_detected_via_data_columns():
    static = StaticLTM()
    static.data_profile = DataProfile()
    # data_intelligence 命中即锚点
    static.data_intelligence = ["K 列表示类别编号"]
    m = build_load_bearing_map(_ltm(), static, None, ControlState(), response=None)
    k_item = next(c for c in m.constructs if c.construct == "K")
    assert k_item.physical_anchor


def test_load_bearing_priority_root_first():
    response = LoadBearingAnalysisResponse(
        constructs=[
            LoadBearingConstruct(
                construct="根度量", construct_type="metric", is_root=True,
                required_experiment="calibration",
            ),
            LoadBearingConstruct(
                construct="显眼参数", construct_type="parameter", is_root=False,
                physical_anchor="题面给定参数",
                required_experiment="perturbation",
            ),
        ],
        conclusions=[
            LoadBearingConclusion(
                question_ref="判定可行性", answer_type="verdict",
                verdict_shape="all_negative", construct_refs=["根度量"],
            )
        ],
    )
    m = build_load_bearing_map(_ltm(), None, None, ControlState(), response=response)
    by_name = {c.construct: c for c in m.constructs}
    assert by_name["根度量"].is_root is True
    assert m.root_gaps == ["根度量"]
    assert m.contract.priority_order[0] == by_name["根度量"].id
    conclusion = m.conclusions[0]
    assert conclusion.fallback_required is True
    assert "交叉" in conclusion.fallback_spec


def test_one_sided_conclusion_forces_fallback():
    response = LoadBearingAnalysisResponse(
        constructs=[],
        conclusions=[
            LoadBearingConclusion(
                question_ref="全部不可行", answer_type="verdict",
                verdict_shape="all_negative",
            )
        ],
    )
    m = build_load_bearing_map(_ltm(), None, None, ControlState(), response=response)
    assert m.conclusions[0].fallback_required is True
    assert m.shape_risks


def test_reconcile_merges_evidence():
    m = LoadBearingMap(
        constructs=[
            ConstructItem(
                id="c1", construct="M", verification_status="unverified",
                required_experiment="calibration",
            )
        ],
        contract=VerificationContract(acceptance_anchors={"c1": "8_sensitivity.tex"}),
    )
    empirical = EmpiricalLayer(
        findings=[
            EmpiricalFinding(
                id="f1", run_id="run_5", source_node="coder",
                assumption_tested="度量 M", evidence="校准结果通过",
                verdict="confirmed", confidence=0.9,
            )
        ]
    )
    updated = reconcile_load_bearing_map(m, empirical)
    assert updated.constructs[0].verification_status == "evidence_linked"
    assert updated.constructs[0].evidence_run_ids == ["run_5"]


def test_merge_artifacts_reducer_replaces_load_bearing_map():
    map_a = LoadBearingMap(ltm_version="v1.0")
    map_b = LoadBearingMap(ltm_version="v1.1")
    merged = merge_artifacts_reducer(
        ArtifactBundle(load_bearing_map=map_a),
        ArtifactBundle(load_bearing_map=map_b),
    )
    assert merged.load_bearing_map.ltm_version == "v1.1"


def test_route_after_milestone_reviewer_1_goes_to_analyzer():
    assert route_after_milestone_reviewer_1({"control": ControlState()}) == "load_bearing_analyzer"
    state = {"control": ControlState(need_rebrainstorm=True)}
    assert route_after_milestone_reviewer_1(state) == "mathematician"
    state = {
        "control": ControlState(
            need_rebrainstorm=True,
            modeling_revision_count=4,
            modeling_revision_budget=4,
        )
    }
    assert route_after_milestone_reviewer_1(state) == "hitl_architecture"


def _make_paper_for_contract(tmp_path: Path, sensitivity_text: str = "") -> Path:
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{ctexart}\n\\begin{document}\n"
        "\\input{sections/8_sensitivity}\n\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "sections" / "8_sensitivity.tex").write_text(
        f"\\section{{敏感性分析}}\n{sensitivity_text}\n",
        encoding="utf-8",
    )
    return paper


def _map_with_required_item() -> LoadBearingMap:
    return LoadBearingMap(
        constructs=[
            ConstructItem(
                id="c1", construct="根构造", is_root=True,
                verification_status="unverified", physical_anchor="题面对象",
                required_experiment="perturbation",
            )
        ],
        contract=VerificationContract(
            priority_order=["c1"],
            required_items=[
                ConstructItem(
                    id="c1", construct="根构造", is_root=True,
                    verification_status="unverified", physical_anchor="题面对象",
                    required_experiment="perturbation",
                )
            ],
            acceptance_anchors={"c1": "8_sensitivity.tex"},
        ),
    )


def test_paper_check_contract_fails_when_anchor_section_missing(tmp_path):
    paper = _make_paper_for_contract(tmp_path, "")
    report = check_paper(
        paper, compile_pdf=False, load_bearing_map=_map_with_required_item()
    )
    assert not report["passed"]
    assert any("承重契约" in i and "扰动" in i for i in report["issues"])


def test_paper_check_contract_passes_when_satisfied(tmp_path):
    paper = _make_paper_for_contract(tmp_path, "对根构造做 ±20% 扰动扫描，结论保持成立。")
    report = check_paper(
        paper, compile_pdf=False, load_bearing_map=_map_with_required_item()
    )
    assert report["passed"], report["issues"]
    assert report["checks"]["承重契约"] == "通过"


def test_prompt_injection_load_bearing():
    m = _map_with_required_item()
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            dynamic_ltm=_ltm(),
            control=ControlState(),
            artifacts=ArtifactBundle(load_bearing_map=m),
            extra={
                "integrity_warnings": "无",
                "method_knowledge_enabled": False,
            },
        ),
    )
    assert "承重结构分析" in prompt
    assert "load_bearing_active=true" in prompt
    assert "priority_order" in prompt


def test_analyzer_node_with_mocked_llm(tmp_path, monkeypatch):
    runtime = AgentRuntime.from_settings(
        AppSettings(
            output_dir=tmp_path,
            api_key_env="MISSING_KEY_FOR_TEST",
            search_enabled=False,
        )
    )

    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        assert name == "load_bearing_analyzer"
        return LoadBearingAnalysisResponse(
            constructs=[
                LoadBearingConstruct(
                    construct="度量 M", construct_type="metric", is_root=True,
                    required_experiment="calibration",
                )
            ],
            conclusions=[
                LoadBearingConclusion(
                    question_ref="是否可行", answer_type="verdict",
                    verdict_shape="all_negative", construct_refs=["度量 M"],
                )
            ],
            reasoning="ok",
        )

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)
    state = {
        "static_ltm": StaticLTM(raw_problem="测试题"),
        "dynamic_ltm": _ltm(),
        "ltm_archive": [],
        "control": ControlState(sub_questions=["是否可行"], current_sub_question_index=0),
        "artifacts": ArtifactBundle(),
        "empirical": EmpiricalLayer(),
        "prompt_audit": {},
        "process_log": [],
    }
    result = load_bearing_analyzer_node(state, runtime=runtime)
    m = result["artifacts"].load_bearing_map
    assert m is not None
    assert m.analysis_incomplete is False
    assert m.constructs[0].is_root is True
    assert m.root_gaps == ["度量 M"]
    assert m.conclusions[0].fallback_required is True
    assert result["control"].phase == "load_bearing_analyzed"
    assert any(e.stage == "load_bearing_analyzer" for e in result["process_log"])
