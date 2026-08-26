from pathlib import Path

from modeling_assistant.config import load_settings
from modeling_assistant.prompts import PromptCatalog, PromptContext
from modeling_assistant.schemas.state import (
    ColumnProfile,
    ControlState,
    DataProfile,
    DynamicLTM,
    ProblemFact,
    StaticLTM,
)


def test_load_settings_from_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MODELING_ASSISTANT_LLM_MODEL=deepseek-test",
                "MODELING_ASSISTANT_SEARCH_ENABLED=true",
                "MODELING_ASSISTANT_MAX_DEBATE_ROUNDS=4",
                "MODELING_ASSISTANT_INNOVATION_WEIGHT=0.6",
                "MODELING_ASSISTANT_FEASIBILITY_WEIGHT=0.4",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.llm_model == "deepseek-test"
    assert settings.search_enabled is True
    assert settings.max_debate_rounds == 4
    assert settings.innovation_weight == 0.6
    assert settings.feasibility_weight == 0.4


def test_load_deepseek_api_key_and_model_from_env_file(tmp_path: Path, monkeypatch):
    """.env 中常见的 DEEPSEEK_API_KEY / DEEPSEEK_MODEL 应被正确识别。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=sk-test-key",
                "DEEPSEEK_MODEL=deepseek-v4-pro",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MODELING_ASSISTANT_LLM_MODEL", raising=False)

    settings = load_settings(env_file)

    assert settings.llm_model == "deepseek-v4-pro"
    assert settings.api_key == "sk-test-key"


def test_default_search_enabled():
    """Searcher 默认应启用真实检索。"""
    settings = load_settings()
    assert settings.search_enabled is True


def test_llm_max_tokens_overrides_from_env(monkeypatch):
    """V17：MODELING_ASSISTANT_LLM_MAX_TOKENS_OVERRIDES 应按节点覆盖输出上限。"""
    monkeypatch.setenv(
        "MODELING_ASSISTANT_LLM_MAX_TOKENS_OVERRIDES",
        '{"writer": 20000, "arbiter": 3000}',
    )
    settings = load_settings()
    assert settings.max_tokens_for("writer") == 20000
    assert settings.max_tokens_for("arbiter") == 3000
    # 未覆盖节点保留默认覆盖/全局默认
    assert settings.max_tokens_for("coder") == 32768
    assert settings.max_tokens_for("no_such_node") == settings.llm_max_tokens


def test_llm_max_tokens_overrides_defaults():
    """V17：默认覆盖表中长输出节点保持大上限，小节点被压低。"""
    settings = load_settings()
    assert settings.max_tokens_for("writer") == 32768
    assert settings.max_tokens_for("clarifier") == 24576
    assert settings.max_tokens_for("arbiter") == 4096
    assert settings.max_tokens_for("searcher") == 2048


def test_prompt_catalog_renders_ltm_without_history():
    prompt = PromptCatalog().render(
        "coder",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(
                assumptions=["只使用当前 LTM"],
                equations=["y = ax + b"],
                objective="线性拟合",
            ),
        ),
    )

    assert "不接收完整对话历史" in prompt
    assert "线性拟合" in prompt
    assert "y = ax + b" in prompt


def test_prompt_catalog_renders_branch_from_version():
    """PromptContext 应正确渲染 branch_from_version。"""
    prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(branch_from_version="v1.0"),
        ),
    )

    assert "v1.0" in prompt
    assert "分支重建" in prompt


def test_prompt_catalog_renders_realist_thresholds():
    """Realist 模板应包含阈值与候选方案。"""
    from modeling_assistant.schemas.state import PlanCandidate

    prompt = PromptCatalog().render(
        "realist",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            control=ControlState(
                innovation_threshold=70,
                feasibility_threshold=65,
                top_k_plans=[
                    PlanCandidate(
                        id="p1",
                        title="方案",
                        description="描述",
                        innovation_score=80,
                        feasibility_score=70,
                    )
                ],
            ),
        ),
    )

    assert "70" in prompt  # innovation_threshold
    assert "65" in prompt  # feasibility_threshold
    assert "方案" in prompt  # plan title


def test_prompt_catalog_renders_coder_error_log():
    """Coder Prompt 应注入历史错误日志。"""
    prompt = PromptCatalog().render(
        "coder",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(coder_error_log=["SyntaxError: invalid syntax"]),
        ),
    )
    assert "SyntaxError" in prompt
    assert "历史错误日志" in prompt


def test_prompt_catalog_renders_recent_stderr_for_self_repair():
    """Coder Prompt 在自修复模式下应注入 recent_stderr。"""
    prompt = PromptCatalog().render(
        "coder",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
            extra={"recent_stderr": "AttributeError: 'numpy.ndarray' object has no attribute 'values'"},
        ),
    )
    assert "AttributeError" in prompt
    assert "自修复" in prompt


def test_precheck_code_detects_syntax_error():
    """precheck_code 应检测语法错误。"""
    from modeling_assistant.agents.runtime import precheck_code

    # 字符串字面量跨行错误（run_2 的失败原因）
    bad_code = "x = ('hello' + '\\n' 'world' + '\\n' 'again')"
    # 构造真正的语法错误
    bad_code = "def f():\n    return 'unterminated string"
    error = precheck_code(bad_code)
    assert "SyntaxError" in error
    assert "预检拦截" in error


def test_precheck_code_detects_forbidden_imports():
    """precheck_code 应检测禁止库 import。"""
    from modeling_assistant.agents.runtime import precheck_code

    # run_1 的失败原因：imblearn
    bad_code = "from imblearn.over_sampling import SMOTE\nimport pandas as pd\ndf = pd.DataFrame()"
    error = precheck_code(bad_code)
    assert "ModuleNotFoundError" in error
    assert "imblearn" in error
    assert "预检拦截" in error


def test_precheck_code_allows_permitted_imports():
    """precheck_code 应允许已安装的库。"""
    from modeling_assistant.agents.runtime import precheck_code

    good_code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "from sklearn.utils import resample\n"
        "df = pd.DataFrame({'x': [1, 2, 3]})\n"
    )
    error = precheck_code(good_code)
    assert error == ""


def test_precheck_code_detects_multiple_forbidden_imports():
    """precheck_code 应一次检测多个禁止库。"""
    from modeling_assistant.agents.runtime import precheck_code

    bad_code = (
        "import xgboost as xgb\n"
        "import shap\n"
        "from imblearn.over_sampling import SMOTE\n"
    )
    error = precheck_code(bad_code)
    assert "xgboost" in error
    assert "shap" in error
    assert "imblearn" in error


def test_merge_artifacts_reducer_clears_placeholder_on_real_figures():
    """merge_artifacts_reducer 在 incoming 含真实图片时应移除 base 中的 placeholder。"""
    from modeling_assistant.schemas.state import ArtifactBundle, merge_artifacts_reducer

    # base 含 placeholder（第一次 drawer 失败残留）
    base = ArtifactBundle(figure_paths=["outputs/figures/placeholder.png"])
    # incoming 含真实图片（第二次 drawer 成功）
    incoming = ArtifactBundle(figure_paths=["outputs/figures/figure1.png", "outputs/figures/figure2.png"])

    merged = merge_artifacts_reducer(base, incoming)
    assert "placeholder.png" not in str(merged.figure_paths)
    assert "figure1.png" in str(merged.figure_paths)
    assert "figure2.png" in str(merged.figure_paths)


def test_merge_artifacts_reducer_keeps_placeholder_when_no_real_figures():
    """merge_artifacts_reducer 在 incoming 只有 placeholder 时应保留 base 的 placeholder。"""
    from modeling_assistant.schemas.state import ArtifactBundle, merge_artifacts_reducer

    base = ArtifactBundle(figure_paths=["outputs/figures/placeholder.png"])
    incoming = ArtifactBundle(figure_paths=["outputs/figures/placeholder.png"])

    merged = merge_artifacts_reducer(base, incoming)
    # 仍是 placeholder（去重后）
    assert len(merged.figure_paths) == 1
    assert "placeholder" in merged.figure_paths[0].lower()


def test_merge_artifacts_reducer_clears_result_paths_on_flag():
    """V9 修复：incoming.clear_result_paths=True 时应清空 base.result_paths。

    场景：coder 第一次成功产出 result_paths=['output.csv']，第二次失败返回
    result_paths=[] 且 clear_result_paths=True。merge 后 result_paths 应为空，
    避免路由错乱（route_after_coder 误判为成功）和 writer 误用旧结果。
    """
    from modeling_assistant.schemas.state import ArtifactBundle, merge_artifacts_reducer

    base = ArtifactBundle(result_paths=["outputs/results/output.csv"])
    incoming = ArtifactBundle(result_paths=[], clear_result_paths=True)

    merged = merge_artifacts_reducer(base, incoming)
    assert merged.result_paths == []


def test_merge_artifacts_reducer_preserves_result_paths_without_flag():
    """V9 修复：incoming.clear_result_paths=False（默认）时应保留追加语义。

    场景：drawer 不设置 clear_result_paths，只更新 figure_paths，不应影响
    coder 之前写入的 result_paths。
    """
    from modeling_assistant.schemas.state import ArtifactBundle, merge_artifacts_reducer

    base = ArtifactBundle(result_paths=["outputs/results/output.csv"])
    incoming = ArtifactBundle(figure_paths=["outputs/figures/figure1.png"])

    merged = merge_artifacts_reducer(base, incoming)
    assert "outputs/results/output.csv" in merged.result_paths
    assert "outputs/figures/figure1.png" in merged.figure_paths


def test_route_after_coder_failed_phase_goes_to_reflection():
    """V9 修复：route_after_coder 在失败 phase 时应返回 'reflection'，不论 result_paths 状态。

    场景：coder_node 失败时 control.phase='code_execution_failed'，但 merge_artifacts_reducer
    可能因旧 state 仍含 result_paths（虽然在 V9 中已通过 clear_result_paths 清空，但双重保险）。
    """
    from modeling_assistant.graph.routing import route_after_coder
    from modeling_assistant.schemas.state import (
        ArtifactBundle,
        ControlState,
        GraphState,
    )

    # 即使 result_paths 非空，失败 phase 也应走 reflection
    state: GraphState = {
        "control": ControlState(phase="code_execution_failed"),
        "artifacts": ArtifactBundle(result_paths=["outputs/results/output.csv"]),
    }
    assert route_after_coder(state) == "reflection"


def test_route_after_coder_success_phase_goes_to_result_reviewer():
    """V9 修复：route_after_coder 在非失败 phase 且 result_paths 非空时走 result_reviewer。"""
    from modeling_assistant.graph.routing import route_after_coder
    from modeling_assistant.schemas.state import (
        ArtifactBundle,
        ControlState,
        GraphState,
    )

    state: GraphState = {
        "control": ControlState(phase="code_executed_successfully"),
        "artifacts": ArtifactBundle(result_paths=["outputs/results/output.csv"]),
    }
    assert route_after_coder(state) == "result_reviewer"


def test_route_after_result_reviewer_routes_to_acceptance_on_pass():
    """V14：ResultReviewer 通过 → 小题人工验收；失败 → reflection 消费预算。"""
    from modeling_assistant.graph.routing import route_after_result_reviewer
    from modeling_assistant.schemas.state import ControlState, GraphState

    # 失败 phase
    state_failed: GraphState = {
        "control": ControlState(phase="result_review_failed"),
    }
    assert route_after_result_reviewer(state_failed) == "reflection"

    # 成功 phase：V14 改为进入小题人工验收闸门
    state_passed: GraphState = {
        "control": ControlState(phase="result_review_passed"),
    }
    assert route_after_result_reviewer(state_passed) == "sub_question_acceptance"


def test_drawer_prompt_renders_recent_stderr_for_self_repair():
    """V9 修复：Drawer Prompt 在自修复模式下应注入 recent_stderr。"""
    prompt = PromptCatalog().render(
        "drawer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
            extra={"recent_stderr": "KeyError: 'sex'"},
        ),
    )
    assert "KeyError" in prompt
    assert "自修复" in prompt
    assert "列名" in prompt


def test_prompt_catalog_renders_rebrainstorm_feedback():
    """Mathematician Prompt 应注入 Milestone Reviewer 1 的反馈。"""
    prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(rebrainstorm_feedback=["假设列表为空。"]),
        ),
    )
    assert "假设列表为空" in prompt


def test_architect_prompt_renders_last_result_review_issues():
    """V10 修复：Architect Prompt 在 last_result_review_issues 非空时应注入拒绝原因。"""
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(
                last_result_review_issues=[
                    "[output.csv] 数值列 'optimal_day' 为常量，无区分信息。",
                ],
            ),
        ),
    )
    assert "ResultReviewer 拒绝原因" in prompt
    assert "optimal_day" in prompt
    assert "常量列" in prompt  # 针对性策略关键词


def test_architect_prompt_no_result_review_issues_when_empty():
    """V10 修复：Architect Prompt 在 last_result_review_issues 为空时应注入空数组。"""
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "ResultReviewer 拒绝原因" in prompt
    # 空时也应渲染（[]），但不应触发"V10 修复：ResultReviewer 拒绝处理策略"的具体策略
    # 不过策略部分会始终渲染，只是 last_result_review_issues_json 为 []


def test_writer_prompt_renders_result_preview():
    """V10 修复：Writer Prompt 在 result_preview 非空时应注入真实数值预览。"""
    preview = "=== 结果文件 output.csv ===\n形状: 3 行 × 4 列\n列名: ['group', 'optimal_week']\n数据：..."
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
            extra={"result_preview": preview, "integrity_warnings": "无"},
        ),
    )
    assert "结果文件预览" in prompt
    assert "optimal_week" in prompt


def test_prompt_catalog_data_columns_grouped_by_file():
    """V16：data_columns_json 应按文件分组并含样例值，而非扁平合并。"""
    from modeling_assistant.schemas.state import ColumnProfile, DataProfile, FileSummary

    profile = DataProfile(
        file_summaries=[
            FileSummary(
                path="订单信息.xlsx",
                rows=3,
                cols=2,
                columns=[
                    ColumnProfile(name="客户编号", dtype="int", sample_values=["1", "2"]),
                    ColumnProfile(name="需求量(kg)", dtype="float", sample_values=["12.5"]),
                ],
            ),
            FileSummary(
                path="距离矩阵.xlsx",
                rows=2,
                cols=2,
                columns=[
                    ColumnProfile(name="0", dtype="float"),
                    ColumnProfile(name="1", dtype="float"),
                ],
            ),
        ]
    )
    static = StaticLTM(raw_problem="测试", data_profile=profile)
    prompt = PromptCatalog().render(
        "coder",
        PromptContext(static_ltm=static, dynamic_ltm=DynamicLTM()),
    )
    import json
    import re

    # 从渲染结果中提取 data_columns_json 内容
    match = re.search(r'"file": "订单信息\.xlsx"', prompt)
    assert match, "列名清单应按文件分组，包含 订单信息.xlsx"
    assert "距离矩阵.xlsx" in prompt
    assert "客户编号" in prompt
    # 扁平合并时的特征不应出现（旧的 [{"name":...}] 顶层结构）
    assert '"file"' in prompt


def test_writer_prompt_handles_empty_result_preview():
    """V10 修复：Writer Prompt 在 result_preview 为空时应正常渲染（不报 KeyError）。"""
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="问题"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
            extra={"integrity_warnings": "无"},
        ),
    )
    # 即使 result_preview 为空，模板也应能正常渲染
    assert "结果文件预览" in prompt


def test_result_reviewer_node_sets_last_result_review_issues():
    """V10 修复：result_reviewer_node 失败时应填充 last_result_review_issues。"""
    from modeling_assistant.schemas.state import ArtifactBundle
    from modeling_assistant.validation.results import result_reviewer_node

    # 构造一个会触发 ResultReviewer 拒绝的 state：result_paths 指向一个常量列 CSV
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "output.csv"
        # 常量列 CSV：所有行 optimal_week 都是 10.0
        csv_path.write_text("group,optimal_week\nA,10.0\nB,10.0\nC,10.0\n", encoding="utf-8")

        from modeling_assistant.schemas.state import GraphState
        state: GraphState = {
            "static_ltm": StaticLTM(raw_problem="test"),
            "dynamic_ltm": DynamicLTM(),
            "ltm_archive": [],
            "control": ControlState(),
            "artifacts": ArtifactBundle(result_paths=[str(csv_path)]),
            "prompt_audit": {},
        }
        result = result_reviewer_node(state)
        # 失败时应填充 last_result_review_issues
        assert result["control"].last_result_review_issues
        assert any("常量" in issue for issue in result["control"].last_result_review_issues)
        # V10 修复：强制走 architect（而非根据文本判断 architect/clarifier）
        assert result["control"].coder_rollback_target == "architect"


def test_coder_node_backs_up_successful_results(tmp_path: Path, monkeypatch):
    """V10 修复：coder_node 成功执行时应备份 output.csv 为 output_run_N.csv。"""
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.agents.nodes import coder_node
    from modeling_assistant.schemas.responses import CoderResponse
    from modeling_assistant.schemas.state import ArtifactBundle, GraphState

    # 准备临时输出目录
    output_dir = tmp_path / "outputs"
    (output_dir / "results").mkdir(parents=True)
    # 使用 POSIX 路径避免 Windows 反斜杠被解释为 Unicode 转义
    output_dir_posix = output_dir.as_posix()

    # 构造一个会成功的 Coder 代码：生成 output.csv
    code = (
        "import pandas as pd\n"
        "from pathlib import Path\n"
        f"df = pd.DataFrame({{'group': ['A', 'B', 'C'], 'optimal_week': [13.0, 15.0, 17.0]}})\n"
        f"df.to_csv(Path(r'{output_dir_posix}/results/output.csv'), index=False)\n"
    )

    # Mock runtime
    runtime = AgentRuntime.from_settings(
        AppSettings(
            output_dir=output_dir,
            api_key_env="MISSING_KEY_FOR_TEST",
            coder_external_mode="builtin",
        )
    )

    # Mock invoke_structured 返回 CoderResponse
    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        return CoderResponse(
            code=code,
            result_path=f"{output_dir_posix}/results/output.csv",
        )

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)

    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(coder_run_count=5),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    result = coder_node(state, runtime=runtime)

    # 验证备份文件已创建
    backup_path = output_dir / "results" / "output_run_5.csv"
    assert backup_path.exists(), f"备份文件应存在: {backup_path}"
    # 验证 result_paths 已设置
    assert result["artifacts"].result_paths
    assert "output.csv" in result["artifacts"].result_paths[0]


def test_writer_node_loads_backup_when_result_paths_empty(tmp_path: Path, monkeypatch):
    """V10 修复：writer_node 在 result_paths 为空时应扫描并加载最新的 output_run_*.csv 备份。"""
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.agents.nodes import writer_node
    from modeling_assistant.schemas.responses import WriterResponse
    from modeling_assistant.schemas.state import ArtifactBundle, GraphState

    output_dir = tmp_path / "outputs"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True)

    # 创建两个备份文件，run_2 应该是更新的（编号更大）
    (results_dir / "output_run_1.csv").write_text(
        "group,week\nA,13.0\nB,15.0\n", encoding="utf-8"
    )
    (results_dir / "output_run_2.csv").write_text(
        "group,week\nA,14.0\nB,16.0\nC,18.0\n", encoding="utf-8"
    )

    runtime = AgentRuntime.from_settings(
        AppSettings(output_dir=output_dir, api_key_env="MISSING_KEY_FOR_TEST")
    )

    # Mock invoke_structured 返回 WriterResponse
    captured_prompt = []

    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        captured_prompt.append(system_prompt)
        return WriterResponse(latex_content="\\documentclass{article}\nTest\n")

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)

    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(objective="测试目标", assumptions=["A1"], equations=["y=ax"]),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),  # result_paths 为空
        "prompt_audit": {},
    }
    result = writer_node(state, runtime=runtime)

    # 验证 writer 加载了最新的备份（run_2）
    assert result["artifacts"].result_paths
    assert "output_run_2.csv" in result["artifacts"].result_paths[0]
    # 验证 prompt 中包含真实数值预览
    assert len(captured_prompt) == 1
    assert "16.0" in captured_prompt[0] or "18.0" in captured_prompt[0]


# ═════════════════════════════════════════════════════════════════════
# V11 三层防线单元测试
# ═════════════════════════════════════════════════════════════════════


# ── 第一层：fact_extractor 纯机器提取 ──────────────────────────────

def test_extract_facts_from_problem_detects_units():
    """V11 第一层：应从题目原文提取带单位的数值常量。"""
    from modeling_assistant.data.facts import extract_facts_from_problem

    raw = (
        "导弹飞行速度300 m/s，云团以3 m/s的速度匀速下沉，"
        "云团中心10 m范围内，起爆20 s内可提供有效遮蔽。"
        "无人机速度70~140 m/s，每架无人机投放两枚至少间隔1 s。"
    )
    facts = extract_facts_from_problem(raw)

    # 至少提取到这些关键常量
    values_units = {(f.value, f.unit) for f in facts}
    assert (300.0, "m/s") in values_units
    assert (3.0, "m/s") in values_units
    assert (10.0, "m") in values_units
    assert (20.0, "s") in values_units
    # 上下文应包含原文片段
    has_3ms = any(f.value == 3.0 and "3 m/s" in f.context for f in facts)
    assert has_3ms


def test_extract_facts_dedup_same_value_unit():
    """V11 第一层：相同 (value, unit) 只保留第一次。"""
    from modeling_assistant.data.facts import extract_facts_from_problem

    raw = "速度3 m/s，再次提到3 m/s，还有3 m/s。"
    facts = extract_facts_from_problem(raw)
    # 去重后只有一个 3 m/s
    count_3ms = sum(1 for f in facts if f.value == 3.0 and f.unit == "m/s")
    assert count_3ms == 1


def test_fact_extractor_node_writes_to_static_ltm():
    """V11 第一层：fact_extractor_node 应把 facts 写入 static_ltm.problem_facts。"""
    from modeling_assistant.agents.nodes import fact_extractor_node
    from modeling_assistant.schemas.state import GraphState

    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="速度3 m/s，高度10 m。"),
        "control": ControlState(),
    }
    result = fact_extractor_node(state)
    assert result["control"].phase == "facts_extracted"
    facts = result["static_ltm"].problem_facts
    assert len(facts) >= 2
    values = {f.value for f in facts}
    assert 3.0 in values
    assert 10.0 in values


# ── 第一层：parse_hint 字符串列解析建议 ────────────────────────────

def test_infer_parse_hint_for_week_string():
    """V11 第一层：'16W' 样例应生成 str.replace('W','').astype(float) 提示。"""
    from modeling_assistant.data.facts import infer_parse_hint

    col = ColumnProfile(name="孕周", dtype="text", sample_values=["16W", "17W", "18W"])
    hint = infer_parse_hint(col)
    assert "str.replace" in hint
    assert "W" in hint
    assert "astype(float)" in hint


def test_infer_parse_hint_for_date_string():
    """V11 第一层：日期字符串应生成 pd.to_datetime 提示。"""
    from modeling_assistant.data.facts import infer_parse_hint

    col = ColumnProfile(name="检测日期", dtype="text", sample_values=["2023-01-15", "2023-02-20"])
    hint = infer_parse_hint(col)
    assert "pd.to_datetime" in hint


def test_infer_parse_hint_for_percent():
    """V11 第一层：'95%' 应生成 rstrip('%').astype(float) / 100 提示。"""
    from modeling_assistant.data.facts import infer_parse_hint

    col = ColumnProfile(name="命中率", dtype="text", sample_values=["95%", "87%"])
    hint = infer_parse_hint(col)
    assert "rstrip" in hint
    assert "%" in hint


def test_data_loader_fills_parse_hints(tmp_path: Path):
    """V11 第一层：data_profile_node 应自动填充 parse_hint。"""
    from modeling_assistant.data.loader import data_profile_node
    from modeling_assistant.schemas.state import GraphState

    # 构造含字符串列的 CSV
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "孕周,类型\n16W,A\n17W,B\n18W,A\n",
        encoding="utf-8",
    )

    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config import load_settings
    settings = load_settings()
    settings.output_dir = tmp_path / "out"
    runtime = AgentRuntime.from_settings(settings)

    state: GraphState = {
        "static_ltm": StaticLTM(
            raw_problem="test",
            data_attachments=[str(csv_path)],
        ),
        "control": ControlState(),
    }
    result = data_profile_node(state, runtime=runtime)
    profile = result["static_ltm"].data_profile
    assert profile is not None
    # 找到孕周列，检查 parse_hint
    week_col = next(c for c in profile.columns if c.name == "孕周")
    assert week_col.parse_hint
    assert "W" in week_col.parse_hint


def test_data_profile_node_reclassifies_facts_v11_4(tmp_path: Path):
    """V11.4：data_profile_node 应在数据画像完成后重新分类 problem_facts 的 category。

    场景：fact_extractor 在 data_profile 之前跑，此时 columns=None，
    "GC 含量正常范围 40%-60%" 被标为 physical_param。
    data_profile_node 完成后应基于真实列名重新分类为 data_range。
    """
    from modeling_assistant.data.loader import data_profile_node
    from modeling_assistant.schemas.state import GraphState

    # 构造含 GC含量 列的 CSV
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("GC含量\n0.5\n0.55\n0.45\n", encoding="utf-8")

    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config import load_settings
    settings = load_settings()
    settings.output_dir = tmp_path / "out"
    runtime = AgentRuntime.from_settings(settings)

    # 模拟 fact_extractor 跑完后（columns=None，所有 % 单位 fact 都是 physical_param）
    state: GraphState = {
        "static_ltm": StaticLTM(
            raw_problem="GC 含量正常范围为40% ~ 60%",
            data_attachments=[str(csv_path)],
            problem_facts=[
                ProblemFact(
                    value=40.0, unit="%",
                    context="重要指标，正常 GC 含量范围为40% ~ 60%",
                    category="physical_param",  # 初始分类（无列名时）
                ),
                ProblemFact(
                    value=60.0, unit="%",
                    context="正常 GC 含量范围为40% ~ 60%",
                    category="physical_param",  # 初始分类（无列名时）
                ),
            ],
        ),
        "control": ControlState(),
    }
    result = data_profile_node(state, runtime=runtime)
    facts = result["static_ltm"].problem_facts
    # V11.4 重新分类后应标为 data_range
    assert all(f.category == "data_range" for f in facts), \
        f"data_profile_node 应重新分类 GC 范围 fact 为 data_range，实际：{[(f.value, f.category) for f in facts]}"


# ── 第二层：LTM 写入前常量校验 ────────────────────────────────────

def test_check_ltm_against_facts_detects_missing_constant():
    """V11 第二层：LTM 缺失关键常量时应告警。"""
    from modeling_assistant.validation.constants import check_ltm_against_facts

    static_ltm = StaticLTM(
        raw_problem="速度3 m/s",
        problem_facts=[
            ProblemFact(value=3.0, unit="m/s", context="云团以3 m/s的速度下沉"),
        ],
    )
    # LTM 里完全没有 3 这个数值，也没有任何 m/s 单位数值
    dynamic_ltm = DynamicLTM(
        assumptions=["云团下沉速度 v_sink = 1.0"],
        equations=["y = v_sink * t"],
        objective="计算遮蔽时间",
    )
    issues = check_ltm_against_facts(dynamic_ltm, static_ltm)
    # 应该告警关键常量未引用（唯一值物理量必须被引用）
    assert any("3.0" in i and "未在 LTM 中引用" in i for i in issues)


def test_check_ltm_against_facts_detects_wrong_value_with_unit():
    """V11.1 第二层主防线：LTM 里出现 '1.0 m/s' 但题目是 3.0 m/s 时应告警。"""
    from modeling_assistant.validation.constants import check_ltm_against_facts

    static_ltm = StaticLTM(
        raw_problem="速度3 m/s",
        problem_facts=[
            ProblemFact(value=3.0, unit="m/s", context="云团以3 m/s的速度下沉"),
        ],
    )
    # LTM 里写了 1.0 m/s（错误数值 + 正确单位）
    dynamic_ltm = DynamicLTM(
        assumptions=["云团下沉速度 v_sink = 1.0 m/s"],
        equations=["y = v_sink * t"],
        objective="计算遮蔽时间",
    )
    issues = check_ltm_against_facts(dynamic_ltm, static_ltm)
    # 主防线应告警：LTM 中出现 '1.0 m/s'，但题目常量为 [3.0] m/s
    assert any("1.0 m/s" in i and "3.0" in i for i in issues)


def test_check_ltm_against_facts_no_noise_for_math_coefficients():
    """V11.1 第二层：LTM 里的数学系数（0.5、1.0、100 等）不应产生噪音告警。"""
    from modeling_assistant.validation.constants import check_ltm_against_facts

    static_ltm = StaticLTM(
        raw_problem="速度3 m/s，高度10 m",
        problem_facts=[
            ProblemFact(value=3.0, unit="m/s", context="3 m/s"),
            ProblemFact(value=10.0, unit="m", context="10 m"),
        ],
    )
    # LTM 里有大量数学系数，但都不带物理单位
    dynamic_ltm = DynamicLTM(
        assumptions=[
            "v_sink = 3.0 m/s（原文：3 m/s）",  # 正确引用
            "遮蔽高度 h = 10.0 m（原文：10 m）",  # 正确引用
            "权重 w1 = 0.5",  # 数学系数，不带单位
            "权重 w2 = 0.5",
            "迭代次数 N = 100",
            "收敛阈值 eps = 1e-6",
        ],
        equations=[
            "y = 3.0 * t + 0.5 * x",  # 3.0 引用了，0.5 是系数
            "loss = 0.5 * (y - y_pred) ** 2",
        ],
        objective="计算遮蔽时间",
    )
    issues = check_ltm_against_facts(dynamic_ltm, static_ltm)
    # 不应有任何"数值冲突"告警（数学系数不带单位，不参与校验）
    assert not any("冲突" in i for i in issues)
    # 也不应有"关键常量未引用"告警（3.0 和 10.0 都已引用）
    assert not any("未在 LTM 中引用" in i for i in issues)


def test_check_ltm_against_facts_passes_when_constant_present():
    """V11 第二层：LTM 正确引用关键常量时应通过。"""
    from modeling_assistant.validation.constants import check_ltm_against_facts

    static_ltm = StaticLTM(
        raw_problem="速度3 m/s",
        problem_facts=[
            ProblemFact(value=3.0, unit="m/s", context="云团以3 m/s的速度下沉"),
        ],
    )
    dynamic_ltm = DynamicLTM(
        assumptions=["云团下沉速度 v_sink = 3.0 m/s（原文：云团以3 m/s的速度下沉）"],
        equations=["y = 3.0 * t"],
        objective="计算遮蔽时间",
    )
    issues = check_ltm_against_facts(dynamic_ltm, static_ltm)
    # 不应有"关键常量未在 LTM 中引用"的告警
    assert not any("未在 LTM 中引用" in i for i in issues)


def test_check_ltm_against_facts_handles_duplicate_values():
    """V11.1 第二层辅助防线：重复值（如两个 10.0 m）不强制要求引用。"""
    from modeling_assistant.validation.constants import check_ltm_against_facts

    static_ltm = StaticLTM(
        raw_problem="半径10 m，高度10 m",
        problem_facts=[
            ProblemFact(value=10.0, unit="m", context="半径10 m"),
            ProblemFact(value=10.0, unit="m", context="高度10 m"),
        ],
    )
    # LTM 里只引用了一次 10.0 m（另一个没引用）
    dynamic_ltm = DynamicLTM(
        assumptions=["目标半径 R = 10.0 m"],
        equations=["V = pi * R^2 * h"],
        objective="计算体积",
    )
    issues = check_ltm_against_facts(dynamic_ltm, static_ltm)
    # 重复值不强制要求引用，不应告警
    assert not any("未在 LTM 中引用" in i for i in issues)


def test_clarifier_node_records_constant_issues_in_audit(monkeypatch):
    """V11 第二层：clarifier_node 应把常量校验告警记录到 audit。"""
    from modeling_assistant.agents.nodes import clarifier_node
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config import load_settings
    from modeling_assistant.schemas.responses import ClarifierResponse
    from modeling_assistant.schemas.state import GraphState

    settings = load_settings()
    runtime = AgentRuntime.from_settings(settings)

    # Mock invoke_structured：返回缺少数值 3.0 的 LTM
    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        return ClarifierResponse(
            assumptions=["v_sink = 1.0 m/s"],
            nomenclature={"v_sink": "下沉速度"},
            equations=["y = v_sink * t"],
            objective="遮蔽",
            solution_outline="test",
            commit_summary="test",
        )

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)

    state: GraphState = {
        "static_ltm": StaticLTM(
            raw_problem="速度3 m/s",
            problem_facts=[ProblemFact(value=3.0, unit="m/s", context="3 m/s")],
        ),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(),
    }
    result = clarifier_node(state, runtime=runtime)
    # 应在 audit 中记录常量校验告警
    assert "clarifier_constant_issues" in result["prompt_audit"]


# ── 第三层：代码常量 AST 扫描 ──────────────────────────────────────

def test_check_code_against_facts_detects_missing_constant_in_code():
    """V11 第三层：代码缺失关键常量时应告警。"""
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="速度3 m/s",
        problem_facts=[ProblemFact(value=3.0, unit="m/s", context="3 m/s")],
    )
    # 代码里写了 1.0 而不是 3.0
    code = "v_sink = 1.0\nimport pandas as pd\ndf = pd.DataFrame({'t': [1, 2]})\ndf['y'] = v_sink * df['t']\ndf.to_csv('out.csv', index=False)"
    issues = check_code_against_facts(code, static_ltm)
    # 应告警代码常量缺失
    assert any("代码常量缺失" in i and "3.0" in i for i in issues)


def test_check_code_against_facts_passes_when_constant_present():
    """V11 第三层：代码正确包含关键常量时应通过。"""
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="速度3 m/s",
        problem_facts=[ProblemFact(value=3.0, unit="m/s", context="3 m/s")],
    )
    code = "v_sink = 3.0\nimport pandas as pd\ndf = pd.DataFrame({'t': [1, 2]})\ndf['y'] = v_sink * df['t']\ndf.to_csv('out.csv', index=False)"
    issues = check_code_against_facts(code, static_ltm)
    assert not any("代码常量缺失" in i for i in issues)


def test_check_code_against_facts_detects_invalid_column():
    """V11 第三层：代码访问不存在的列名时应告警。

    V11.3 修复：派生列识别。`df['weight'] = df['weight'] * 2` 是自引用模式，
    'weight' 既在写入位置也在读取位置，但写入会创建派生列，读取该派生列是合法的。
    因此本测试改为访问真正不存在的列 'unknown_col'（既不在数据画像，也不是代码创建的）。
    """
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="test",
        data_profile=DataProfile(
            columns=[
                ColumnProfile(name="age", dtype="int"),
                ColumnProfile(name="sex", dtype="text"),
            ]
        ),
    )
    # 代码里访问了 df['unknown_col']，但数据里没有该列，代码也没创建它
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'age': [1, 2]})\n"
        "df['new_col'] = df['unknown_col'] * 2\n"  # 读取 unknown_col（未创建）
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert any("unknown_col" in i and "不在数据画像中" in i for i in issues)


def test_check_code_against_facts_accepts_percentage_decimal_equivalent():
    """V11.3 修复：百分比单位接受小数等价值。

    题目说"4%"，LLM 写成 0.04（科学计算标准写法）应通过校验。
    """
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="Y染色体浓度达到4%",
        problem_facts=[ProblemFact(value=4.0, unit="%", context="Y染色体浓度达到4%")],
    )
    # 代码用 0.04（4% 的小数等价值）
    code = (
        "threshold = 0.04\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'y': [0.05, 0.03]})\n"
        "df['pass'] = df['y'] >= threshold\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert not any("代码常量缺失" in i for i in issues), \
        "4% 写成 0.04 应通过校验（小数等价值）"


def test_check_code_against_facts_accepts_derived_column_read():
    """V11.3 修复：派生列创建后读取不应告警。

    `df['末次月经_dt'] = pd.to_datetime(df['末次月经'])` 创建派生列，
    后续读取 `df['末次月经_dt']` 应通过校验。
    """
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="test",
        data_profile=DataProfile(
            columns=[
                ColumnProfile(name="末次月经", dtype="text"),
                ColumnProfile(name="检测日期", dtype="text"),
            ]
        ),
    )
    # 创建派生列后读取
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'末次月经': ['2023-01-01'], '检测日期': ['2023-03-01']})\n"
        "df['末次月经_dt'] = pd.to_datetime(df['末次月经'])\n"
        "df['检测日期_dt'] = pd.to_datetime(df['检测日期'])\n"
        "df['孕周'] = (df['检测日期_dt'] - df['末次月经_dt']).dt.days / 7.0\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert not any("末次月经_dt" in i for i in issues), \
        "派生列 '末次月经_dt' 创建后读取应通过校验"
    assert not any("检测日期_dt" in i for i in issues), \
        "派生列 '检测日期_dt' 创建后读取应通过校验"


def test_check_code_against_facts_handles_syntax_error():
    """V11 第三层：代码有语法错误时应返回明确告警。"""
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(raw_problem="test")
    code = "def f(:\n    return 1"
    issues = check_code_against_facts(code, static_ltm)
    assert any("语法错误" in i for i in issues)


# ── V11.4 测试：fact category 分类 + 校验器跳过 ────────────────

def test_classify_fact_identifies_data_range_with_column_match():
    """V11.4：双重判据识别 data_range（强信号词 + 列名匹配）。"""
    from modeling_assistant.data.facts import classify_fact

    # GC 含量正常范围 + 列名"GC含量"匹配 → data_range
    fact = ProblemFact(value=60.0, unit="%", context="GC 含量正常范围 40%-60%")
    columns = [ColumnProfile(name="GC含量", dtype="float")]
    assert classify_fact(fact, columns) == "data_range"


def test_classify_fact_does_not_misjudge_physical_param_as_data_range():
    """V11.4：弱信号词（"速度范围"）不应误判为 data_range。

    "无人机速度范围 70~140 m/s" 里的 70 和 140 是真物理量，必须保留校验。
    "速度范围"不在强信号词列表，应判为 physical_param。
    """
    from modeling_assistant.data.facts import classify_fact

    fact = ProblemFact(value=140.0, unit="m/s", context="无人机速度范围 70~140 m/s")
    columns = []  # 即使有列名，"速度范围"也不是强信号词
    assert classify_fact(fact, columns) == "physical_param"


def test_classify_fact_strong_keyword_without_column_match_keeps_physical_param():
    """V11.4：有强信号词但无列名匹配时，保守判为 physical_param。"""
    from modeling_assistant.data.facts import classify_fact

    fact = ProblemFact(value=60.0, unit="%", context="某物质正常范围 40%-60%")
    # columns 里没有匹配 context 的列名
    columns = [ColumnProfile(name="温度", dtype="float")]
    assert classify_fact(fact, columns) == "physical_param"


def test_classify_fact_identifies_count_unit():
    """V11.4：计数单位（枚/次/架等）识别为 count。"""
    from modeling_assistant.data.facts import classify_fact

    fact = ProblemFact(value=3.0, unit="枚", context="投放3枚烟幕弹")
    assert classify_fact(fact, None) == "count"


def test_check_code_against_facts_skips_data_range_fact():
    """V11.4：data_range 类 fact 应跳过字面量检查。

    场景：题目说"GC 含量正常范围 40%-60%"，代码不写字面量也应通过。
    """
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="GC 含量正常范围 40%-60%",
        problem_facts=[
            ProblemFact(
                value=60.0, unit="%",
                context="GC 含量正常范围 40%-60%",
                category="data_range",
            ),
        ],
    )
    # 代码完全不写 40/60/0.4/0.6 字面量
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'GC': [0.5, 0.55]})\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert not any("代码常量缺失" in i for i in issues), \
        "data_range 类 fact 应跳过字面量检查"


def test_check_code_against_facts_skips_count_fact():
    """V11.4：count 类 fact 应跳过字面量检查。"""
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="投放3枚烟幕弹",
        problem_facts=[
            ProblemFact(
                value=3.0, unit="枚",
                context="投放3枚烟幕弹",
                category="count",
            ),
        ],
    )
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'x': [1, 2]})\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert not any("代码常量缺失" in i for i in issues), \
        "count 类 fact 应跳过字面量检查"


def test_check_code_against_facts_still_checks_physical_param():
    """V11.4：physical_param 类 fact 仍应执行字面量检查（不漏检真物理量）。"""
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="无人机速度范围 70~140 m/s",
        problem_facts=[
            ProblemFact(
                value=140.0, unit="m/s",
                context="无人机速度范围 70~140 m/s",
                category="physical_param",  # 即使 context 含"范围"，也保留校验
            ),
        ],
    )
    # 代码不写 140
    code = (
        "v = 100.0\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'t': [1, 2]})\n"
        "df['y'] = v * df['t']\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert any("代码常量缺失" in i and "140.0" in i for i in issues), \
        "physical_param 类 fact 仍应被校验"


def test_check_code_against_facts_accepts_rename_target_column():
    """V11.4：rename 后的新列名应通过校验（valid_columns 纳入 rename 目标列）。"""
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="test",
        data_profile=DataProfile(
            columns=[ColumnProfile(name="唯一比对的读段数", dtype="int")]
        ),
    )
    # LLM 把中文列名 rename 成英文缩写后访问
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'唯一比对的读段数': [100, 200]})\n"
        "df = df.rename(columns={'唯一比对的读段数': 'Reads_mapped'})\n"
        "df['log_reads'] = df['Reads_mapped'].apply(lambda x: x + 1)\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert not any("Reads_mapped" in i for i in issues), \
        "rename 后的新列名 'Reads_mapped' 应通过校验"


def test_check_code_against_facts_accepts_rename_inplace_form():
    """V11.4：rename 的 inplace=True 形式也应被识别。"""
    from modeling_assistant.validation.constants import check_code_against_facts

    static_ltm = StaticLTM(
        raw_problem="test",
        data_profile=DataProfile(
            columns=[ColumnProfile(name="旧列名", dtype="int")]
        ),
    )
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'旧列名': [1, 2]})\n"
        "df.rename(columns={'旧列名': 'new_col'}, inplace=True)\n"
        "df['result'] = df['new_col'] * 2\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    issues = check_code_against_facts(code, static_ltm)
    assert not any("new_col" in i for i in issues), \
        "inplace=True 形式的 rename 目标列也应通过校验"


def test_extract_facts_assigns_category_data_range():
    """V11.4：extract_facts_from_problem 应正确为 data_range fact 填充 category。"""
    from modeling_assistant.data.facts import extract_facts_from_problem

    raw_problem = "GC 含量正常范围为40% ~ 60%"
    columns = [ColumnProfile(name="GC含量", dtype="float")]
    facts = extract_facts_from_problem(raw_problem, columns=columns)
    # 至少有一个 fact 被标为 data_range
    data_range_facts = [f for f in facts if f.category == "data_range"]
    assert len(data_range_facts) >= 1, \
        f"应有 fact 被标为 data_range，实际：{[f.category for f in facts]}"


def test_extract_facts_keeps_physical_param_for_speed_range():
    """V11.4：extract_facts 对"速度范围"类应保留 physical_param，不误判。"""
    from modeling_assistant.data.facts import extract_facts_from_problem

    raw_problem = "无人机速度范围 70~140 m/s"
    facts = extract_facts_from_problem(raw_problem, columns=[])
    # 所有 m/s 单位的 fact 应为 physical_param
    speed_facts = [f for f in facts if f.unit == "m/s"]
    assert all(f.category == "physical_param" for f in speed_facts), \
        f"速度范围类应保留 physical_param，实际：{[(f.value, f.category) for f in speed_facts]}"


# ── V11.4 测试：CoderResponse fallback 解析 ────────────────────

def test_coder_fallback_parser_extracts_pure_code_block():
    """V11.4：LLM 偶发返回纯 Python 代码块时，fallback 应提取 code。"""
    from modeling_assistant.agents.runtime import _coder_fallback_parser

    # LLM 直接返回纯代码块（不带 JSON 包装）
    raw = "```python\nimport os\nimport pandas as pd\ndf = pd.DataFrame({'x': [1]})\ndf.to_csv('out.csv', index=False)\n```"
    result = _coder_fallback_parser(raw)
    assert result is not None
    assert "import pandas" in result.code
    assert result.result_path == "results/output.csv"


def test_coder_fallback_parser_returns_none_for_non_pure_code_block():
    """V11.4：返回内容前后有非空白字符时，fallback 不应误匹配。"""
    from modeling_assistant.agents.runtime import _coder_fallback_parser

    # 代码块前后有文字，不是纯代码块返回
    raw = "这是代码：\n```python\nimport os\n```\n请检查"
    result = _coder_fallback_parser(raw)
    assert result is None


def test_coder_fallback_parser_returns_none_for_empty():
    """V11.4：空输入返回 None。"""
    from modeling_assistant.agents.runtime import _coder_fallback_parser

    assert _coder_fallback_parser("") is None
    assert _coder_fallback_parser(None) is None


# ── Prompt 渲染验证 ──────────────────────────────────────────────

def test_clarifier_prompt_renders_problem_facts():
    """V11 第二层：Clarifier Prompt 应注入 problem_facts 列表。"""
    prompt = PromptCatalog().render(
        "clarifier",
        PromptContext(
            static_ltm=StaticLTM(
                raw_problem="test",
                problem_facts=[
                    ProblemFact(value=3.0, unit="m/s", context="云团以3 m/s的速度下沉"),
                ],
            ),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "problem_facts" in prompt or "3.0" in prompt
    assert "m/s" in prompt
    assert "原文" in prompt


def test_clarifier_prompt_has_assumption_tag_discipline():
    """V20：Clarifier Prompt 应包含放置标签与【关键】规则，且无旧【关键假设】标记。"""
    prompt = PromptCatalog().render(
        "clarifier",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "【全文】" in prompt
    assert "【问题N】" in prompt
    assert "【关键】" in prompt
    assert "【关键假设】" not in prompt
    assert "不得重复列出" in prompt


def test_writer_prompt_has_assumption_placement_rules():
    """V20：Writer Prompt 应包含模型假设写作规则（放置、条数、禁止细节）。"""
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "模型假设写作规则" in prompt
    assert "3_assumptions.tex" in prompt
    assert "不超过 6" in prompt
    assert "【问题N】" in prompt
    assert "【关键】" in prompt


def test_architect_prompt_uses_critical_tag():
    """V20：Architect Prompt 的关键假设实验规则应使用【关键】而非旧标记。"""
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "【关键】" in prompt
    assert "【关键假设】" not in prompt
    assert "扰动或对照实验" in prompt


def test_milestone_reviewer_prompt_uses_placement_tags():
    """V20：Milestone Reviewer 1 应检查放置标签与【关键】标注。"""
    prompt = PromptCatalog().render(
        "milestone_reviewer_1",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "【全文】" in prompt
    assert "【问题N】" in prompt
    assert "【关键】" in prompt
    assert "【关键假设】" not in prompt


def test_final_reviewer_prompt_has_assumption_chapter_checks():
    """V20：Final Reviewer Prompt 应含假设章硬检查且无旧【关键假设】标记。"""
    prompt = PromptCatalog().render(
        "final_reviewer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "假设章硬检查" in prompt
    assert "【问题N】" in prompt
    assert "【关键】" in prompt
    assert "【关键假设】" not in prompt


def test_final_reviewer_prompt_has_degenerate_solution_check():
    """V21：终审应把主方案退化解判为硬问题，不能因「结果诚实」通过。"""
    prompt = PromptCatalog().render(
        "final_reviewer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "退化解硬检查" in prompt
    assert "退化解" in prompt
    assert "结果诚实" in prompt


def test_final_reviewer_prompt_has_citation_and_person_hard_checks():
    """V21：正文引用数为 0 与人称混用应升级为终审硬问题。"""
    prompt = PromptCatalog().render(
        "final_reviewer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "引用纪律" in prompt
    assert "引用数为 0" in prompt
    assert "人称与术语" in prompt
    assert "我们" in prompt


def test_style_injection_defaults_raise_writing_and_add_highlight():
    """V21：句法规则注入强度升到 1.0；highlight 独立成层供 dropout。"""
    from modeling_assistant.config.settings import AppSettings

    injection = AppSettings().style_injection
    assert injection["writing"] == 1.0
    assert injection["structure"] == 1.0
    assert injection["chart"] == 0.8
    assert injection["highlight"] == 0.5


def test_coder_external_mode_defaults_to_human():
    """V22：默认 Coder/Drawer 任务由人工执行（human 模式，含 figures.py）。"""
    from modeling_assistant.config.settings import AppSettings

    assert AppSettings().coder_external_mode == "human"


def test_method_knowledge_uses_hitl_confirmed_problem_type():
    """V22：方法知识注入以 HITL 确认的题型为准，覆盖自动判定。"""
    prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="城市物流配送调度优化问题"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(problem_type="evaluation"),
        ),
    )
    assert "evaluation" in prompt
    assert "AHP" in prompt


def test_mathematician_prompt_has_occam_rule():
    """V21：Mathematician 应覆盖简单到复杂的完整谱系，奥卡姆只作评估规则。"""
    prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "奥卡姆原则" in prompt
    assert "选更简单的那个" in prompt
    assert "完整谱系" in prompt
    assert "不在发散阶段自我审查" in prompt


def test_mathematician_prompt_requires_more_plans():
    """V23：Mathematician 每轮至少产出 5 个候选方案，支撑方案池。"""
    prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "至少 5 个" in prompt
    assert "方案池" in prompt


def test_clarifier_prompt_has_selected_plan():
    """V23：Clarifier 应明确只围绕当前选定方案提炼 LTM。"""
    prompt = PromptCatalog().render(
        "clarifier",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(selected_plan_id="plan_1"),
        ),
    )
    assert "当前选定方案" in prompt
    assert "忽略其他候选方案" in prompt
    assert "selected_plan_json" not in prompt  # 占位符已被替换


def test_realist_prompt_has_occam_rule():
    """V21：Realist 评估时应惩罚无收益的过度复杂化。"""
    prompt = PromptCatalog().render(
        "realist",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            control=ControlState(),
        ),
    )
    assert "奥卡姆原则" in prompt
    assert "复杂度未带来可解释性或结果改进" in prompt


def test_architect_prompt_has_occam_rule():
    """V21：Architect 设计方案时应拒绝无实质收益的复杂方法。"""
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "奥卡姆原则" in prompt
    assert "不引入无实质收益的复杂方法" in prompt


def test_architect_prompt_has_main_conclusion_figure_requirement():
    """V21：Architect 每题必须规划至少 1 张主结论呈现图。"""
    prompt = PromptCatalog().render(
        "architect",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "主结论呈现图" in prompt
    assert "主结论呈现规则" in prompt
    assert "一目了然" in prompt
    assert "content_spec" in prompt


def test_writer_prompt_has_main_conclusion_figure_requirement():
    """V21：Writer 必须在该题主结论处引用主结论呈现图。"""
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "主结论呈现图" in prompt
    assert "一目了然" in prompt


def test_writer_prompt_has_language_discipline():
    """V21：Writer 应包含四条可检查的语言纪律（结果先行/语气/人称/引用）。"""
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "语言纪律" in prompt
    assert "结果先行" in prompt
    assert "发现 + 对策" in prompt
    assert "本文/本研究" in prompt
    assert "引用纪律" in prompt


def test_writer_prompt_has_abstract_and_restatement_skeleton():
    """V21：摘要五段式与问题重述四小节骨架应写入模板层。"""
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "摘要固定五段式" in prompt
    assert "本文创新性在于" in prompt
    assert "1.1 问题背景" in prompt
    assert "1.4 总体路线" in prompt


def test_writer_prompt_has_expression_revision_checklist():
    """V21：论文修订反馈应固定附加表达修订清单（人称/引用/长句/负结果/小结）。"""
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "表达修订清单" in prompt
    assert "长句" in prompt
    assert "小结" in prompt
    assert "禁止只写" in prompt


def test_writer_prompt_has_signature_moves_section():
    """V21：Writer 应提供标志性句式骨架注入位（可套用骨架、禁止复制内容）。"""
    prompt = PromptCatalog().render(
        "writer",
        PromptContext(
            static_ltm=StaticLTM(raw_problem="test"),
            dynamic_ltm=DynamicLTM(objective="目标"),
            control=ControlState(),
        ),
    )
    assert "标志性句式骨架" in prompt
    assert "允许套用句法骨架" in prompt


def test_coder_prompt_renders_problem_facts_and_parse_hints():
    """V11 第三层：Coder Prompt 应注入 problem_facts 和 data_parse_hints。"""
    prompt = PromptCatalog().render(
        "coder",
        PromptContext(
            static_ltm=StaticLTM(
                raw_problem="test",
                data_profile=DataProfile(
                    columns=[
                        ColumnProfile(
                            name="孕周",
                            dtype="text",
                            sample_values=["16W"],
                            parse_hint="df['孕周'].str.replace('W', '', regex=False).astype(float)",
                        ),
                    ],
                ),
                problem_facts=[
                    ProblemFact(value=3.0, unit="m/s", context="3 m/s"),
                ],
            ),
            dynamic_ltm=DynamicLTM(),
            control=ControlState(),
        ),
    )
    assert "3.0" in prompt
    assert "m/s" in prompt
    assert "parse_hint" in prompt or "str.replace" in prompt
    assert "V11" in prompt


# ── V12：数据不进 prompt，只进提炼信息 ────────────────────────────

def test_prompt_does_not_include_raw_data():
    """V12 修复：sample_head / 全量相关性矩阵 / 全量样例值不得进入任何 prompt。"""
    from modeling_assistant.schemas.state import FileSummary

    profile = DataProfile(
        total_rows=10,
        total_cols=2,
        columns=[
            ColumnProfile(name="a", dtype="float", sample_values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        ],
        sample_head=[{"a": "1.0"}, {"a": "2.0"}],
        correlation_matrix={"a": {"a": 1.0}},
        file_summaries=[
            FileSummary(
                path="data/附件1.csv",
                rows=10,
                cols=2,
                columns=[
                    ColumnProfile(name="a", dtype="float", sample_values=[1.0, 2.0, 3.0, 4.0, 5.0]),
                ],
            )
        ],
    )
    state = StaticLTM(
        raw_problem="test",
        data_profile=profile,
    )
    coder_prompt = PromptCatalog().render(
        "coder",
        PromptContext(static_ltm=state, dynamic_ltm=DynamicLTM(), control=ControlState()),
    )
    math_prompt = PromptCatalog().render(
        "mathematician",
        PromptContext(static_ltm=state, dynamic_ltm=DynamicLTM(), control=ControlState()),
    )
    # 原始数据不得出现
    assert "sample_head" not in coder_prompt
    assert "correlation_matrix" not in coder_prompt
    assert "sample_head" not in math_prompt
    assert "correlation_matrix" not in math_prompt
    # 全量样例不得出现（5 个样例只应保留前 3 个）
    assert "1.0, 2.0, 3.0, 4.0, 5.0" not in coder_prompt
    # 紧凑摘要应保留文件边界与列结构
    assert "data/附件1.csv" in coder_prompt
    assert "数据概要" in coder_prompt


def test_data_loader_builds_file_summaries(tmp_path):
    """V12 修复：多附件各自保留 rows/cols/columns，不再只呈现合并后的大表。"""
    from modeling_assistant.data.loader import load_data_profile

    f1 = tmp_path / "data1.csv"
    f2 = tmp_path / "data2.csv"
    f1.write_text("age,score\n1,2\n3,4\n", encoding="utf-8")
    f2.write_text("x,y\n0.1\n", encoding="utf-8")

    profile = load_data_profile([str(f1), str(f2)])
    assert len(profile.file_summaries) == 2
    assert profile.file_summaries[0].rows == 2
    assert profile.file_summaries[0].cols == 2
    assert {c.name for c in profile.file_summaries[0].columns} == {"age", "score"}
    assert profile.file_summaries[1].rows == 1
    # 兼容旧字段：合并画像仍保留
    assert profile.total_rows == 3


def test_data_analyst_node_writes_intelligence(tmp_path, monkeypatch):
    """V12 新增：data_analyst_node 把 LLM 提炼的数据情报写入 static_ltm.data_intelligence。"""
    from modeling_assistant.agents.nodes import data_analyst_node
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.schemas.responses import DataIntelligenceResponse
    from modeling_assistant.schemas.state import FileSummary, GraphState

    runtime = AgentRuntime.from_settings(
        AppSettings(output_dir=tmp_path / "out", api_key_env="MISSING_KEY_FOR_TEST")
    )

    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        return DataIntelligenceResponse(
            insights=["附件1是反射率光谱表，需按文件分组拟合", "订单表与距离矩阵需按站点关联"]
        )

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)
    state: GraphState = {
        "static_ltm": StaticLTM(
            raw_problem="test",
            data_profile=DataProfile(
                file_summaries=[FileSummary(path="a.csv", rows=3, cols=1, columns=[ColumnProfile(name="x", dtype="float")])]
            ),
        ),
        "control": ControlState(),
    }
    result = data_analyst_node(state, runtime=runtime)
    assert len(result["static_ltm"].data_intelligence) == 2
    assert result["control"].phase == "data_intelligence_extracted"


def test_architect_node_stores_result_contract(tmp_path, monkeypatch):
    """V12 新增：Architect 声明的 result_contract 应随 artifacts 下传。"""
    from modeling_assistant.agents.nodes import architect_node
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.schemas.responses import ArchitectResponse, ResultColumnSpec, ResultContract
    from modeling_assistant.schemas.state import ArtifactBundle, GraphState

    runtime = AgentRuntime.from_settings(
        AppSettings(output_dir=tmp_path / "out", api_key_env="MISSING_KEY_FOR_TEST")
    )

    def mock_invoke(self, name, state, response_cls, system_prompt=None, fallback_parser=None):
        return ArchitectResponse(
            outline={"摘要": "x"},
            pseudocode=["步骤1"],
            result_contract=ResultContract(
                description="最优时点表",
                allow_single_row=False,
                columns=[ResultColumnSpec(name="group", dtype="category")],
            ),
        )

    monkeypatch.setattr(AgentRuntime, "invoke_structured", mock_invoke)
    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(objective="目标"),
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    result = architect_node(state, runtime=runtime)
    assert result["artifacts"].result_contract is not None
    assert result["artifacts"].result_contract.columns[0].name == "group"


def test_result_reviewer_node_accepts_single_row_with_contract(tmp_path):
    """V12 修复：有契约时 result_reviewer_node 应放行合法的单行标量答案。"""
    import tempfile
    from pathlib import Path

    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.schemas.state import ArtifactBundle, GraphState
    from modeling_assistant.validation.results import result_reviewer_node

    csv_path = Path(tempfile.mkdtemp()) / "output.csv"
    csv_path.write_text("answer\n12.5\n", encoding="utf-8")
    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(),
        "control": ControlState(),
        "artifacts": ArtifactBundle(
            result_paths=[str(csv_path)],
            result_contract=ResultContract(
                allow_single_row=True,
                columns=[ResultColumnSpec(name="answer", dtype="float", min=0.0, max=60.0)],
            ),
        ),
        "prompt_audit": {},
    }
    result = result_reviewer_node(state)
    assert result["control"].phase == "result_review_passed"
    assert not result["control"].last_result_review_issues


# ── V13：编程手模式（builtin / codex / human）────────────────────

def test_parse_hitl_decision_supports_revise_and_auto():
    """V13：_parse_hitl_decision 应识别 revise 与 auto。"""
    from modeling_assistant.agents.nodes import _parse_hitl_decision

    assert _parse_hitl_decision("revise 换用整数规划")["type"] == "revise"
    assert _parse_hitl_decision("revise 换用整数规划")["version"] == "换用整数规划"
    assert _parse_hitl_decision("auto")["type"] == "auto"
    assert _parse_hitl_decision("approve")["type"] == "approve"


def test_parse_hitl_decision_set_problem_type():
    """V22：题型确认支持 set <题型> 覆盖。"""
    from modeling_assistant.agents.nodes import _parse_hitl_decision

    d = _parse_hitl_decision("set evaluation")
    assert d["type"] == "set"
    assert d["version"] == "evaluation"


def test_parse_hitl_decision_choose_plan():
    """V23：方案池定夺支持 choose <plan_id> 改选方案。"""
    from modeling_assistant.agents.nodes import _parse_hitl_decision

    d = _parse_hitl_decision("choose plan_2")
    assert d["type"] == "choose"
    assert d["version"] == "plan_2"


def test_route_after_architect_external_skips_review_after_approved():
    """V13：架构审核通过后，重试直接进任务包分发，不再重复打断审核。"""
    from modeling_assistant.graph.routing import route_after_architect_external

    state = {"control": ControlState(implementation_architecture_reviewed=False)}
    assert route_after_architect_external(state) == "hitl_implementation_review"
    state = {"control": ControlState(implementation_architecture_reviewed=True)}
    assert route_after_architect_external(state) == "dispatch_implementation"


def test_route_after_implementation_review_and_human():
    """V13：实现审核与人工交付后的路由。"""
    from modeling_assistant.graph.routing import (
        route_after_implementation_human,
        route_after_implementation_review,
    )

    assert route_after_implementation_review({"control": ControlState()}) == "dispatch_implementation"
    state = {"control": ControlState(phase="hitl_implementation_revised")}
    assert route_after_implementation_review(state) == "architect"
    state = {"control": ControlState(rollback_to_version="v1.0")}
    assert route_after_implementation_review(state) == "rollback"

    assert route_after_implementation_human({"control": ControlState(phase="implementation_human_ready")}) == "coder"
    assert route_after_implementation_human({"control": ControlState(phase="implementation_auto")}) == "coder"
    assert route_after_implementation_human({"control": ControlState(phase="implementation_revised")}) == "architect"


def test_coder_node_human_mode_reads_solution(tmp_path):
    """V13：human 模式下 coder_node 直接执行人工交付的 solution.py。"""
    from modeling_assistant.agents.nodes import coder_node
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.schemas.state import ArtifactBundle, GraphState

    output_dir = tmp_path / "outputs"
    task_dir = tmp_path / "tasks"
    task_dir.mkdir(parents=True)
    (output_dir / "results").mkdir(parents=True)
    code = (
        "import os\nfrom pathlib import Path\nimport pandas as pd\n"
        "out = os.environ['MODELING_OUTPUT_DIR']\n"
        "p = Path(out) / 'results' / 'output.csv'\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "pd.DataFrame({'answer': [12.5]}).to_csv(p, index=False)\n"
    )
    (task_dir / "solution.py").write_text(code, encoding="utf-8")

    runtime = AgentRuntime.from_settings(
        AppSettings(
            output_dir=output_dir,
            api_key_env="MISSING_KEY_FOR_TEST",
            coder_external_mode="human",
        )
    )
    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(),
        "control": ControlState(),
        "artifacts": ArtifactBundle(coder_task_dir=str(task_dir)),
        "prompt_audit": {},
    }
    result = coder_node(state, runtime=runtime)
    assert result["control"].phase == "code_executed_successfully"
    assert result["artifacts"].result_paths
    assert "output.csv" in result["artifacts"].result_paths[0]


def test_coder_node_human_mode_missing_solution(tmp_path):
    """V13：human 模式未交付 solution.py 时应走 code_generation_empty 失败路径。"""
    from modeling_assistant.agents.nodes import coder_node
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.schemas.state import ArtifactBundle, GraphState

    output_dir = tmp_path / "outputs"
    task_dir = tmp_path / "tasks"
    task_dir.mkdir(parents=True)
    runtime = AgentRuntime.from_settings(
        AppSettings(
            output_dir=output_dir,
            api_key_env="MISSING_KEY_FOR_TEST",
            coder_external_mode="human",
        )
    )
    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(),
        "control": ControlState(),
        "artifacts": ArtifactBundle(coder_task_dir=str(task_dir)),
        "prompt_audit": {},
    }
    result = coder_node(state, runtime=runtime)
    assert result["control"].phase == "code_generation_empty"
    assert any("solution.py" in log for log in result["control"].coder_error_log)


def test_coder_node_human_mode_runs_figures(tmp_path):
    """V13：human 模式交付 figures.py 后，图片应被收集进 figure_paths。"""
    from modeling_assistant.agents.nodes import coder_node
    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.schemas.state import ArtifactBundle, GraphState

    output_dir = tmp_path / "outputs"
    task_dir = tmp_path / "tasks"
    task_dir.mkdir(parents=True)
    (output_dir / "results").mkdir(parents=True)
    (task_dir / "solution.py").write_text(
        "import os\nfrom pathlib import Path\nimport pandas as pd\n"
        "out = os.environ['MODELING_OUTPUT_DIR']\n"
        "p = Path(out) / 'results' / 'output.csv'\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "pd.DataFrame({'x': [1, 2]}).to_csv(p, index=False)\n",
        encoding="utf-8",
    )
    (task_dir / "figures.py").write_text(
        "import os, matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "out = os.environ['MODELING_OUTPUT_DIR']\n"
        "d = os.path.join(out, 'figures')\n"
        "os.makedirs(d, exist_ok=True)\n"
        "plt.plot([1, 2, 3], [1, 4, 9])\n"
        "plt.savefig(os.path.join(d, 'figure1.png'))\n",
        encoding="utf-8",
    )

    runtime = AgentRuntime.from_settings(
        AppSettings(
            output_dir=output_dir,
            api_key_env="MISSING_KEY_FOR_TEST",
            coder_external_mode="human",
        )
    )
    state: GraphState = {
        "static_ltm": StaticLTM(raw_problem="test"),
        "dynamic_ltm": DynamicLTM(),
        "control": ControlState(),
        "artifacts": ArtifactBundle(coder_task_dir=str(task_dir)),
        "prompt_audit": {},
    }
    result = coder_node(state, runtime=runtime)
    assert result["control"].phase == "code_executed_successfully"
    assert any("figure1.png" in p for p in result["artifacts"].figure_paths)


def test_human_mode_graph_end_to_end(tmp_path, monkeypatch):
    """V13：human 模式整图链路——架构审核 → 人工写码 → 执行验证 → 成稿。"""
    from langgraph.types import Command

    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.graph.builder import build_graph
    from modeling_assistant.schemas.responses import (
        AnalystResponse,
        ArchitectResponse,
        ClarifierResponse,
        CoderResponse,
        DataIntelligenceResponse,
        LoadBearingAnalysisResponse,
        MathematicianResponse,
        MilestoneReviewer1Response,
        PlanEvaluation,
        RealistResponse,
        ReflectionResponse,
        ResultContract,
        WriterResponse,
    )
    from modeling_assistant.schemas.state import ArtifactBundle

    output_dir = tmp_path / "outputs"
    solution = (
        "import os\nfrom pathlib import Path\nimport pandas as pd\n"
        "out = os.environ['MODELING_OUTPUT_DIR']\n"
        "p = Path(out) / 'results' / 'output.csv'\n"
        "q1 = Path(out) / 'results' / 'q1.csv'\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "pd.DataFrame({'answer': [12.5]}).to_csv(p, index=False)\n"
        "pd.DataFrame({'answer': [12.5]}).to_csv(q1, index=False)\n"
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

    settings = AppSettings(
        output_dir=output_dir,
        api_key_env="MISSING_KEY_FOR_TEST",
        coder_external_mode="human",
        search_enabled=False,
    )
    app = build_graph(runtime=AgentRuntime.from_settings(settings))
    config = {"configurable": {"thread_id": "human-e2e"}}
    state = {
        "static_ltm": StaticLTM(raw_problem="x"),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    cur = state
    seen_human = False
    final = None
    for _ in range(30):
        result = app.invoke(cur, config)
        interrupts = result.get("__interrupt__")
        if not interrupts:
            final = result
            break
        interrupt_value = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        stage = interrupt_value.get("stage") if isinstance(interrupt_value, dict) else str(interrupt_value)
        if stage == "implementation_human":
            task_dir = Path(interrupt_value.get("task_dir"))
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "solution.py").write_text(solution, encoding="utf-8")
            seen_human = True
        cur = Command(resume="approve")
    assert seen_human, "应经过人工编程手交付暂停点"
    assert final is not None and final["control"].phase == "completed"
    assert final["artifacts"].result_paths


# ── V14：小题循环（拆分 / 验收 / 跨小题 / 两题全链路）──────────────

def test_split_sub_questions_recognizes_common_markers():
    from modeling_assistant.agents.nodes import split_sub_questions

    text = "前言。问题1 求A。问题2 求B。问题3 求C。"
    parts = split_sub_questions(text)
    assert len(parts) == 3
    assert "问题1" in parts[0]
    assert "问题2" in parts[1]
    assert "问题3" in parts[2]

    text2 = "（1）求A（2）求B"
    parts2 = split_sub_questions(text2)
    assert len(parts2) == 2


def test_split_sub_questions_no_markers_returns_whole():
    from modeling_assistant.agents.nodes import split_sub_questions

    text = "只有一个题目，没有小题分隔。"
    assert split_sub_questions(text) == [text]


def test_parse_hitl_decision_sub_question_commands():
    from modeling_assistant.agents.nodes import _parse_hitl_decision

    assert _parse_hitl_decision("pass")["type"] == "pass"
    d = _parse_hitl_decision("fail code 数值不对")
    assert d["type"] == "fail" and d["level"] == "code"
    assert "数值不对" in d["version"]
    assert _parse_hitl_decision("fail architecture 伪代码有误")["level"] == "architecture"
    assert _parse_hitl_decision("fail model 假设错误")["level"] == "model"
    d = _parse_hitl_decision("cross 1 Q1 的假设有问题")
    assert d["type"] == "cross" and d["target"] == 0
    assert "q1" in d["version"]
    assert _parse_hitl_decision("edit 问题1：a；问题2：b")["type"] == "edit"


def test_route_after_acceptance():
    from modeling_assistant.graph.routing import route_after_acceptance

    # pass → 还有下一题 → mathematician
    state = {
        "control": ControlState(
            phase="sub_question_passed",
            sub_questions=["q1", "q2"],
            current_sub_question_index=1,
        )
    }
    assert route_after_acceptance(state) == "mathematician"
    # pass → 最后一题 → collect_artifacts
    state["control"] = ControlState(
        phase="sub_question_passed",
        sub_questions=["q1", "q2"],
        current_sub_question_index=2,
    )
    assert route_after_acceptance(state) == "collect_artifacts"
    # fail code → 回到人工交付
    state["control"] = ControlState(
        phase="sub_question_fail_code", sub_question_attempts=1, sub_question_budget=4
    )
    assert route_after_acceptance(state) == "hitl_implementation_human"
    # fail architecture → architect
    state["control"] = ControlState(
        phase="sub_question_fail_architecture", sub_question_attempts=1, sub_question_budget=4
    )
    assert route_after_acceptance(state) == "architect"
    # fail model → mathematician
    state["control"] = ControlState(
        phase="sub_question_fail_model", sub_question_attempts=1, sub_question_budget=4
    )
    assert route_after_acceptance(state) == "mathematician"
    # 预算耗尽 → hitl_modeling
    state["control"] = ControlState(
        phase="sub_question_fail_model", sub_question_attempts=4, sub_question_budget=4
    )
    assert route_after_acceptance(state) == "hitl_modeling"
    # cross → 跨小题 HITL
    state["control"] = ControlState(phase="cross_sub_question_requested")
    assert route_after_acceptance(state) == "cross_sub_question_hitl"


def test_write_coder_task_package_uses_sub_question_dir(tmp_path):
    import json

    from modeling_assistant.handoff.spec import write_coder_task_package
    from modeling_assistant.schemas.state import ArtifactBundle

    control = ControlState(
        sub_questions=["问题1", "问题2"],
        current_sub_question_index=0,
    )
    state = {
        "static_ltm": StaticLTM(raw_problem="x"),
        "dynamic_ltm": DynamicLTM(),
        "control": control,
        "artifacts": ArtifactBundle(),
    }
    md_path, json_path = write_coder_task_package(state, tmp_path)
    assert md_path.parent.name == "q1"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sub_question"]["index"] == 0
    assert payload["constraints"]["output_path"].endswith("q1.csv")


def test_two_sub_questions_human_mode_end_to_end(tmp_path, monkeypatch):
    """V14：两小题 human 模式全链路——拆分 → q1 建模/实现/验收 → q2（上下文含 q1）→ Writer。"""
    from langgraph.types import Command

    from modeling_assistant.agents.runtime import AgentRuntime
    from modeling_assistant.config.settings import AppSettings
    from modeling_assistant.graph.builder import build_graph
    from modeling_assistant.schemas.responses import (
        AnalystResponse,
        ArchitectResponse,
        ClarifierResponse,
        CoderResponse,
        DataIntelligenceResponse,
        LoadBearingAnalysisResponse,
        MathematicianResponse,
        MilestoneReviewer1Response,
        PlanEvaluation,
        RealistResponse,
        ReflectionResponse,
        ResultContract,
        WriterResponse,
    )
    from modeling_assistant.schemas.state import ArtifactBundle

    output_dir = tmp_path / "outputs"
    solution = (
        "import os\nfrom pathlib import Path\nimport pandas as pd\n"
        "out = os.environ['MODELING_OUTPUT_DIR']\n"
        "p = Path(out) / 'results' / 'output.csv'\n"
        "q1 = Path(out) / 'results' / 'q1.csv'\n"
        "q2 = Path(out) / 'results' / 'q2.csv'\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "pd.DataFrame({'answer': [12.5]}).to_csv(p, index=False)\n"
        "pd.DataFrame({'answer': [12.5]}).to_csv(q1, index=False)\n"
        "pd.DataFrame({'answer': [12.5]}).to_csv(q2, index=False)\n"
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

    settings = AppSettings(
        output_dir=output_dir,
        api_key_env="MISSING_KEY_FOR_TEST",
        coder_external_mode="human",
        search_enabled=False,
    )
    app = build_graph(runtime=AgentRuntime.from_settings(settings))
    config = {"configurable": {"thread_id": "two-sub-q-human"}}
    state = {
        "static_ltm": StaticLTM(raw_problem="问题1 求A。问题2 求B。"),
        "dynamic_ltm": DynamicLTM(),
        "ltm_archive": [],
        "control": ControlState(),
        "artifacts": ArtifactBundle(),
        "prompt_audit": {},
    }
    cur = state
    human_writes = 0
    final = None
    for _ in range(60):
        result = app.invoke(cur, config)
        interrupts = result.get("__interrupt__")
        if not interrupts:
            final = result
            break
        interrupt_value = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        stage = interrupt_value.get("stage") if isinstance(interrupt_value, dict) else str(interrupt_value)
        if stage == "sub_question_split":
            cur = Command(resume="approve")
        elif stage == "implementation_human":
            task_dir = Path(interrupt_value.get("task_dir"))
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "solution.py").write_text(solution, encoding="utf-8")
            human_writes += 1
            cur = Command(resume="approve")
        elif stage == "sub_question_acceptance":
            cur = Command(resume="pass")
        elif stage == "cross_sub_question":
            cur = Command(resume="accept")
        else:
            cur = Command(resume="approve")

    assert human_writes == 2, f"应经历两次人工交付，实际 {human_writes}"
    assert final is not None and final["control"].phase == "completed"
    ctrl = final["control"]
    assert len(ctrl.sub_results) == 2
    assert len(final["ltm_archive"]) >= 2
    paths = final["artifacts"].result_paths
    assert any("q1.csv" in p for p in paths)
    assert any("q2.csv" in p for p in paths)
