from modeling_assistant.memory.validation import validate_dynamic_ltm
from modeling_assistant.schemas.state import DynamicLTM


def test_validate_clean_ltm_passes():
    """符号定义完整、公式只引用已定义符号 → 通过。"""
    ltm = DynamicLTM(
        assumptions=["x 是自变量"],
        nomenclature={"x": "自变量", "y": "因变量"},
        equations=["y = 2 * x + 1"],
        objective="最小化 y",
        solution_outline="线性拟合",
    )
    errors = validate_dynamic_ltm(ltm)
    assert errors == []


def test_classify_assumptions_groups_by_tags():
    """V20：按【全文】/【问题N】/【关键】分组假设，未标签假设进 unlabeled。"""
    from modeling_assistant.validation.assumption_tags import classify_assumptions

    result = classify_assumptions(
        [
            "【全文】题目所给数据真实可靠（原文：...）",
            "【问题1】男胎 Y 染色体浓度取值于(0,1)，在 logit 尺度建模（原文：...）",
            "【问题3】【关键】约束风险最小化在单调条件下等价于首穿时点"
            "（依据：...；风险：...；可验证性：...）",
            "没有标签的假设",
        ]
    )
    assert result["full"] == ["【全文】题目所给数据真实可靠（原文：...）"]
    # 【问题3】【关键】同时属于「问题」与「关键」两组
    assert len(result["question"]) == 2
    assert "【问题1】" in result["question"][0]
    assert "【问题3】【关键】" in result["question"][1]
    # 关键假设同时出现在 critical 分组
    assert len(result["critical"]) == 1
    assert "【问题3】【关键】" in result["critical"][0]
    assert result["unlabeled"] == ["没有标签的假设"]


def test_classify_assumptions_critical_full_in_both_groups():
    """V20：【全文】【关键】假设应同时出现在 full 与 critical 分组。"""
    from modeling_assistant.validation.assumption_tags import classify_assumptions

    result = classify_assumptions(
        ["【全文】【关键】系统在短期内处于稳定状态（依据：...；可验证性：...）"]
    )
    assert len(result["full"]) == 1
    assert len(result["critical"]) == 1
    assert result["question"] == []
    assert result["unlabeled"] == []


def test_question_tag_helpers():
    """V20：has_question_tag / question_index 兼容空白并区分全文标签。"""
    from modeling_assistant.validation.assumption_tags import (
        has_question_tag,
        question_index,
    )

    assert has_question_tag("【问题2】xxx")
    assert has_question_tag("【问题 2】xxx")
    assert not has_question_tag("【全文】xxx")
    assert question_index("【问题2】xxx") == 2
    assert question_index("【全文】xxx") is None


def test_validate_undefined_symbol_in_equation():
    """公式闭环校验已移除：公式引用未定义符号不再报错。

    设计理由：数学建模公式天然含向量分量(P_M)、下标(M0)、自定义函数(cover)等，
    穷举定义不可能，硬校验误判率极高（实测导致死循环）。
    真正的符号一致性由 Coder 执行反馈 + milestone LLM 语义审查保证。
    """
    ltm = DynamicLTM(
        nomenclature={"x": "自变量"},
        equations=["y = 2 * x + 1"],  # y 未在 nomenclature 中定义
        objective="最小化 y",
    )
    errors = validate_dynamic_ltm(ltm)
    assert errors == []  # 公式闭环校验已移除，不再报错


def test_validate_duplicate_descriptions():
    """不同符号有相同描述 → 警告歧义。"""
    ltm = DynamicLTM(
        nomenclature={
            "x1": "自变量",
            "x2": "自变量",  # 与 x1 描述相同
        },
        equations=["x1 + x2"],
        objective="最小化 x1 + x2",
    )
    errors = validate_dynamic_ltm(ltm)
    assert any("歧义" in err or "描述" in err for err in errors)


def test_validate_math_functions_are_reserved():
    """公式闭环校验已移除：数学函数（sin、cos 等）自然不会被视为未定义符号。"""
    ltm = DynamicLTM(
        nomenclature={"x": "自变量", "y": "因变量"},
        equations=["y = sin(x) + cos(x)"],
        objective="最小化 y",
    )
    errors = validate_dynamic_ltm(ltm)
    # 公式闭环校验已移除，不再检查公式中的符号 → 无错误
    assert errors == []


def test_validate_empty_ltm_passes():
    """空 LTM（无公式、无符号）应通过校验。"""
    ltm = DynamicLTM()
    errors = validate_dynamic_ltm(ltm)
    assert errors == []


from modeling_assistant.agents.searcher import SearchResult, validate_search_results


def test_validate_search_results_filters_placeholders():
    """校验应过滤占位和空结果。"""
    results = [
        SearchResult(title="[占位] 参考模型", summary="占位摘要"),
        SearchResult(title="真实论文", summary="这是一篇关于优化和交通的真实论文。"),
        SearchResult(title="", summary="空标题"),
    ]
    validated = validate_search_results(results, keywords=["优化", "交通"])
    assert len(validated) == 1
    assert validated[0].title == "真实论文"


def test_validate_search_results_deduplicates_by_title():
    """校验应按标题去重。"""
    results = [
        SearchResult(title="相同标题", summary="摘要 A"),
        SearchResult(title="相同标题", summary="摘要 B"),
        SearchResult(title="另一篇", summary="摘要 C"),
    ]
    validated = validate_search_results(results, keywords=[])
    assert len(validated) == 2


def test_validate_search_results_checks_relevance():
    """校验应过滤与关键词不相关的结果。"""
    results = [
        SearchResult(title="相关论文", summary="包含优化和机器学习。"),
        SearchResult(title="不相关论文", summary="这是一篇生物学论文。"),
    ]
    validated = validate_search_results(results, keywords=["优化", "机器学习"], min_relevance_keywords=2)
    assert len(validated) == 1
    assert validated[0].title == "相关论文"


# ── V12：ResultReviewer 契约化校验 ────────────────────────────────

def _write_csv(tmp_path, content: str):
    import tempfile
    from pathlib import Path

    d = tempfile.mkdtemp()
    p = Path(d) / "output.csv"
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_validate_result_accepts_single_row_when_contract_allows():
    """V12 修复：契约声明 allow_single_row=true 时，单行标量答案应通过。"""
    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "answer\n12.5\n")
    contract = ResultContract(
        description="有效遮蔽时长（秒）",
        allow_single_row=True,
        columns=[ResultColumnSpec(name="answer", dtype="float", min=0.0, max=60.0)],
    )
    report = validate_result(path, contract=contract)
    assert report["passed"], f"单行标量答案应通过契约校验：{report['issues']}"
    assert not report["issues"]


def test_validate_result_rejects_constant_column_when_contract_requires_distinct():
    """契约要求区分度的列是常量 → 必须拒绝。"""
    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "group,optimal_week\nA,10.0\nB,10.0\nC,10.0\n")
    contract = ResultContract(
        columns=[
            ResultColumnSpec(name="group", dtype="category"),
            ResultColumnSpec(
                name="optimal_week", dtype="float",
                min=0.0, max=40.0, distinct_required=True,
            ),
        ]
    )
    report = validate_result(path, contract=contract)
    assert not report["passed"]
    assert any("区分度" in issue for issue in report["issues"])


def test_validate_result_allows_constant_column_not_required_distinct():
    """契约未要求区分度的共享常量列 → 只告警，不拒绝。"""
    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "id,threshold,value\n1,20,1.1\n2,20,2.2\n3,20,3.3\n")
    contract = ResultContract(
        columns=[
            ResultColumnSpec(name="id", dtype="int", distinct_required=True),
            ResultColumnSpec(name="threshold", dtype="float"),
            ResultColumnSpec(name="value", dtype="float", distinct_required=True),
        ]
    )
    report = validate_result(path, contract=contract)
    assert report["passed"], f"共享常量列不应拒绝：{report['issues']}"
    assert any("常量" in w for w in report["warnings"])


def test_validate_result_rejects_out_of_range_when_contract_declares():
    """契约声明列范围后，越界值必须拒绝。"""
    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "value\n12.5\n")
    contract = ResultContract(
        allow_single_row=True,
        columns=[ResultColumnSpec(name="value", dtype="float", min=0.0, max=10.0)],
    )
    report = validate_result(path, contract=contract)
    assert not report["passed"]
    assert any("上限" in issue for issue in report["issues"])


def test_validate_result_legacy_behavior_without_contract():
    """无契约时保留旧的通用启发式（单行/常量列仍拒绝），保证向后兼容。"""
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "answer\n12.5\n")
    report = validate_result(path)
    assert not report["passed"]
    assert any("只有一行" in issue for issue in report["issues"])


def test_validate_result_rejects_all_constant_answer_columns():
    """V21：多行结果中所有数值答案列均为常量 → 退化解，必须判 fail。"""
    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "id,optimal_week\n1,10.0\n2,10.0\n3,10.0\n")
    contract = ResultContract(
        columns=[
            ResultColumnSpec(name="id", dtype="int"),
            ResultColumnSpec(name="optimal_week", dtype="float", min=0.0, max=40.0),
        ]
    )
    report = validate_result(path, contract=contract)
    assert not report["passed"]
    assert any("退化解" in issue and "常量" in issue for issue in report["issues"])


def test_validate_result_rejects_boundary_collapse():
    """V21：多行结果最优值全部落在契约边界 → 退化解，必须判 fail。"""
    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "group,optimal_week\nA,40.0\nB,40.0\nC,40.0\n")
    contract = ResultContract(
        columns=[
            ResultColumnSpec(name="group", dtype="category"),
            ResultColumnSpec(name="optimal_week", dtype="float", min=0.0, max=40.0),
        ]
    )
    report = validate_result(path, contract=contract)
    assert not report["passed"]
    assert any("退化解" in issue and "边界" in issue for issue in report["issues"])


def test_validate_result_rejects_duplicate_rows_without_contract():
    """V21：无契约时整表行完全相同 → 退化解，必须判 fail。"""
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "a,b\n1.0,2.0\n1.0,2.0\n1.0,2.0\n")
    report = validate_result(path)
    assert not report["passed"]
    assert any("退化解" in issue and "行完全相同" in issue for issue in report["issues"])


def test_validate_result_allows_single_row_scalar_at_boundary():
    """V21：单行标量答案落在边界值不算退化（allow_single_row 放行）。"""
    from modeling_assistant.schemas.responses import ResultColumnSpec, ResultContract
    from modeling_assistant.validation.results import validate_result

    path = _write_csv(None, "answer\n40.0\n")
    contract = ResultContract(
        description="最优检测时点（周）",
        allow_single_row=True,
        columns=[ResultColumnSpec(name="answer", dtype="float", min=0.0, max=40.0)],
    )
    report = validate_result(path, contract=contract)
    assert report["passed"], f"单行标量答案不应判退化：{report['issues']}"
