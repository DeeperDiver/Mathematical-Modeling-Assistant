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
