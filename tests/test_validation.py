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
