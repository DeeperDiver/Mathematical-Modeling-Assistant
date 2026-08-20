"""V16 ArXiv 检索查询构造测试：关键词优先 + 题面截断。"""

from __future__ import annotations

from modeling_assistant.agents.searcher import ArxivSearcher, SearchQuery


def test_build_query_uses_keywords_and_truncates_statement():
    """有关键词时取前 5 个，题面只补充前 150 字。"""
    query = SearchQuery(
        keywords=["绿色物流", "车辆路径", "时间窗", "异构车队", "碳排放", "第六个词"],
        problem_statement="长题面" * 200,
        max_results=5,
    )
    result = ArxivSearcher._build_search_query(query)
    # 关键词只取前 5 个
    assert "绿色物流" in result
    assert "第六个词" not in result
    # 题面截断到 150 字
    statement_part = result.replace("绿色物流 车辆路径 时间窗 异构车队 碳排放", "")
    assert len(statement_part.strip()) <= 150


def test_build_query_without_keywords_truncates_statement():
    """无关键词时题面截断到 200 字。"""
    query = SearchQuery(
        keywords=[],
        problem_statement="长题面" * 500,
        max_results=5,
    )
    result = ArxivSearcher._build_search_query(query)
    assert len(result) <= 200


def test_build_query_fallback_default():
    """关键词与题面都为空时回退默认查询。"""
    result = ArxivSearcher._build_search_query(SearchQuery())
    assert result == "mathematical modeling"


def test_build_query_strips_empty_keywords():
    """空白关键词应被过滤。"""
    query = SearchQuery(keywords=["", "  ", "车辆路径"])
    result = ArxivSearcher._build_search_query(query)
    assert result == "车辆路径"
