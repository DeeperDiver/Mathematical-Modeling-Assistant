"""论文检索接口 —— 抽象基类 + Stub 实现。

后续接入 ArXiv API、Web Search 等真实检索时，只需实现 Searcher 接口即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class SearchResult:
    """单条检索结果。"""

    title: str
    source: str = ""
    summary: str = ""
    url: str | None = None


@dataclass(slots=True)
class SearchQuery:
    """检索查询参数。"""

    keywords: list[str] = field(default_factory=list)
    problem_statement: str = ""
    max_results: int = 5


class Searcher(ABC):
    """论文检索抽象基类。

    所有检索实现（ArXiv、Web、本地知识库）必须实现 search() 方法。
    """

    @abstractmethod
    def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行检索，返回结果列表。"""
        ...


class StubSearcher(Searcher):
    """占位检索器 —— 返回提示信息，后续替换为真实检索。"""

    def search(self, query: SearchQuery) -> list[SearchResult]:
        return [
            SearchResult(
                title="[占位] 参考模型",
                source="internal",
                summary="后续接入 ArXiv / Web 检索后替换为真实高价值文献摘要。",
            ),
            SearchResult(
                title="[占位] 相关方法",
                source="internal",
                summary=f"针对关键词 {query.keywords} 的检索结果将在接入真实检索后返回。",
            ),
        ]


class ArxivSearcher(Searcher):
    """基于 arxiv 库的真实论文检索器。"""

    def search(self, query: SearchQuery) -> list[SearchResult]:
        import arxiv

        search_query = query.problem_statement
        if query.keywords:
            search_query = " ".join(query.keywords) + " " + search_query

        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=query.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results: list[SearchResult] = []
        for paper in client.results(search):
            results.append(SearchResult(
                title=paper.title,
                source="arxiv",
                summary=(paper.summary or "")[:500],
                url=paper.entry_id,
            ))
        return results