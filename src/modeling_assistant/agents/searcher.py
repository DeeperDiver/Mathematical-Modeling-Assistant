"""论文检索接口 —— 抽象基类 + Stub 实现。

后续接入 ArXiv API、Web Search 等真实检索时，只需实现 Searcher 接口即可。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchResult:
    """单条检索结果。"""

    title: str
    authors: str = ""
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


def validate_search_results(
    results: list[SearchResult],
    keywords: list[str],
    min_relevance_keywords: int = 1,
) -> list[SearchResult]:
    """校验检索结果：去重、过滤占位、相关性检查。

    Args:
        results: 原始检索结果列表。
        keywords: 用于相关性校验的关键词列表。
        min_relevance_keywords: 摘要中至少命中几个关键词才算相关。

    Returns:
        校验通过的结果列表。若全部未通过，返回空列表。
    """
    validated: list[SearchResult] = []
    seen_titles: set[str] = set()

    for r in results:
        title = r.title.strip()
        summary = r.summary.strip()

        # 1. 去重（按标题，大小写不敏感）
        title_key = title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # 2. 过滤占位/空结果
        if "[占位]" in title or "[占位]" in summary:
            continue
        if not title or not summary:
            continue

        # 3. 相关性校验：摘要至少命中 N 个关键词
        if keywords:
            summary_lower = summary.lower()
            hits = sum(1 for kw in keywords if kw.lower() in summary_lower)
            if hits < min_relevance_keywords:
                logger.info(
                    "检索结果 '%s' 未通过相关性校验（命中 %d/%d 个关键词）。",
                    title,
                    hits,
                    min_relevance_keywords,
                )
                continue

        validated.append(r)

    return validated


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

    @staticmethod
    def _build_search_query(query: SearchQuery) -> str:
        """构造 ArXiv 查询串：关键词优先 + 题面截断。

        V16 修复：原实现把完整 problem_statement（可长达数千字）直接拼进查询，
        导致 ArXiv API 返回 400/414（URL 过长/非法字符）。现在：
        - 关键词取前 5 个，题面只补充前 150 字；
        - 无关键词时题面截断到 200 字，空则回退默认查询。
        """
        keywords = [k.strip() for k in (query.keywords or []) if k.strip()]
        problem_statement = (query.problem_statement or "").strip()
        if keywords:
            parts = keywords[:5]
            if problem_statement:
                parts.append(problem_statement[:150])
            return " ".join(parts)
        if problem_statement:
            return problem_statement[:200]
        return "mathematical modeling"

    def search(self, query: SearchQuery) -> list[SearchResult]:
        import arxiv

        search_query = self._build_search_query(query)

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
                authors=", ".join(a.name for a in (paper.authors or [])[:5]),
                source="arxiv",
                summary=(paper.summary or "")[:500],
                url=paper.entry_id,
            ))
        return results
