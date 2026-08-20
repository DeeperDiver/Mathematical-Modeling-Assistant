"""优秀论文摄取与提炼。

输入「题目 + 优秀论文」对，输出 L1 单篇卡片；对同题型多篇卡片聚合生成
L2 题型指南。PDF 解析依赖 pdfplumber（可选）：缺失时跳过 PDF 并提示，
tex/md/txt 与现成 JSON 卡片不受影响。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from modeling_assistant.data.exemplars import load_cards, save_card, save_guide
from modeling_assistant.schemas.state import ExemplarPaper, TypeStyleGuide

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".tex", ".md", ".txt"}
PROBLEM_FILENAMES = ("problem.txt", "题目.txt")

# 简单的题型关键词（兜底，与 memory/exemplar_search 的判定规则保持一致）
_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "optimization": ("优化", "调度", "规划", "配送", "路径", "排班", "选址", "分配", "库存"),
    "physics": ("物理", "运动", "轨迹", "速度", "加速度", "材料", "干涉", "弹道", "外延"),
    "forecasting": ("预测", "时序", "时间序列", "趋势", "预报", "客流"),
    "evaluation": ("评价", "评估", "评分", "综合", "比较", "层次分析"),
    "data_mining": ("分类", "聚类", "挖掘", "识别", "检测", "特征", "异常"),
}


def extract_text(path: str | Path) -> str:
    """按扩展名提取论文文本。

    PDF 优先读取同目录 <name>.pdf.ocr.txt 缓存（扫描件 OCR 结果）；
    无缓存时依赖 pdfplumber，缺失时抛出 ImportError。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        ocr_cache = path.with_suffix(path.suffix + ".ocr.txt")
        if ocr_cache.exists():
            cached = ocr_cache.read_text(encoding="utf-8", errors="replace")
            if len(cached.strip()) > 200:
                return cached
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber 未安装，无法解析 PDF。请安装（pip install pdfplumber）"
                "或提供 .tex/.md/.txt 源码 / 现成 JSON 卡片。"
            ) from exc
        parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    parts.append(text)
        return "\n".join(parts)
    if suffix in (".tex", ".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def extract_structure(text: str, ext: str = "") -> dict[str, str]:
    """启发式提取章节骨架。

    tex/md 用标题正则；纯文本用「数字编号 + 中文标题」行模式。
    """
    structure: dict[str, str] = {}
    if not text:
        return structure
    if ext in (".tex", ".md"):
        patterns = [
            re.compile(r"\\(?:sub)?section\{([^}]+)\}"),
            re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE),
        ]
        for pattern in patterns:
            for match in pattern.finditer(text):
                title = match.group(1).strip()
                if title and title not in structure:
                    structure[title] = ""
    else:
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^\d+(\.\d+)*[\s、.．]", line):
                title = re.sub(r"^\d+(\.\d+)*[\s、.．]", "", line).strip()
                # 截断过长的行，避免把正文当章节
                if 2 <= len(title) <= 30 and title not in structure:
                    structure[title] = ""
            if len(structure) >= 12:
                break
    return structure


def find_problem_file(directory: str | Path) -> Path | None:
    """在目录中查找题面文件（problem.txt / 题目.txt）。"""
    root = Path(directory)
    for name in PROBLEM_FILENAMES:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _fallback_problem_type(text: str, paper_text: str = "") -> str:
    """关键词兜底判定题型；无命中默认 data_mining（保守）。"""
    combined = text + "\n" + paper_text
    scores = {
        t: sum(1 for kw in kws if kw in combined)
        for t, kws in _TYPE_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "data_mining"


def ingest_paper(
    path: str | Path,
    problem_text: str = "",
    *,
    runtime: Any | None = None,
    contest: str = "",
    problem_type: str = "",
) -> ExemplarPaper | None:
    """把单篇论文提炼为 L1 卡片。

    runtime 提供 LLM 时走结构化提炼；否则（或 LLM 失败）生成确定性 fallback 卡片。
    """
    path = Path(path)
    # 现成 JSON 卡片直接校验入库，不需要文本提炼
    if path.suffix.lower() == ".json":
        try:
            return ExemplarPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("JSON 卡片加载失败 %s: %s", path, exc)
            return None
    try:
        paper_text = extract_text(path)
    except ImportError as exc:
        logger.warning("跳过 %s: %s", path, exc)
        return None
    if not paper_text.strip():
        logger.warning("跳过 %s: 未提取到文本", path)
        return None

    resolved_type = problem_type or _fallback_problem_type(problem_text, paper_text)
    card = _llm_ingest(path, paper_text, problem_text, resolved_type, contest, runtime)
    if card is None:
        card = ExemplarPaper(
            id=path.stem,
            title=path.stem,
            source_path=str(path),
            problem_type=resolved_type,
            contest=contest,
            structure=extract_structure(paper_text, path.suffix.lower()),
            tags=[resolved_type],
        )
        logger.info("LLM 提炼不可用，使用确定性 fallback 卡片: %s", path)
    return card


def _llm_ingest(
    path: Path,
    paper_text: str,
    problem_text: str,
    problem_type: str,
    contest: str,
    runtime: Any,
) -> ExemplarPaper | None:
    if runtime is None or getattr(runtime, "client", None) is None:
        return None
    template_path = Path(__file__).resolve().parents[1] / "prompts" / "templates" / "exemplar_ingest.md"
    template = template_path.read_text(encoding="utf-8")
    system_prompt = template.format(
        raw_problem=problem_text or "（未提供题面）",
        paper_title=path.stem,
        paper_text=paper_text[:20000],
        problem_type=problem_type,
        contest=contest,
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(
            runtime.invoke, "exemplar_ingest", {}, system_prompt
        )
        try:
            # 单次调用限时：DeepSeek 偶发挂起，快速失败进入重试/fallback
            raw = future.result(timeout=120)
            from modeling_assistant.agents.runtime import _extract_json

            data = json.loads(_extract_json(raw))
            # id 由文件名决定，保证唯一性与摄入幂等（可断点续传）
            data["id"] = path.stem
            data.setdefault("source_path", str(path))
            data.setdefault("problem_type", problem_type)
            data.setdefault("contest", contest)
            return ExemplarPaper.model_validate(data)
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, FutureTimeout):
                exc = TimeoutError("LLM 提炼超时（120s）")
            logger.warning(
                "LLM 卡片提炼失败 %s（attempt %d/3）: %s", path, attempt + 1, exc
            )
        finally:
            # 不等待后台线程（httpx 300s 超时仍在跑），避免 with 阻塞
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
    logger.warning("LLM 卡片提炼最终失败 %s: %s", path, last_exc)
    return None


def aggregate_guides(
    cards: list[ExemplarPaper],
    *,
    min_occurrences: int = 3,
    runtime: Any | None = None,
) -> list[TypeStyleGuide]:
    """按 (problem_type, contest) 分组聚合 L2 题型指南。

    共性字段只收「≥ min_occurrences 篇共有」的特征；不足的保留在卡片 highlights。
    runtime 可用时先尝试 LLM 二次提炼，失败/不可用时回退确定性统计。
    """
    groups: dict[tuple[str, str], list[ExemplarPaper]] = {}
    for card in cards:
        key = (card.problem_type or "unknown", card.contest or "")
        groups.setdefault(key, []).append(card)

    guides: list[TypeStyleGuide] = []
    for (problem_type, contest), group in sorted(groups.items()):
        guide = _llm_aggregate_group(problem_type, contest, group, runtime)
        if guide is None:
            guide = _deterministic_aggregate(problem_type, contest, group, min_occurrences)
        guides.append(guide)
    return guides


def _deterministic_aggregate(
    problem_type: str,
    contest: str,
    group: list[ExemplarPaper],
    min_occurrences: int,
) -> TypeStyleGuide:
    n = len(group)
    structure_counter: Counter[str] = Counter()
    for card in group:
        structure_counter.update(
            n for n in (_normalize_section_name(s) for s in card.structure.keys()) if n
        )
    figure_counter: Counter[str] = Counter()
    for card in group:
        figure_counter.update(f.figure_type for f in card.figures)
    pitfall_counter: Counter[str] = Counter()
    for card in group:
        pitfall_counter.update(card.pitfalls)

    # 动态阈值：组越大要求越高，防止大组把偶然出现的章节当共性
    threshold = max(min_occurrences, round(n * 0.2))
    common_structure = [s for s, c in structure_counter.items() if c >= threshold]
    variants = [s for s, c in structure_counter.items() if 2 <= c < threshold]
    recommended_figures = [f for f, c in figure_counter.items() if c >= threshold]
    common_pitfalls = [p for p, c in pitfall_counter.items() if c >= max(2, min_occurrences - 1)]

    # 文风基线：同一键下最常见取值，且出现次数 >= max(2, min-1)
    writing_keys: set[str] = set()
    for card in group:
        writing_keys.update(card.writing_style.keys())
    baseline: dict[str, str] = {}
    need = max(2, min_occurrences - 1)
    for key in sorted(writing_keys):
        values = Counter(card.writing_style.get(key, "") for card in group if card.writing_style.get(key))
        if values:
            value, count = values.most_common(1)[0]
            if count >= need:
                baseline[key] = value

    return TypeStyleGuide(
        problem_type=problem_type,
        contest=contest,
        common_structure=common_structure,
        structure_variants=variants,
        recommended_figures=recommended_figures,
        writing_baseline=baseline,
        common_pitfalls=common_pitfalls,
        exemplar_ids=[c.id for c in group],
        quality_score=round(sum(c.quality_score for c in group) / n, 3) if n else 0.5,
    )


def _normalize_section_name(name: str) -> str:
    """归一化章节名：消除论文间命名差异（"模型的建立"→"模型建立"等），
    使指南共性统计更准确、更可泛化。"""
    s = re.sub(r"[\s\u3000]+", "", name or "")
    if not s:
        return name or ""
    # 先去掉"的"等虚字，保证"模型的建立"能命中"模型建立"
    s = s.replace("的", "")
    canonical = {
        "摘要": "摘要",
        "问题重述": "问题重述",
        "问题分析": "问题分析",
        "模型假设": "模型假设",
        "符号说明": "符号说明",
        "模型建立": "模型建立",
        "模型求解": "模型求解",
        "模型检验": "模型检验",
        "灵敏度分析": "灵敏度分析",
        "结果分析": "结果分析",
        "模型评价": "模型评价",
        "模型推广": "模型推广",
    }
    for key in sorted(canonical, key=len, reverse=True):
        if key in s:
            return canonical[key]
    # 去掉"模型一/问题1"等编号前缀与括号尾注
    s = re.sub(
        r"^(模型[一二三四五六七八九十]+|问题[一二三四五六七八九十\d]+|第[一二三四五六七八九十\d]+[题问节])",
        "",
        s,
    )
    s = re.sub(r"[（(][^）)]*[)）]$", "", s)
    return s if len(s) >= 2 else ""


def _llm_aggregate_group(
    problem_type: str,
    contest: str,
    group: list[ExemplarPaper],
    runtime: Any,
) -> TypeStyleGuide | None:
    if runtime is None or getattr(runtime, "client", None) is None:
        return None
    template_path = (
        Path(__file__).resolve().parents[1] / "prompts" / "templates" / "exemplar_aggregate.md"
    )
    template = template_path.read_text(encoding="utf-8")
    cards_json = json.dumps(
        [
            {
                "id": c.id,
                "structure": c.structure,
                "figures": [f.model_dump() for f in c.figures],
                "writing_style": c.writing_style,
                "pitfalls": c.pitfalls,
            }
            for c in group
        ],
        ensure_ascii=False,
        indent=2,
    )
    system_prompt = template.format(
        problem_type=problem_type,
        contest=contest or "（未指定）",
        cards_json=cards_json,
    )
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(
        runtime.invoke, "exemplar_aggregate", {}, system_prompt
    )
    try:
        raw = future.result(timeout=90)
        from modeling_assistant.agents.runtime import _extract_json

        data = json.loads(_extract_json(raw))
        data.setdefault("problem_type", problem_type)
        data.setdefault("contest", contest)
        data.setdefault("exemplar_ids", [c.id for c in group])
        return TypeStyleGuide.model_validate(data)
    except Exception as exc:
        logger.warning("LLM 题型聚合失败 %s/%s: %s", problem_type, contest, exc)
        return None
    finally:
        future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
