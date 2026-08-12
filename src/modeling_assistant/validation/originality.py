"""输出原创性护栏：检查 Writer 输出与示例库（卡片摘录 + 原文）的 n-gram 重合度。

用于防止 Exemplar 注入导致模型整句/近义复制示例表达。重合率超过阈值时
返回警告，由 Writer 节点写入 prompt_audit，论文仍正常产出（可人工复核）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from modeling_assistant.schemas.state import ExemplarContext

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    """去空白与常见标点，保留中英文与数字。"""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def ngrams(text: str, n: int) -> set[str]:
    """字符 n-gram（对中文与英文混排均适用）。"""
    cleaned = _clean(text)
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def overlap_ratio(output_text: str, reference_texts: Iterable[str], n: int = 8) -> float:
    """输出文本中被任一参考文本共享的 n-gram 占比。"""
    out_grams = ngrams(output_text, n)
    if not out_grams:
        return 0.0
    ref_grams: set[str] = set()
    for ref in reference_texts:
        ref_grams |= ngrams(ref, n)
    if not ref_grams:
        return 0.0
    matched = len(out_grams & ref_grams)
    return matched / len(out_grams)


def check_originality(
    output_text: str,
    reference_texts: Iterable[str],
    *,
    n: int = 8,
    threshold: float = 0.15,
) -> dict:
    """返回查重报告。"""
    ratio = overlap_ratio(output_text, reference_texts, n=n)
    out_grams = ngrams(output_text, n)
    return {
        "passed": ratio < threshold,
        "overlap_ratio": round(ratio, 4),
        "total_ngrams": len(out_grams),
        "threshold": threshold,
        "ngram_size": n,
    }


def _read_reference_text(path: Path) -> str:
    """读取参考文本；PDF 尝试 pdfplumber，失败返回空。"""
    try:
        if path.suffix.lower() == ".pdf":
            import pdfplumber

            parts: list[str] = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text:
                        parts.append(text)
            return "\n".join(parts)
        if path.suffix.lower() in (".tex", ".md", ".txt"):
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("查重参考文本读取失败 %s: %s", path, exc)
    return ""


def check_writer_output(
    latex_text: str,
    exemplars: ExemplarContext | None,
    *,
    n: int = 8,
    threshold: float = 0.15,
) -> list[str]:
    """检查 Writer 的 LaTeX 输出与示例库的重合度，返回警告列表（空=通过）。"""
    if not latex_text or exemplars is None or not exemplars.active:
        return []

    references: list[str] = []
    for card in exemplars.cards:
        references.extend(card.quotes or [])
        if card.source_path:
            src = Path(card.source_path)
            if src.exists():
                text = _read_reference_text(src)
                if text:
                    references.append(text)

    if not references:
        return []

    report = check_originality(latex_text, references, n=n, threshold=threshold)
    if report["passed"]:
        return []
    return [
        f"疑似复制示例表达：{n}-gram 重合率 {report['overlap_ratio']:.2%} "
        f"超过阈值 {threshold:.0%}（共 {report['total_ngrams']} 个 n-gram）。"
        "请检查论文中是否整句引用了示例论文的句子，并改写为原创表达。"
    ]
