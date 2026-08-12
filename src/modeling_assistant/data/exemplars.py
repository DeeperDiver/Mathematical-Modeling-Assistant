"""优秀论文知识库的加载与保存：L1 卡片、L2 题型指南、L3 全局偏好。"""

from __future__ import annotations

import logging
from pathlib import Path

from modeling_assistant.schemas.state import (
    ExemplarPaper,
    GlobalStyleProfile,
    TypeStyleGuide,
)

logger = logging.getLogger(__name__)


def load_cards(cards_dir: str | Path) -> list[ExemplarPaper]:
    """扫描目录下所有 *.json 卡片；损坏文件跳过并告警。"""
    cards: list[ExemplarPaper] = []
    root = Path(cards_dir)
    if not root.exists():
        return cards
    for path in sorted(root.glob("*.json")):
        try:
            cards.append(ExemplarPaper.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("卡片加载失败 %s: %s", path, exc)
    return cards


def load_guides(guides_dir: str | Path) -> list[TypeStyleGuide]:
    """扫描目录下所有题型指南。"""
    guides: list[TypeStyleGuide] = []
    root = Path(guides_dir)
    if not root.exists():
        return guides
    for path in sorted(root.glob("*.json")):
        try:
            guides.append(TypeStyleGuide.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("题型指南加载失败 %s: %s", path, exc)
    return guides


def load_global_profile(profile_path: str | Path) -> GlobalStyleProfile:
    """读取 L3 全局偏好（YAML）；缺失或损坏时返回空偏好。"""
    path = Path(profile_path)
    if not path.exists():
        return GlobalStyleProfile()
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return GlobalStyleProfile.model_validate(data)
    except Exception as exc:
        logger.warning("全局偏好加载失败 %s: %s", path, exc)
        return GlobalStyleProfile()


def save_card(card: ExemplarPaper, cards_dir: str | Path) -> Path:
    """保存单篇卡片为 {id}.json。"""
    root = Path(cards_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{card.id}.json"
    path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return path


def save_guide(guide: TypeStyleGuide, guides_dir: str | Path) -> Path:
    """保存题型指南为 {problem_type}_{contest}.json（contest 为空时仅题型名）。"""
    root = Path(guides_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{guide.contest}" if guide.contest else ""
    path = root / f"{guide.problem_type}{suffix}.json"
    path.write_text(guide.model_dump_json(indent=2), encoding="utf-8")
    return path
