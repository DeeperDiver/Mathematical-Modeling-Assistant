from __future__ import annotations

from modeling_assistant.schemas.state import DynamicLTM, LtmSnapshot


def next_version(archive: list[LtmSnapshot], major_bump: bool = False) -> str:
    if not archive:
        return "v1.0"

    last = archive[-1].version.removeprefix("v")
    major_text, minor_text = last.split(".", maxsplit=1)
    major = int(major_text)
    minor = int(minor_text)

    if major_bump:
        return f"v{major + 1}.0"
    return f"v{major}.{minor + 1}"


def make_snapshot(
    dynamic_ltm: DynamicLTM,
    archive: list[LtmSnapshot],
    reason: str,
    commit_summary: str = "",
    major_bump: bool = False,
    checkpoint_id: str | None = None,
) -> LtmSnapshot:
    return LtmSnapshot(
        version=next_version(archive, major_bump=major_bump),
        dynamic_ltm=dynamic_ltm.model_copy(deep=True),
        commit_summary=commit_summary,
        reason=reason,
        checkpoint_id=checkpoint_id,
    )


def checkout_snapshot(archive: list[LtmSnapshot], version: str) -> DynamicLTM:
    for snapshot in archive:
        if snapshot.version == version:
            return snapshot.dynamic_ltm.model_copy(deep=True)
    raise ValueError(f"Snapshot version not found: {version}")


def archive_summary(archive: list[LtmSnapshot]) -> list[dict]:
    """返回仅含摘要信息的轻量视图，供 Agent 浏览历史时使用。

    默认只暴露 version、commit_summary、reason、created_at，
    不包含完整的 dynamic_ltm，避免 Token 爆炸。
    """
    return [
        {
            "version": s.version,
            "commit_summary": s.commit_summary or s.reason,
            "created_at": s.created_at.isoformat(),
        }
        for s in archive
    ]


def get_checkpoint_id_for_version(
    archive: list[LtmSnapshot], version: str
) -> str | None:
    """获取指定版本对应的 LangGraph checkpoint_id，用于 thread_ts 状态回滚。"""
    for s in archive:
        if s.version == version:
            return s.checkpoint_id
    return None
