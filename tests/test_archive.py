from modeling_assistant.memory.archive import checkout_snapshot, make_snapshot
from modeling_assistant.schemas.state import DynamicLTM


def test_make_snapshot_appends_versions_without_mutating_source():
    original = DynamicLTM(objective="first")
    snapshot_1 = make_snapshot(original, [], reason="initial")
    original.objective = "changed"
    snapshot_2 = make_snapshot(original, [snapshot_1], reason="second")

    assert snapshot_1.version == "v1.0"
    assert snapshot_1.dynamic_ltm.objective == "first"
    assert snapshot_2.version == "v1.1"


def test_checkout_snapshot_returns_copy():
    snapshot = make_snapshot(DynamicLTM(objective="stable"), [], reason="initial")
    checked_out = checkout_snapshot([snapshot], "v1.0")
    checked_out.objective = "local edit"

    assert snapshot.dynamic_ltm.objective == "stable"


def test_major_bump_creates_v2_0():
    """major_bump=True 应将版本号从 v1.x 跳到 v2.0。"""
    original = DynamicLTM(objective="initial")
    snapshot_v1 = make_snapshot(original, [], reason="initial")
    assert snapshot_v1.version == "v1.0"

    # 后续 minor bump
    snapshot_v1_1 = make_snapshot(original, [snapshot_v1], reason="minor change")
    assert snapshot_v1_1.version == "v1.1"

    # major bump → v2.0
    snapshot_v2 = make_snapshot(
        original,
        [snapshot_v1, snapshot_v1_1],
        reason="objective fundamentally changed",
        major_bump=True,
    )
    assert snapshot_v2.version == "v2.0"


def test_major_bump_from_empty_archive():
    """空 archive 上 major_bump 应得到 v1.0（next_version 的默认行为）。"""
    snapshot = make_snapshot(
        DynamicLTM(objective="x"),
        [],
        reason="initial major",
        major_bump=True,
    )
    assert snapshot.version == "v1.0"
