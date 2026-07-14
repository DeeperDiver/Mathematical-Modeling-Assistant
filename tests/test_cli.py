from __future__ import annotations

from modeling_assistant.cli import _collect_attachment_paths


def test_collect_attachment_paths_files_and_directories(tmp_path):
    """_collect_attachment_paths 应同时支持文件和目录递归收集。"""
    file_a = tmp_path / "data.csv"
    file_a.write_text("a,b\n1,2")
    subdir = tmp_path / "extras"
    subdir.mkdir()
    file_b = subdir / "notes.txt"
    file_b.write_text("notes")

    result = _collect_attachment_paths([str(tmp_path)])
    assert str(file_a) in result
    assert str(file_b) in result
