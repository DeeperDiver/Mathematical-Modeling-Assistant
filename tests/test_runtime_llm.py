"""V15.1 LLM 调用健壮性测试：空 content 触发重试、max_tokens 传递。"""

from __future__ import annotations

from pathlib import Path

from modeling_assistant.agents.runtime import AgentRuntime
from modeling_assistant.config.settings import AppSettings
from modeling_assistant.schemas.responses import MathematicianResponse
from modeling_assistant.schemas.state import DynamicLTM, StaticLTM


def _runtime(tmp_path: Path, monkeypatch) -> AgentRuntime:
    monkeypatch.setenv("FAKE_TEST_KEY", "sk-test")
    settings = AppSettings(
        output_dir=tmp_path,
        api_key_env="FAKE_TEST_KEY",
        llm_max_tokens=4096,
    )
    return AgentRuntime.from_settings(settings)


def test_runtime_preserves_versioned_vendor_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TEST_KEY", "sk-test")
    runtime = AgentRuntime.from_settings(AppSettings(
        output_dir=tmp_path,
        api_key_env="FAKE_TEST_KEY",
        api_base_url="https://open.bigmodel.cn/api/paas/v4",
    ))
    assert str(runtime.client.base_url) == "https://open.bigmodel.cn/api/paas/v4/"


def test_runtime_adds_v1_to_bare_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TEST_KEY", "sk-test")
    runtime = AgentRuntime.from_settings(AppSettings(
        output_dir=tmp_path,
        api_key_env="FAKE_TEST_KEY",
        api_base_url="https://api.deepseek.com",
    ))
    assert str(runtime.client.base_url) == "https://api.deepseek.com/v1/"


def test_invoke_passes_reasoning_effort_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TEST_KEY", "sk-test")
    runtime = AgentRuntime.from_settings(AppSettings(
        output_dir=tmp_path,
        api_key_env="FAKE_TEST_KEY",
        reasoning_effort="max",
        reasoning_effort_overrides={"arbiter": "high"},
    ))
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_stream(["ok"])

    monkeypatch.setattr(runtime.client.chat.completions, "create", fake_create)
    runtime.invoke(
        "arbiter",
        {
            "static_ltm": StaticLTM(raw_problem="测试"),
            "dynamic_ltm": DynamicLTM(),
        },
        system_prompt="p",
    )

    assert captured["reasoning_effort"] == "high"
    assert runtime.settings.reasoning_effort_for("coder") == "max"


class _FakeMsg:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMsg(content)
        self.finish_reason = "stop"


class _FakeUsage:
    pass


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str | None):
        self.finish_reason = finish_reason
        self.delta = type("Delta", (), {"content": content})()


class _FakeStreamChunk:
    """模拟 stream 响应块。"""

    def __init__(self, content: str = "", finish_reason: str | None = None, usage=None):
        self.choices = (
            [_FakeChoice(content, finish_reason)] if content or finish_reason else []
        )
        self.usage = usage


def _fake_stream(responses: list[str]) -> list:
    """把若干完整 content 转成流式块序列（每个块一段 + 尾块带 usage）。"""
    chunks = []
    for text in responses:
        chunks.append(_FakeStreamChunk(content=text))
    chunks.append(_FakeStreamChunk(finish_reason="stop", usage=_FakeUsage()))
    return chunks


def test_invoke_structured_retries_on_empty_content(tmp_path, monkeypatch):
    """首次返回空 content 应触发重试，第二次成功解析。"""
    runtime = _runtime(tmp_path, monkeypatch)
    calls = {"n": 0, "max_tokens": None}

    def fake_create(model, messages, temperature, max_tokens, stream, stream_options):
        calls["n"] += 1
        calls["max_tokens"] = max_tokens
        if calls["n"] == 1:
            return _fake_stream([""])
        return _fake_stream(
            [
                '{"plans": ['
                '{"id": "p1", "title": "基线方案"},'
                '{"id": "p2", "title": "主方案"},'
                '{"id": "p3", "title": "挑战方案"},'
                '{"id": "p4", "title": "替代方案"}'
                ']}'
            ]
        )

    monkeypatch.setattr(
        runtime.client.chat.completions,
        "create",
        fake_create,
    )

    state = {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "dynamic_ltm": DynamicLTM(),
    }
    result = runtime.invoke_structured("mathematician", state, MathematicianResponse)

    assert isinstance(result, MathematicianResponse)
    assert calls["n"] == 2
    # max_tokens 应显式传给 API
    assert calls["max_tokens"] == runtime.settings.max_tokens_for("mathematician")
    assert calls["max_tokens"] == 32768  # V17 分节点覆盖值


def test_invoke_passes_per_node_max_tokens_fallback(tmp_path, monkeypatch):
    """V17：有覆盖的节点用节点上限，未覆盖的节点用全局默认。"""
    runtime = _runtime(tmp_path, monkeypatch)
    caps: dict[str, int] = {}

    def fake_create(model, messages, temperature, max_tokens, stream, stream_options):
        caps["arbiter"] = max_tokens
        caps["unknown"] = max_tokens
        return _fake_stream(["{}"])

    monkeypatch.setattr(runtime.client.chat.completions, "create", fake_create)
    state = {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "dynamic_ltm": DynamicLTM(),
    }
    # 有覆盖节点
    runtime.invoke("arbiter", state, system_prompt="p")
    # 未覆盖节点（走 invoke，直接返回文本）
    runtime.invoke("no_such_node", state, system_prompt="p")

    assert caps["arbiter"] == runtime.settings.max_tokens_for("arbiter") == 4096
    assert caps["unknown"] == runtime.settings.llm_max_tokens == 4096
    assert runtime.settings.max_tokens_for("searcher") == 2048
    assert runtime.settings.max_tokens_for("coder") == 32768


def test_invoke_structured_all_empty_raises(tmp_path, monkeypatch):
    """连续返回空 content 时应最终抛出 RuntimeError。"""
    runtime = _runtime(tmp_path, monkeypatch)

    def fake_create(model, messages, temperature, max_tokens, stream, stream_options):
        return _fake_stream([""])

    monkeypatch.setattr(runtime.client.chat.completions, "create", fake_create)

    state = {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "dynamic_ltm": DynamicLTM(),
    }
    import pytest

    with pytest.raises(RuntimeError):
        runtime.invoke_structured("mathematician", state, MathematicianResponse, max_retries=1)


def test_usage_recorded_to_log_and_jsonl(tmp_path, monkeypatch):
    """V17：每次调用的 token 用量（含缓存命中/未命中）应记录并落盘 usage.jsonl。"""
    import json as _json

    runtime = _runtime(tmp_path, monkeypatch)

    class _Usage:
        prompt_tokens = 100
        completion_tokens = 200
        total_tokens = 300
        prompt_cache_hit_tokens = 60
        prompt_cache_miss_tokens = 40

    def fake_create(model, messages, temperature, max_tokens, stream, stream_options):
        return [
            _FakeStreamChunk(
                content=(
                    '{"plans": ['
                    '{"id": "p1", "title": "基线方案"},'
                    '{"id": "p2", "title": "主方案"},'
                    '{"id": "p3", "title": "挑战方案"},'
                    '{"id": "p4", "title": "替代方案"}'
                    ']}'
                ),
                finish_reason="stop",
            ),
            # 真实 API 的 usage 尾块：无 choices、带 usage
            _FakeStreamChunk(usage=_Usage()),
        ]

    monkeypatch.setattr(runtime.client.chat.completions, "create", fake_create)
    state = {
        "static_ltm": StaticLTM(raw_problem="测试"),
        "dynamic_ltm": DynamicLTM(),
    }
    runtime.invoke_structured("mathematician", state, MathematicianResponse)

    assert len(runtime.usage_log) == 1
    usage = runtime.usage_log[0]
    assert usage["prompt_name"] == "mathematician"
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 200
    assert usage["cache_hit_tokens"] == 60
    assert usage["cache_miss_tokens"] == 40
    assert usage["finish_reason"] == "stop"

    jsonl = tmp_path / "logs" / "usage.jsonl"
    assert jsonl.exists()
    lines = [line for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert _json.loads(lines[-1])["prompt_name"] == "mathematician"
