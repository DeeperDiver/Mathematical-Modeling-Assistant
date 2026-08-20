"""运行过程记录（V17）。

目标：让用户拿到报告就能重新评估建模方案。为此：
- 每个关键节点把「输入摘要 / 输出 / 决策」写入结构化 process_log；
- 建模阶段（mathematician / realist / arbiter / clarifier / milestone_reviewer_1）
  逐轮留痕：候选方案、评分、verdict、选中的方案、提交的 LTM、评审结论；
- 渲染的 system prompt 存档到 outputs/logs/prompts/，供回溯「模型当时看到了什么」；
- 每条记录先落盘 JSONL（崩溃也不丢），最后由 CLI 或脚本生成 Markdown 报告。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProcessLogEntry(BaseModel):
    """单条运行过程记录。"""

    seq: int = 0
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: str = ""  # 节点名：mathematician / realist / clarifier ...
    phase: str = ""  # 当时的 control.phase
    event: str = ""  # 事件名：plans_generated / ltm_committed ...
    summary: str = ""  # 一句话摘要（时间线可读）
    details: dict[str, Any] = Field(default_factory=dict)


def make_entry(
    control: Any,
    stage: str,
    event: str,
    summary: str,
    details: dict[str, Any] | None = None,
    seq: int = 0,
) -> ProcessLogEntry:
    phase = ""
    if control is not None:
        try:
            phase = control.phase or ""
        except Exception:
            phase = ""
    return ProcessLogEntry(
        seq=seq,
        stage=stage,
        phase=phase,
        event=event,
        summary=summary,
        details=dict(details or {}),
    )


def _logs_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_log_line(output_dir: str | Path, entry: ProcessLogEntry) -> Path | None:
    """把单条记录追加到 outputs/logs/process_log.jsonl（先落盘，崩溃不丢）。"""
    try:
        log_dir = _logs_dir(output_dir)
        path = log_dir / "process_log.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return path
    except Exception as exc:
        logger.warning("process_log 落盘失败: %s", exc)
        return None


def archive_prompt(
    output_dir: str | Path,
    stage: str,
    tag: str,
    prompt_text: str,
) -> Path | None:
    """把渲染后的 system prompt 存档（建模阶段回溯「模型看到了什么」）。"""
    if not prompt_text:
        return None
    try:
        log_dir = _logs_dir(output_dir)
        prompts_dir = log_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
        path = prompts_dir / f"{stage}_{safe_tag}.md"
        path.write_text(prompt_text, encoding="utf-8")
        return path
    except Exception as exc:
        logger.warning("prompt 存档失败 %s: %s", stage, exc)
        return None


def load_process_log(output_dir: str | Path) -> list[ProcessLogEntry]:
    """从 JSONL 恢复全部记录（崩溃后重建报告用）。"""
    path = Path(output_dir) / "logs" / "process_log.jsonl"
    entries: list[ProcessLogEntry] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(ProcessLogEntry.model_validate_json(line))
        except Exception as exc:
            logger.warning("process_log 行解析失败，跳过: %s", exc)
    return entries


# ── 报告生成 ──────────────────────────────────────────────────────────

_MODELING_STAGES = {
    "mathematician",
    "realist",
    "arbiter",
    "clarifier",
    "milestone_reviewer_1",
}


def _render_details(details: dict[str, Any]) -> str:
    """按事件类型渲染 details 为 Markdown 片段。"""
    lines: list[str] = []
    if not details:
        return ""

    plans = details.get("plans")
    if isinstance(plans, list) and plans:
        lines.append("")
        lines.append("| 方案 | 创新 | 可行 | 综合 |")
        lines.append("|---|---|---|---|")
        for p in plans:
            inn = p.get("innovation_score", p.get("innovation", "?"))
            fea = p.get("feasibility_score", p.get("feasibility", "?"))
            total = "—"
            try:
                total = round(0.5 * float(inn) + 0.5 * float(fea), 1)
            except (TypeError, ValueError):
                pass
            lines.append(
                f"| {p.get('id', '?')}：{p.get('title', '')[:30]} | {inn} | {fea} | {total} |"
            )

    evals = details.get("evaluations")
    if isinstance(evals, list) and evals:
        lines.append("")
        lines.append("| 方案 | 创新 | 可行 | verdict | 反馈 |")
        lines.append("|---|---|---|---|---|")
        for e in evals:
            lines.append(
                f"| {e.get('plan_id', '?')} | {e.get('innovation_score', '?')} | "
                f"{e.get('feasibility_score', '?')} | {e.get('verdict', '?')} | "
                f"{str(e.get('feedback', ''))[:40]} |"
            )

    assumptions = details.get("assumptions")
    if isinstance(assumptions, list) and assumptions:
        lines.append("")
        lines.append("假设：")
        lines.extend(f"- {a}" for a in assumptions)
    equations = details.get("equations")
    if isinstance(equations, list) and equations:
        lines.append("")
        lines.append("公式/方程：")
        lines.extend(f"- {e}" for e in equations)
    for key in ("objective", "solution_outline", "commit_summary", "commit_message"):
        val = details.get(key)
        if val:
            lines.append(f"- **{key}**：{val}")

    # 其余标量/短列表字段
    scalar_keys = {
        "debate_round",
        "selected_plan_id",
        "innovation_score",
        "feasibility_score",
        "innovation_threshold",
        "feasibility_threshold",
        "need_rebrainstorm",
        "rollback_to_version",
        "action",
        "reason",
        "approval",
        "verdict",
        "version",
        "score",
        "decision",
        "run_id",
        "result_file",
        "findings_count",
        "llm_verdict",
        "prompt_file",
    }
    for key in scalar_keys:
        if key in details and details[key] not in (None, ""):
            lines.append(f"- **{key}**：{details[key]}")
    return "\n".join(lines)


def build_process_report(
    entries: list[ProcessLogEntry],
    meta: dict[str, Any] | None = None,
) -> str:
    """把 process_log 渲染为可读 Markdown 报告。"""
    lines: list[str] = ["# 运行过程报告（Process Report）", ""]
    meta = dict(meta or {})
    if meta:
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        for key, value in meta.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    lines.append("## 一、时间线")
    lines.append("")
    lines.append("| # | 时间(UTC) | 阶段 | 事件 | 摘要 |")
    lines.append("|---|---|---|---|---|")
    for e in entries:
        ts = e.ts[11:19] if len(e.ts) >= 19 else e.ts
        lines.append(
            f"| {e.seq} | {ts} | {e.stage} | {e.event} | {e.summary.replace('|', '｜')} |"
        )
    lines.append("")

    modeling = [e for e in entries if e.stage in _MODELING_STAGES]
    if modeling:
        lines.append("## 二、建模阶段详细（可据此重新评估方案）")
        lines.append("")
        for i, e in enumerate(modeling, start=1):
            lines.append(f"### 2.{i} [{e.stage}] {e.event}")
            lines.append("")
            lines.append(f"**摘要**：{e.summary}")
            rendered = _render_details(e.details)
            if rendered:
                lines.append("")
                lines.append(rendered)
            lines.append("")

    execution = [
        e
        for e in entries
        if e.stage
        in {
            "coder",
            "result_reviewer",
            "reflection",
            "load_bearing_analyzer",
            "sub_question_acceptance",
            "cross_sub_question",
            "hitl_modeling",
            "split_sub_questions",
        }
    ]
    if execution:
        lines.append("## 三、执行、验收与小题循环")
        lines.append("")
        for e in execution:
            lines.append(f"### [{e.stage}] {e.event}")
            lines.append("")
            lines.append(f"**摘要**：{e.summary}")
            rendered = _render_details(e.details)
            if rendered:
                lines.append("")
                lines.append(rendered)
            lines.append("")

    final = [
        e
        for e in entries
        if e.stage in {"writer", "final_reviewer", "hitl_final", "hitl_architecture", "hitl_arbitration"}
    ]
    if final:
        lines.append("## 四、架构确认、成稿与终审")
        lines.append("")
        for e in final:
            lines.append(f"### [{e.stage}] {e.event}")
            lines.append("")
            lines.append(f"**摘要**：{e.summary}")
            rendered = _render_details(e.details)
            if rendered:
                lines.append("")
                lines.append(rendered)
            lines.append("")

    return "\n".join(lines)


def write_process_report(
    output_dir: str | Path,
    entries: list[ProcessLogEntry],
    meta: dict[str, Any] | None = None,
) -> Path:
    """生成并保存 outputs/logs/process_report.md。"""
    log_dir = _logs_dir(output_dir)
    path = log_dir / "process_report.md"
    path.write_text(build_process_report(entries, meta=meta), encoding="utf-8")
    return path


def summarize_usage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 usage.jsonl / runtime.usage_log 的 token 消耗。"""
    calls = len(entries)
    prompt = sum(int(e.get("prompt_tokens", 0) or 0) for e in entries)
    completion = sum(int(e.get("completion_tokens", 0) or 0) for e in entries)
    cache_hit = sum(int(e.get("cache_hit_tokens", 0) or 0) for e in entries)
    cache_miss = sum(int(e.get("cache_miss_tokens", 0) or 0) for e in entries)
    by_node: dict[str, dict[str, int]] = {}
    for e in entries:
        name = str(e.get("prompt_name", "?"))
        node = by_node.setdefault(name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        node["calls"] += 1
        node["prompt_tokens"] += int(e.get("prompt_tokens", 0) or 0)
        node["completion_tokens"] += int(e.get("completion_tokens", 0) or 0)
    top = sorted(
        by_node.items(), key=lambda kv: kv[1]["completion_tokens"], reverse=True
    )[:8]
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "cache_hit_rate": round(cache_hit / prompt, 4) if prompt else 0.0,
        "top_nodes_by_completion": [
            {
                "node": name,
                "calls": stat["calls"],
                "prompt_tokens": stat["prompt_tokens"],
                "completion_tokens": stat["completion_tokens"],
            }
            for name, stat in top
        ],
    }
