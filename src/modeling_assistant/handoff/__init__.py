"""编程手任务包：把"方案与实现架构"提取出来，交给另一个 AI 实现。"""

from modeling_assistant.handoff.spec import (
    build_architecture_spec_md,
    write_coder_task_package,
)

__all__ = ["build_architecture_spec_md", "write_coder_task_package"]
