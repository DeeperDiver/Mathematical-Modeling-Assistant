# Modeling Assistant

基于 LangGraph 的数学建模 Multi-Agent 协作框架骨架。

当前版本实现了：

- Core State One：静态 LTM
- Core State Two：动态 LTM
- LTM Archive：追加式历史快照
- Control State：流程控制、评分、产物路径与 HITL 信号
- AppSettings：`.env` / 环境变量 / CLI 覆盖的统一配置
- AgentRuntime：未来 LLM、检索、绘图与执行能力的统一接入层
- PromptCatalog：本地 Markdown 模板渲染，动态注入 LTM
- 五阶段节点骨架与条件路由
- Clarifier 写入动态 LTM 前的自动快照
- Archive checkout / rollback 的服务函数

## 快速运行

```powershell
python -m modeling_assistant.cli --problem "给定城市交通流量数据，预测拥堵并优化信号灯配时。"
```

可选配置：

```powershell
python -m modeling_assistant.cli `
  --env-file .env `
  --llm-model deepseek-chat `
  --output-dir outputs `
  --problem "给定城市交通流量数据，预测拥堵并优化信号灯配时。"
```

`.env` 示例：

```text
MODELING_ASSISTANT_LLM_MODEL=deepseek-chat
MODELING_ASSISTANT_API_KEY_ENV=DEEPSEEK_API_KEY
MODELING_ASSISTANT_SEARCH_ENABLED=false
MODELING_ASSISTANT_OUTPUT_DIR=outputs
MODELING_ASSISTANT_MAX_DEBATE_ROUNDS=3
MODELING_ASSISTANT_INNOVATION_THRESHOLD=60
MODELING_ASSISTANT_FEASIBILITY_THRESHOLD=60
```

## 目录

```text
src/modeling_assistant/
  agents/       节点函数
  config/       配置读取
  graph/        LangGraph 构建与条件路由
  memory/       LTM 快照、版本号、回滚
  prompts/      Agent system prompt 模板
  schemas/      状态与领域模型
  cli.py        最小命令行入口
tests/          架构级测试
```

真实模型调用、论文检索、绘图和算法执行将在后续实现中接入 `AgentRuntime` 接口。
