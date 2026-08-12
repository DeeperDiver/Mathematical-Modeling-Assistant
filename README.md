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

## 优秀论文表达学习（Exemplar Learning System）

把「题目 + 优秀论文」对提炼为结构化表达知识（L1 单篇卡片 → L2 题型指南 → L3 全局偏好），
运行时检索并分级注入 Architect / Drawer / Writer / Reviewer 的 prompt，
提升论文结构、图表与文风，同时不污染建模正确性。

快速开始：

```powershell
# 1. 把论文放入 exemplars/raw/（同一目录可放 problem.txt 作为题面）
# 2. 摄入并聚合：
python scripts/ingest_exemplars.py --input exemplars/raw --output exemplars
# 3. 运行时启用：
python -m modeling_assistant.cli --problem "..." --exemplars-dir exemplars
```

常用脚本：

- `scripts/ingest_exemplars.py`：批量摄入论文，生成卡片并聚合题型指南。
- `scripts/leave_one_out_eval.py`：留一验证，量化每张卡片的独有贡献。

详细设计见 `docs/exemplar_learning_plan.md`。
