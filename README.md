# Modeling Assistant

基于 LangGraph 的数学建模 Multi-Agent 协作框架：从赛题输入、数据画像、方案辩论，
到代码执行、实证反思、LaTeX 成稿与人类终审的完整闭环。

## 核心特性

- **四层 Graph State**：静态 LTM（问题/数据/文献）、动态 LTM（假设/符号/公式/目标）、
  LTM Archive（版本快照与回滚）、Control State（流程控制/评分/预算/HITL）
- **Empirical Layer**：独立于 LTM 的执行证据层，形成「假设 → 验证 → 修正」闭环
- **AgentRuntime**：LLM / 论文检索 / 代码执行 / 绘图的统一接入层（OpenAI 兼容，默认 DeepSeek）
- **PromptCatalog**：本地 Markdown 模板渲染，动态注入 LTM 与实证发现
- **五阶段节点骨架与条件路由**：发散→剪枝→澄清→并行执行→成稿，
  内置 Meta-Router、统一建模预算、预算耗尽强制 HITL 等防死循环机制
- **三层常量防线**：机器提取题面常量（problem_facts）→ Clarifier 写 LTM 前校验 →
  Coder 代码生成后 AST 扫描数值与列名
- **方法知识库**：内置数学建模规范知识库（选型决策树、题型防错、编码防错、
  图表规范），按节点/题型切片注入 Mathematician / Realist / Coder / Clarifier / Drawer，
  只影响领域判断，不改变图结构（`MODELING_ASSISTANT_METHOD_KNOWLEDGE_ENABLED=false` 可关闭）
- **国赛 LaTeX 模板 + 论文验收**：writer 按国赛（CUMCM）模板骨架输出章节文件
  （模板复制到 `outputs/paper/`，main.tex 保留封面/摘要/目录/三线表格式）；
  final_reviewer 先做确定性验收（章节/占位符/内部泄露/图片引用/编译），再调 LLM
  做灵活审查（数值一致性、图表解读、表达质量）；终审可输入 `rewrite <反馈>` 回 Writer
  重写（有预算防死循环）。每个问题章节强制以「问题小结」收尾
  （本题做了什么 → 得到什么 → 对下一题的支撑），并由确定性验收与终审双重把关；
  模板目录可用 `MODELING_ASSISTANT_PAPER_TEMPLATE_DIR` 指定。
- **假设审慎化与关键假设实验**：Clarifier 撰写的假设必须审慎、有依据，
  关键假设以 `【关键假设】` 标注（依据/风险/可验证性）；架构 HITL 逐条人工审核
  假设，可 `revise <反馈>` 打回 Clarifier 修改；Architect 规划阶段为每条关键
  假设产出扰动/对照实验（进入图表表格计划），final_reviewer 验收实验结论是否
  被正文引用，确保「假设若不成立结论是否依然成立」被论文回答。
- **承重结构分析（Load-Bearing Analysis）**：milestone 通过后、人审之前新增
  `load_bearing_analyzer` 节点，把「结论」与「结论所依赖的构造」显式连接成
  承重图（指标/方法库/模型/参数/阈值/抽象结构/假设/数据项），逐条给出验证状态、
  物理锚点、承重度与验证契约（校准/扰动/对照/交叉/案例/物证），按
  「承重度 × (1 − 验证完成度)」排序、根构造优先；Architect 按契约规划实验与
  可视化，Writer 让承重构造可见并逐条引用验证，paper_check 与 final_reviewer
  按契约对账（根构造缺验证、锚点缺口无可视化、单向结论无兜底均打回）。
  实证层证据按构造回流更新承重图，构成「规划 → 执行 → 证据 → 对账」闭环。
- **完整图表规划闭环**：Architect 在架构阶段规划全文图表（每张图声明
  `kind`（data 数据图 / flowchart 技术路线图 / diagram 结构图）、图注、目标章节、
  内容规格与必需标志，参考行文技艺的图片位置规划）→ Drawer 按 plan_id 唯一命名
  生成并登记图表注册表（技术路线图用 matplotlib 绘制，可借鉴
  `assets/figure_templates/` 科研模板）→ Writer 只按注册表引用 → paper_check 对账
  （必需图缺失/未引用即打回）。
- **环境健康检查**：`python -m modeling_assistant.cli --doctor` 检查依赖、论文编译器、
  PDF 视觉工具与 API Key 配置。
- **ResultReviewer**：零 LLM 的确定性结果校验（NaN/Inf、合理区间、常量列、R² 等）
- **结果注册表与论文数字校验**：每题验收通过即锁定唯一权威结果
  （Result Manifest，含指标快照/来源 run_id）；终审前做「论文数字 ↔ 结果文件」
  机器比对（跨文件污染检测、表格引用断链检测），杜绝"Q1 章节抄 Q2 参数"类事故。
- **运行过程记录与 token 记账**：逐节点留痕（建模阶段含候选方案/评分/
  提交的 LTM + system prompt 存档），每次 LLM 调用的输入/输出/缓存命中写入
  `outputs/logs/usage.jsonl`；运行报告头部汇总 token 消耗与输出 top 节点。
- **分节点输出预算**：coder/writer/clarifier 与建模核心 mathematician/realist
  保留大 max_tokens（32K），其余小节点压到 2–16K，避免 reasoner 推理空转
  （`max_tokens_for` 按节点取值）。
- **Exemplar Learning System**：从「题目 + 优秀论文」对学习论文结构、图表与文风（见下文）
- **HITL**：架构确认 / 仲裁回滚 / 建模预算 / 终稿审查四处人工介入，
  CLI 交互或 `--auto-approve`

## 快速运行

```powershell
# 环境健康检查
python -m modeling_assistant.cli --doctor

# 完整建模
python -m modeling_assistant.cli --problem "给定城市交通流量数据，预测拥堵并优化信号灯配时。"
```

可选配置：

```powershell
python -m modeling_assistant.cli `
  --env-file .env `
  --llm-model deepseek-chat `
  --output-dir outputs `
  --exemplars-dir exemplars `
  --problem "给定城市交通流量数据，预测拥堵并优化信号灯配时。"
```

## 优秀论文表达学习（Exemplar Learning System）

把「题目 + 优秀论文」对提炼为结构化表达知识，运行时检索并分级注入
Architect / Drawer / Writer / Reviewer 的 prompt，提升论文结构、图表与文风，
同时不污染建模正确性（公式、数值与方法仍只走 LTM + Coder 验证链）。

### 知识库结构

```text
exemplars/
  raw/      论文原文（已 gitignore，可放 .pdf / .tex / .md / .txt）
  cards/    L1 单篇卡片（JSON，由摄入脚本生成或手工编写）
  guides/   L2 题型指南（JSON，同题型 ≥3 篇聚合生成）
  profile.yaml  L3 全局风格偏好（用户个人设定，最上层软约束）
```

同目录下放置 `problem.txt` / `题目.txt` 可作为该论文对应的题面。

### 使用流程

```powershell
# 1. 把论文放入 exemplars/raw/（可选：同目录放 problem.txt 作为题面）
# 2. 摄入并聚合：
python scripts/ingest_exemplars.py --input exemplars/raw --output exemplars
# 3. 运行时启用：
python -m modeling_assistant.cli --problem "..." --exemplars-dir exemplars
```

摄入脚本支持 `--contest 国赛`、`--problem-type optimization` 等覆盖参数；
无 LLM API key 时自动降级为确定性卡片（章节启发式 + 关键词题型判定）。

### 反馈与防过拟合

- **HITL 终审反馈**：输入 `approve score 80`（0~100 分），系统以滑动平均
  （`feedback_alpha=0.3`）回写卡片与指南的质量权重，并持久化到知识库。
- **注入强度分级**：`style_injection` 数值作为各层注入概率
  （structure=1.0 / chart=0.8 / writing=0.5），writing 额外按
  `style_dropout_rate=0.3` 随机关闭，防止风格同质化。
- **查重护栏**：Writer 输出与示例摘录/原文做 8-gram 重合检测，超过
  `plagiarism_threshold=0.15` 时在 `prompt_audit` 写入警告。
- **留一验证**：量化每张卡片对题型指南的独有贡献：

```powershell
python scripts/leave_one_out_eval.py --exemplars exemplars
```

详细设计见 [docs/exemplar_learning_plan.md](docs/exemplar_learning_plan.md)。

项目已内置 2018~2025 年国赛优秀论文知识库（93 张卡片、5 份题型指南），
运行时默认从 `exemplars/` 加载，直接对建模流程生效。

知识库之上还有**行文技艺层**（`exemplars/craft/` 18 篇深加工 + `exemplars/craft_guides/`
5 类题型指南）：学习优秀论文正文中数学推导、算法分析、模型解释的安排方式，
功能句型与图片位置规划，运行时注入 Writer / Drawer / Architect。

## 配置（.env 示例）

```text
MODELING_ASSISTANT_LLM_MODEL=deepseek-chat
MODELING_ASSISTANT_API_KEY_ENV=DEEPSEEK_API_KEY
MODELING_ASSISTANT_REASONING_EFFORT=max
# 可按节点覆盖推理强度（未列出的节点继承上面的全局值）
MODELING_ASSISTANT_REASONING_EFFORT_OVERRIDES={"searcher":"low","analyst":"high","mathematician":"max","realist":"high","clarifier":"max","architect":"max","coder":"max","writer":"high","final_reviewer":"max"}
# Mathematician 的数据画像最多注入多少列（按当前小题相关性筛选）
MODELING_ASSISTANT_MATHEMATICIAN_MAX_COLUMNS=32
MODELING_ASSISTANT_SEARCH_ENABLED=false
MODELING_ASSISTANT_OUTPUT_DIR=outputs
MODELING_ASSISTANT_MAX_DEBATE_ROUNDS=3
# 正确性等作为硬门槛；通过后创新性用于拉开竞赛竞争力
MODELING_ASSISTANT_INNOVATION_WEIGHT=0.3
MODELING_ASSISTANT_FEASIBILITY_WEIGHT=0.7
MODELING_ASSISTANT_FEASIBILITY_THRESHOLD=60

# Exemplar Learning System
MODELING_ASSISTANT_EXEMPLARS_DIR=exemplars
MODELING_ASSISTANT_EXEMPLAR_MIN_RELEVANCE=0.25
MODELING_ASSISTANT_EXEMPLAR_TOP_K=2
MODELING_ASSISTANT_STYLE_INJECTION={"structure":1.0,"chart":0.8,"writing":0.5}
MODELING_ASSISTANT_STYLE_DROPOUT_RATE=0.3
MODELING_ASSISTANT_PLAGIARISM_NGRAM=8
MODELING_ASSISTANT_PLAGIARISM_THRESHOLD=0.15
MODELING_ASSISTANT_FEEDBACK_ALPHA=0.3

# 方法知识库：从内置规范按节点/题型切片注入 prompt
MODELING_ASSISTANT_METHOD_KNOWLEDGE_ENABLED=true

# 论文 LaTeX 模板目录：默认内置国赛 CUMCM 模板
MODELING_ASSISTANT_PAPER_TEMPLATE_DIR=templates/cumcm-latex

# LLM 输出预算（全局默认 + 分节点覆盖，JSON）
MODELING_ASSISTANT_LLM_MAX_TOKENS=32768
MODELING_ASSISTANT_LLM_MAX_TOKENS_OVERRIDES={"writer":32768,"coder":32768,"architect":32768,"mathematician":32768,"realist":32768,"clarifier":24576,"final_reviewer":16384,"drawer":12288,"reflection":8192,"arbiter":4096,"milestone_reviewer_1":4096,"meta_router":4096,"searcher":2048}

# 分节点采样温度：发散节点更高，评审/固化节点更低
MODELING_ASSISTANT_LLM_TEMPERATURE_OVERRIDES={"mathematician":1.0,"analyst":0.6,"data_analyst":0.3,"realist":0.5,"arbiter":0.2,"clarifier":0.2,"milestone_reviewer_1":0.3,"final_reviewer":0.2}
```

## 目录

```text
src/modeling_assistant/
  agents/       图节点函数（Analyst/Mathematician/Realist/Clarifier/Coder/Drawer/Writer/...）
  config/       配置读取（.env / 环境变量 / CLI 覆盖）
  data/         数据画像、题面常量提取、优秀论文摄取与聚合
  graph/        LangGraph 构建与条件路由
  memory/       LTM 快照/版本/回滚、示例库检索与反馈回写
  prompts/      Agent system prompt 模板
  recording/    运行过程记录（process_log / prompt 存档 / token usage 汇总与报告）
  schemas/      状态与领域模型（含 Exemplar 表达知识模型）
  validation/   结果校验、常量防线、论文↔结果数字比对、图表完整性、原创性查重
  cli.py        命令行入口
outputs/logs/   运行过程记录（process_report.md / process_log.jsonl / usage.jsonl）
exemplars/      优秀论文知识库（raw/ 忽略，cards/ guides/ profile.yaml 纳入版本控制）
scripts/        摄入、留一验证等工具脚本
docs/           设计文档
tests/          单元与集成测试
real_tests/     实例测试题目与端到端脚本（已 gitignore）
```

## 测试

```powershell
python -m pytest -q tests
```

注：`tests/test_graph.py` 的端到端用例需要可用的 LLM API（网络），
其余用例可在无网络环境下运行。
