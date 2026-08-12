# 优秀论文表达学习系统（Exemplar Learning System）实施计划

> 版本：v1.1　日期：2026-08-12　状态：**已实现（P0~P3 完成，P4 待真实 API 环境执行对照）**
> 目标：输入「题目 + 优秀示例论文」对，离线提炼为结构化表达知识，运行时按需注入
> Architect / Drawer / Writer / Reviewer 的 prompt，在不污染建模正确性的前提下提升
> 论文结构、图表与文风的整体质量，并内置防过拟合与评估机制。

---

## 一、总体架构

```text
┌───────────── 离线学习（一次/增量） ─────────────┐
│  (题目, 优秀论文) 对                             │
│      │                                          │
│      ▼                                          │
│  摄取器：PDF/LaTeX/MD 解析                       │
│      │                                          │
│      ▼                                          │
│  LLM 提炼 → L1 单篇卡片（exemplars/cards/*.json）│
│      │                                          │
│      ▼                                          │
│  按 (题型, 赛事) 聚合 → L2 题型指南（guides/）   │
│  + L3 全局风格偏好（profile.yaml）               │
└─────────────────────────────────────────────────┘
                    │ 运行时
                    ▼
┌───────────── 在线注入（每次运行） ──────────────┐
│ 新题 → 题型判定 → TF-IDF 检索（相关性阈值）      │
│      → ExemplarContext 注入 GraphState          │
│      → 分级注入：                               │
│        Architect: 结构骨架（强）                 │
│        Drawer:    图表推荐+风格（中）             │
│        Writer:    文风+摘要套路（中/弱，可 Dropout）│
│        Reviewer:  表达完整性审查（弱）            │
│      → HITL 反馈回写卡片（滑动平均，慢更新）      │
└─────────────────────────────────────────────────┘
```

核心原则：

1. **表达与内容严格分离**。Exemplar 只影响「怎么说」，不影响「算什么」；方法、公式、
   数值仍只走 LTM + Coder 验证链。
2. **检索增强而非全量注入**。每轮只注入与当前题型最相关的 1~3 张卡片 + 1 份题型指南。
3. **卡片是显式、可编辑的中间产物**。LLM 提炼后用户可直接修改 JSON，最终影响力在人。

---

## 二、目录与文件规划

```text
exemplars/
  raw/                  # 用户放置 PDF/tex/md 原文（建议 gitignore）
  cards/                # L1 单篇卡片（建议纳入版本控制）
    optimization_2025_01.json
  guides/               # L2 题型指南（纳入版本控制）
    optimization.json
  profile.yaml          # L3 全局风格偏好（纳入版本控制）

docs/exemplar_learning_plan.md   # 本文档

src/modeling_assistant/
  data/
    exemplars.py        # 卡片/指南的加载、保存、目录扫描
    exemplar_ingest.py  # PDF/tex 解析 + LLM 卡片提炼 + 题型聚合
  memory/
    exemplar_search.py  # 题型判定、TF-IDF 检索、相关性阈值
    exemplar_feedback.py# HITL 反馈回写（滑动平均）
  validation/
    originality.py      # n-gram 查重护栏
  agents/
    nodes.py            # 新增 exemplar_loader_node
  prompts/
    templates/
      exemplar_ingest.md    # 卡片提炼 prompt
      architect.md          # + 结构参考块
      drawer.md             # + 图表参考块
      writer.md             # + 文风参考块 + 防抄袭约束
      milestone_reviewer_1.md # + 表达审查清单（可选）

scripts/
  ingest_exemplars.py    # 批量摄入：python scripts/ingest_exemplars.py --input ...
  leave_one_out_eval.py  # 留一验证：量化每篇示例的贡献
```

---

## 三、数据模型（schemas/state.py 新增）

```python
class ExemplarFigure(BaseModel):
    figure_type: str        # boxplot / scatter / heatmap / pareto / convergence / gantt ...
    purpose: str            # 这张图回答什么问题
    style_notes: str = ""   # 配色、标注、字号、坐标轴习惯
    example_path: str = ""  # 示例图文件路径（可选）

class ExemplarPaper(BaseModel):      # L1 单篇卡片
    id: str
    title: str
    source_path: str = ""            # 原文路径
    problem_type: str = ""           # 题型标签
    contest: str = ""                # 赛事语境：国赛/美赛/华中杯/...
    year: int | None = None
    structure: dict[str, str] = {}   # 章节骨架 {章节名: 目的/写法}
    section_notes: list[str] = []    # 每节写法要点
    figures: list[ExemplarFigure] = []
    writing_style: dict[str, str] = {}   # 文风特征
    summary_style: str = ""          # 摘要写法套路
    highlights: list[str] = []       # 个性亮点（只进 L1，不进 L2）
    pitfalls: list[str] = []         # 雷区
    quotes: list[str] = []           # 短摘录，单条 ≤ 80 字（受查重护栏约束）
    quality_score: float = 0.5       # 用户评分/权重（反馈回写）
    tags: list[str] = []
    created_at: datetime = ...

class TypeStyleGuide(BaseModel):     # L2 题型指南
    problem_type: str
    contest: str = ""
    common_structure: list[str] = []     # 共性骨架（多篇共有才进）
    structure_variants: list[str] = []   # 可选变体
    recommended_figures: list[str] = []
    writing_baseline: dict[str, str] = {}
    common_pitfalls: list[str] = []
    exemplar_ids: list[str] = []
    version: str = "1.0"
    updated_at: datetime = ...

class GlobalStyleProfile(BaseModel): # L3 全局偏好
    color_palette: list[str] = []
    figure_preferences: list[str] = []
    writing_preferences: dict[str, str] = {}
    notes: str = ""

class ExemplarContext(BaseModel):    # 运行时注入包
    active: bool = False
    guide: TypeStyleGuide | None = None
    cards: list[ExemplarPaper] = []
    injection: dict[str, bool] = {"structure": True, "chart": True, "writing": True}
```

GraphState 新增字段：

```python
exemplars: Annotated[ExemplarContext, overwrite_reducer]  # 默认 inactive
```

---

## 四、模块设计

### 4.1 摄取与提炼（data/exemplar_ingest.py）

输入：目录或单文件（.pdf / .tex / .md / .txt / 现成卡片 JSON）。

流程：

1. **文本提取**：PDF 用 pdfplumber（环境已装）；tex/md 直接读源码；JSON 视为已有卡片直接入库。
2. **结构解析**：LaTeX/MD 用正则提取章节树、`\begin{figure}`、`\includegraphics`、
   `\begin{table}`、公式数量；PDF 先抽文本，章节/图表清单由 LLM 辅助识别。
3. **LLM 提炼**：渲染 `exemplar_ingest.md`，要求输出 `ExemplarPaper` JSON；
   卡片必须区分「共性」与「个性」；quotes 单条 ≤ 80 字。
4. **题型聚合**：按 (problem_type, contest) 分组，对组内卡片做二次 LLM 提炼生成
   `TypeStyleGuide`；规则：**至少 3 篇共有才进 common_structure / writing_baseline**，
   否则只留在卡片 highlights。
5. **人工可编辑**：所有产出为 JSON/YAML 明文，用户可直接修改后重新运行聚合。

### 4.2 检索（memory/exemplar_search.py）

```python
def judge_problem_type(state) -> tuple[str, float]   # 题型判定 + 置信度
def search_exemplars(state, profile) -> ExemplarContext
```

- 题型判定：规则关键词优先（优化/调度/规划 → optimization；物理/机理 → physics；
  预测/时序 → forecasting；评价/决策 → evaluation；分类/回归/数据 → data_mining），
  规则命中不足时用 LLM 分类并给出置信度。
- 检索：对卡片文本做 TF-IDF（sklearn，已装），或按题型标签直接匹配；
  返回相关性分 Top-K（K 默认 2）。
- **相关性阈值**：`min_relevance_threshold`（默认 0.25）。低于阈值 → `active=False`，
  完全不注入，流程退回现有行为。
- 赛事语境优先：同 contest 的卡片权重 ×1.2，不同 ×0.8。

### 4.3 Prompt 注入（prompts/catalog.py + templates）

`PromptContext.to_template_vars()` 新增变量：

| 变量 | 内容 | 注入目标 |
|---|---|---|
| `exemplar_active` | true/false | 所有相关模板 |
| `exemplar_structure_json` | 共性骨架 + 变体 | architect.md |
| `exemplar_chart_json` | 图表类型/目的/风格 | drawer.md |
| `exemplar_writing_json` | 文风基线 + 摘要套路 | writer.md |
| `exemplar_highlights_json` | 单篇亮点（可选灵感） | writer.md / architect.md |
| `style_profile_json` | L3 全局偏好 | drawer.md / writer.md |
| `exemplar_quotes_json` | 短摘录（受查重约束） | writer.md |

注入强度分级（`AppSettings.style_injection`）：

```python
style_injection = {"structure": 1.0, "chart": 0.8, "writing": 0.5}
style_dropout_rate = 0.3   # writing 卡片的随机丢弃率（防依赖）
```

每个模板统一追加防抄袭约束块：

> 以上为表达风格参考。允许借鉴结构与写法，**禁止复制示例中的具体句子、
> 公式、数值、图表数据与图表文件**；所有内容仍必须以当前 LTM 与真实执行结果为唯一依据。

### 4.4 图构建与 CLI

- 新节点 `exemplar_loader_node`：接在 `searcher` 之后、`mathematician` 之前，
  调用检索器，把 `ExemplarContext` 写入 state。无示例库或未命中时仅设置
  `active=False`，不改变任何现有路由。
- `cli.py` 新增 `--exemplars-dir`（默认 `exemplars/`）；离线摄入命令：
  `python scripts/ingest_exemplars.py --input <目录或文件> --output exemplars/`。

### 4.5 防过拟合与反馈

| 机制 | 实现 | 位置 |
|---|---|---|
| 相关性阈值 | 低于阈值不注入 | exemplar_search.py |
| 注入强度分级 + Dropout | 按配置概率注入 writing 卡 | catalog.py |
| 查重护栏 | writer 输出与 quotes/原文做 8-gram 重合检测，超阈值写入 integrity warning | validation/originality.py |
| 反馈回写 | HITL 终审 feedback 以滑动平均（α=0.3）更新 quality_score 与指南权重 | memory/exemplar_feedback.py |
| 留一验证 | 剔除第 i 篇 → 用其余篇重写同题 → 比对结构/图表/文风清单 | scripts/leave_one_out_eval.py |

### 4.6 配置项（AppSettings + .env）

```text
MODELING_ASSISTANT_EXEMPLARS_DIR=exemplars
MODELING_ASSISTANT_EXEMPLAR_MIN_RELEVANCE=0.25
MODELING_ASSISTANT_EXEMPLAR_TOP_K=2
MODELING_ASSISTANT_STYLE_INJECTION={"structure":1.0,"chart":0.8,"writing":0.5}
MODELING_ASSISTANT_STYLE_DROPOUT_RATE=0.3
MODELING_ASSISTANT_PLAGIARISM_NGRAM=8
MODELING_ASSISTANT_PLAGIARISM_THRESHOLD=0.15
MODELING_ASSISTANT_FEEDBACK_ALPHA=0.3
```

---

## 五、分阶段实施计划

> 状态标记：✅ 已完成　⏳ 待真实 API 环境执行

### P0：骨架与数据模型（约 0.5 天）

| 任务 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|
| 建 exemplars/ 目录结构与示例卡片 | raw/ cards/ guides/ profile.yaml + 2 张手工卡片 | 目录结构符合本文档 | ✅ |
| 数据模型 + GraphState 字段 + 配置项 | schemas/state.py、config/settings.py 改动 | 模型可序列化，空 ExemplarContext 渲染不崩溃 | ✅ |
| 单元测试 | tests/test_exemplars.py | pytest 通过：模型序列化、空上下文渲染、手工卡片加载 | ✅ |

### P1：摄取与提炼（约 1~2 天）

| 任务 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|
| PDF/tex 解析器 | data/exemplar_ingest.py | 能抽取章节树、图表清单、公式统计 | ✅ |
| LLM 卡片提炼 | prompts/templates/exemplar_ingest.md | 5 组 (题目, 论文) 产出 5 张合规卡片 | ✅（LLM 路径已实现，真实批量摄入待用户提供论文） |
| 题型聚合 | 聚合函数 + 二次提炼 prompt | 产出 ≥2 份题型指南，共性条目有 ≥3 篇来源 | ✅（确定性聚合已通过冒烟） |
| 依赖声明 | pyproject.toml 增加 pdfplumber | `pip install -e .` 可安装 | ✅（声明完成；next_ai 环境安装见假设） |

### P2：检索与注入（约 1 天）

| 任务 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|
| 题型判定 + TF-IDF 检索 + 阈值 | memory/exemplar_search.py | 单测覆盖命中/未命中/低阈值三种情况 | ✅ |
| catalog 模板变量 + 4 个模板改造 | prompts 改动 | 开/关知识库渲染结果正确 | ✅ |
| exemplar_loader_node + 图接线 | nodes.py、builder.py 改动 | 无示例库时流程行为与现状完全一致 | ✅ |
| CLI 参数 | cli.py `--exemplars-dir` | 端到端跑通，prompt_audit 出现 exemplar_* 键 | ✅（代码路径就绪，真实端到端待 API 环境） |

### P3：防过拟合与反馈（约 1~2 天）

| 任务 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|
| 注入强度/Dropout | exemplar_loader_node + 配置 | 单测验证概率注入 | ✅（配置与节点逻辑就绪） |
| 查重护栏 | validation/originality.py | 8-gram 重合检测单测通过，超阈值进 writer 警告 | ✅ |
| 反馈回写 | memory/exemplar_feedback.py | 单测验证滑动平均更新 + HITL 终审接入 | ✅ |
| 留一验证脚本 | scripts/leave_one_out_eval.py | 能输出每篇示例的贡献评分报告 | ✅ |

### P4：评测与打磨（约 1 天）

| 任务 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|
| 真实题对照实验 | 开/关知识库的端到端对比报告 | 用 real_tests 题目跑通，报告结构/图表/文风差异 | ⏳（当前沙箱网络不可达 DeepSeek API，需真实 API 环境执行） |
| 参数调优 | 阈值、强度、dropout 调优记录 | 依据留一评分调整默认值 | ⏳ |
| 文档更新 | README + 本计划标记完成状态 | 运行方式、目录说明完整 | ✅ |

---

## 六、关键决策点（需确认或采用默认值）

1. **exemplars/raw 是否 gitignore**：建议 `exemplars/raw/` 忽略（用户论文属私有资产），
   `cards/`、`guides/`、`profile.yaml` 纳入版本控制（提炼产物可分享、可追溯）。
2. **PDF 依赖**：建议把 pdfplumber 写入 pyproject 依赖（当前 Python 3.13 环境已安装）。
3. **题型分类体系**：先固定 5 类（optimization / physics / forecasting / evaluation /
   data_mining），后续按实际论文扩展；类目定义写入 guides/README。
4. **注入强度默认值**：structure=1.0、chart=0.8、writing=0.5、dropout=0.3，
   留一验证后按数据调整。
5. **Reviewer 表达审查**：P2 先只做结构完整性检查（注入指南的 common_structure），
   全文风格审查放到 P4 视效果决定。

---

## 七、风险与对策

| 风险 | 对策 |
|---|---|
| PDF 提取质量差（扫描件/公式乱码） | 优先支持 tex/md；PDF 提取失败时提示用户提供源码或手工卡片 |
| LLM 提炼幻觉（结构/图表与原文不符） | 卡片保留 source_path；聚合时交叉验证；人工可编辑 |
| 风格同质化 | writing 卡 Dropout + Top-K 轮换 + 注入强度弱化 |
| 内容泄漏/抄袭 | quotes 限 80 字、prompt 硬约束、输出 8-gram 查重 |
| 反馈放大偏见 | 滑动平均慢更新 + 留一验证持续监督 |
| 冷启动/新题型 | 阈值兜底退回无知识库模式；L1 亮点作为弱信号 |
