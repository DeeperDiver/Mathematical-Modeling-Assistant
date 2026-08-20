# 正文行文技艺学习方案（Writing Craft Layer）

> 版本：v1.2　日期：2026-08-19　状态：**已实现（P0~P3 完成，P4 端到端对照待跑）**
> 目标：在现有 Exemplar 表达学习（结构/图表/文风）之上，新增「行文技艺层」，
> 专门学习正文行文的六大技艺：**数学推导、算法分析、模型解释、行文语言、
> 图片位置规划、正文侧重点与论证链条**——让 Writer 不仅知道"写哪些章节、
> 用什么图"，还知道"推导怎么铺陈、算法怎么讲、模型怎么解释、句子怎么写、
> 图放在哪、每节写什么写多少、论证怎么推进"。

---

## 一、核心设计原则

1. **只学"怎么讲"，不学"讲了什么"**：不保存具体公式、数值、算法参数、
   模型系数——只提炼组织顺序、展开节奏、表达动作与功能化句型。
2. **增量兼容**：深加工产物放入独立目录 `exemplars/craft/` 与
   `exemplars/craft_guides/`，不修改现有 cards/guides 的字段与检索逻辑，
   旧功能完全不受影响。
3. **防过拟合不变**：题型级共性仍要求 ≥3 篇共有；范例句 ≤60 字且按功能分类；
   注入仍走强度分级 + dropout + 查重护栏。
4. **可审计**：每条 pattern 记录来源卡片 id 与所在章节，方便追溯与人工修正。

---

## 二、学习对象（三类行文安排）

### 2.1 数学推导（Mathematical Derivation）

学习"公式是如何被引入、推导和收尾的"：

| 观察维度 | 学习内容 |
|---|---|
| 引入方式 | 推导前是否有动机铺垫（物理/业务/数学动机）；是"先定义后推导"还是"边推边定义" |
| 展开顺序 | 符号引入 → 前提假设 → 推导步骤 → 目标函数/结论 → 物理解释 的典型顺序 |
| 详略策略 | 关键公式逐步推导、常规变换直接给出；何时用"由…可得""同理" |
| 符号纪律 | 先定义后使用、下标/上标规则、公式编号与引用习惯 |
| 文公配合 | 文字与公式的比例与节奏（每步推导前后是否有文字解释） |
| 收尾动作 | 推导完成后如何解释每项含义、联系题目背景 |

### 2.2 算法分析（Algorithm Analysis）

学习"算法是如何被描述、论证和报告结果的"：

| 观察维度 | 学习内容 |
|---|---|
| 呈现方式 | 伪代码块 / 流程图 / 步骤列表 的选择与组合 |
| 流程组织 | 输入 → 初始化 → 迭代 → 终止条件 → 输出的叙述顺序 |
| 复杂度论证 | 时间/空间复杂度放在哪里、怎么表述（"共需…次迭代"） |
| 收敛/可行性论证 | 如何论证算法收敛或可行（迭代曲线、目标值变化、边界检查） |
| 结果对应 | 输出结果如何与算法步骤一一对应地报告 |
| 图表支撑 | 用什么图支撑算法论证（收敛曲线、目标函数变化、耗时对比） |

### 2.3 模型解释（Model Interpretation）

学习"模型建立后是如何被解释的"：

| 观察维度 | 学习内容 |
|---|---|
| 解释对象 | 模型整体、参数含义、边界条件、结果意义 |
| 解释顺序 | 先整体后局部、先含义后影响、先理论后实例 |
| 领域连接 | 如何把模型与题目背景/业务意义挂钩（"在本题中，该参数代表…"） |
| 参数讲解 | 参数的单位、取值范围、变化影响如何表述 |
| 边界与敏感性 | 边界条件、极端情形、灵敏度讨论的安排位置 |
| 常见动作 | "该假设的合理性在于…""当…时，模型退化为…"等解释句型 |

### 2.4 行文语言（Writing Language）

学习"句子本身是怎么写的"：

| 观察维度 | 学习内容 |
|---|---|
| 功能句型库 | 按功能分类的句型：摘要结果句、假设铺垫句、过渡衔接句、结果解读句、结论升华句、局限说明句 |
| 衔接与节奏 | 章节之间如何过渡；段内总-分/递进/转折结构；长短句搭配 |
| 详略与语气 | 术语密度、主被动语态选择、客观化表述（"由表可知""实验表明"） |

### 2.5 图片位置规划（Figure Placement）

学习"图为什么出现在那里"：

| 观察维度 | 学习内容 |
|---|---|
| 图-章节-作用映射 | 收敛曲线→模型求解→证明算法收敛；箱线图→灵敏度分析→对比稳定性；热力图→结果分析→展示空间分布 |
| 图表引出与解读 | "如图 X 所示…"的引出方式；图后是否紧跟一句解读 |
| 图表密度 | 每个章节通常配几张图、文字与图表如何衔接 |

### 2.6 正文侧重点与论证链条（Section Focus & Argument Flow）

学习"每节写什么、写多少，全文论证怎么推进"：

| 观察维度 | 学习内容 |
|---|---|
| 每节重点 | 模型建立=定义变量/假设/目标函数；结果分析=总指标→分组对比→图表支撑→原因解释 |
| 篇幅分配 | 各章节占全文比例（如模型建立约 30%、结果分析约 25%） |
| 内部展开顺序 | 每节内部固定套路（符号→假设→推导→目标函数→解释） |
| 论证链条 | 问题→假设→模型→求解→验证→评价 的推进与每步衔接方式 |

---

## 三、数据模型（新增 schema）

```python
class WritingExample(BaseModel):
    function: str        # 句型功能：如"推导动机句""假设铺垫句""结果解读句"
    text: str            # 范例句（≤60 字，去题目化）
    note: str            # 这句在行文中起什么作用

class DerivationPattern(BaseModel):
    section: str = ""                # 通常出现在哪个章节
    trigger: str = ""                # 什么情况触发推导（如目标函数非线性、约束耦合）
    organization: list[str] = []     # 展开顺序（符号→假设→推导→结论→解释）
    notation_usage: str = ""         # 符号使用纪律
    depth_strategy: str = ""         # 详略策略
    text_formula_ratio: str = ""     # 文公配合节奏
    closing_moves: list[str] = []    # 收尾动作
    examples: list[WritingExample] = []

class AlgorithmPattern(BaseModel):
    algorithm_type: str = ""         # 启发式/精确/贪心/DP/随机模拟…
    presentation: list[str] = []     # 伪代码/流程图/步骤列表
    flow_organization: list[str] = []
    complexity_analysis: str = ""
    convergence_justification: str = ""
    result_reporting: str = ""
    support_figures: list[str] = []  # 收敛曲线/目标值变化…
    examples: list[WritingExample] = []

class InterpretationPattern(BaseModel):
    target: str = ""                 # 模型/参数/边界/结果
    organization: list[str] = []
    domain_linking: str = ""
    parameter_meaning: str = ""
    sensitivity_handling: str = ""
    common_moves: list[str] = []
    examples: list[WritingExample] = []

class WritingCraft(BaseModel):       # 单篇深加工卡
    card_id: str
    derivation: list[DerivationPattern] = []
    algorithm: list[AlgorithmPattern] = []
    interpretation: list[InterpretationPattern] = []
    writing: list[WritingPattern] = []            # 行文语言（功能句型库）
    figure_placements: list[FigurePlacement] = [] # 图-章节-作用映射
    section_focuses: list[SectionFocus] = []      # 每节重点与篇幅
    argument_flow: ArgumentFlow | None = None     # 全文论证链条
    created_at: datetime = ...

class CraftGuide(BaseModel):         # 题型级行文技艺指南
    problem_type: str
    derivation_common: list[DerivationPattern] = []    # ≥3 篇共有
    algorithm_common: list[AlgorithmPattern] = []
    interpretation_common: list[InterpretationPattern] = []
    writing_common: list[WritingPattern] = []
    figure_placement_common: list[FigurePlacement] = []
    section_focus_common: list[SectionFocus] = []
    argument_flow_common: ArgumentFlow | None = None
    exemplar_ids: list[str] = []

class WritingPattern(BaseModel):
    function: str                    # 句型功能
    examples: list[WritingExample] = []
    usage_notes: str = ""            # 使用时机与注意事项

class FigurePlacement(BaseModel):
    figure_type: str
    section: str                     # 放在哪个章节
    argument_role: str               # 支撑什么论证
    caption_style: str = ""          # 图注写法

class SectionFocus(BaseModel):
    section: str
    focus: str                       # 写作重点
    weight: float = 0.0              # 篇幅占比（0~1）
    internal_order: list[str] = []   # 节内展开顺序

class ArgumentFlow(BaseModel):
    steps: list[str] = []            # 论证链条步骤
    transitions: list[str] = []      # 步骤间衔接方式
```

---

## 四、提炼管道

1. **输入**：93 篇论文的全文文本（复用 `exemplars/raw/cumcm/**/*.ocr.txt` 缓存）。
2. **提炼**：新增 prompt 模板 `prompts/templates/exemplar_craft_ingest.md`，
   由 LLM 对每篇论文输出完整的 `WritingCraft` JSON（六大技艺）；**优先用 deepseek-v4-pro**
   （行文理解任务质量优先），失败重试 2 次，仍失败则跳过并记录（不降级为
   确定性占位，避免低质污染）。
3. **存储**：`exemplars/craft/{card_id}.json`（独立目录，增量兼容）。
4. **题型聚合**：`exemplars/craft_guides/{problem_type}.json`，
   同一题型 ≥3 篇共有才进共性；聚合规则与 L2 指南一致（含章节/动作归一化）。
5. **可审计**：每个 pattern 保留 `section` 与来源卡 id；`examples` 必须
   去题目化（不出现具体数值、专有模型名外的具体场景）。

## 五、运行时注入

- `PromptCatalog` 新增变量：`craft_derivation_json`、`craft_algorithm_json`、
  `craft_interpretation_json`（从 `exemplars/craft_guides/` 命中题型时注入，
  否则取该题型 top 卡片的 craft 补足）。
- **Writer 模板**新增「行文技艺参考」块，分三小节：
-  - 数学推导安排：引入方式、展开顺序、详略策略、收尾动作 + 句型范例
  - 算法分析安排：呈现方式、流程组织、复杂度/收敛论证 + 句型范例
  - 模型解释安排：解释顺序、领域连接、参数讲解 + 句型范例
  - 行文语言：功能句型库（按"摘要/假设/过渡/解读/升华/局限"分类的范例句）
  - 正文侧重点：每节重点、篇幅占比、内部展开顺序
  - 论证链条：全文推进步骤与衔接方式
- **Drawer 模板**增强：注入 `figure_placements`（图-章节-论证作用映射），
  让图表位置与论证环节绑定，而不是只给图表类型清单。
- **Architect 模板**增强：注入 `section_focuses` 与 `argument_flow`，
  让论文大纲自带每节重点、篇幅分配与论证推进顺序。
- 注入强度：craft 层并入 writing 注入组（强度 0.6），受 dropout 约束；
  防抄袭约束块与 8-gram 查重护栏不变。

## 六、质量评估

1. **深加工覆盖度**：统计 93 篇中成功生成 `WritingCraft` 的数量与失败原因。
2. **模式独有性**：扩展留一验证，对 craft pattern 做覆盖度分析
   （某卡片独有的推导/解释动作在剔除后是否仍被同题型覆盖）。
3. **写作技艺落地检查**：对生成的 LaTeX 做启发式信号检测——
   - 数学推导：是否出现"由…可得/因此/化简得"等推导衔接词 + 公式数量
   - 算法分析：是否含伪代码/流程图/复杂度表述
   - 模型解释：是否出现"该参数表示/当…时/合理性在于"等解释句型
   - 行文语言：功能句型覆盖（摘要/假设/过渡/解读句是否齐备）
   - 图片位置：生成的图是否按规划章节出现（图注/章节关键词匹配）
   - 论证链条：问题/模型/求解/验证/评价各环节是否齐备
   输出"技艺覆盖评分"，用于开/关深加工库的对照。
4. **人工评审**：先深加工 3~5 篇代表论文（覆盖 optimization/physics/
   evaluation 三类），由用户确认六大技艺的 pattern 质量后再全量。

## 七、实施阶段

| 阶段 | 内容 | 预计 |
|---|---|---|
| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | schema + `exemplar_craft_ingest.md` 模板 + 提炼脚本 | ✅ |
| P1 | 每题型精选 4 篇高质量论文深加工（共 18 篇） | ✅（optimization/physics/evaluation/data_mining 各 4 篇，forecasting 2 篇） |
| P2 | 题型级行文技艺聚合（craft_guides/，5 类） | ✅ |
| P3 | PromptCatalog 注入 + Writer/Drawer/Architect 模板改造 + 单测 | ✅（187 项测试通过） |
| P4 | 端到端对照（开/关深加工库写同一道题）与参数调优 | ⏳ |

## 九、2026-08-19 整合记录

- **深加工 18 篇**（v4-pro）：每篇六大技艺齐全——数学推导（触发/顺序/详略/文公配合/
  收尾动作）、算法分析（呈现/流程/复杂度/收敛论证/支撑图）、模型解释（对象/顺序/
  领域连接/参数含义）、6 类功能句型库、图片位置规划（图-章节-论证作用）、章节重点
  与篇幅占比、全文论证链条。
- **题型聚合 5 类**：`exemplars/craft_guides/*.json`，推导按章节聚类、算法按大类
  （启发式/动态规划/精确/随机模拟/机器学习）归一化，共性与范例去重，防过拟合不变。
- **运行时注入**：Writer 获推导/算法/解释/句型，Drawer 获图片位置规划，
  Architect 获章节重点与论证链条；craft 层与 writing 注入开关同步（dropout 约束）。
- **测试**：新增行文技艺单测（聚合、归一化、模板注入），全量 187 项通过。

## 八、预期效果与边界

- **预期**：Writer 不再"凭感觉安排推导/算法/解释"，而是有明确的展开顺序、
  详略策略与功能化句型可循；Drawer 的图表位置与论证环节绑定。
- **边界**：深加工仍是"参考增强"，不保证 100% 达到样例质量；最终由 HITL
  终审把关。本方案只学习"行文技艺"，不学习任何公式/数值/算法参数内容。
