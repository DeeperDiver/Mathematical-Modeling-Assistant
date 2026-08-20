
# 数学建模 Multi-Agent 协作框架 (基于 LangGraph)
*—— 基于 SSOT (唯一可信源) 与 Git式版本控制的增强版*

## 一、 核心架构思想：LTM (长期记忆) 驱动与版本快照库

摒弃传统的“对话流转”模式，采用**“全局结构化长期记忆（LTM）”，确定的设定将被结构化提取并写入 LTM，后续所有 Agent 均只基于 LTM 进行工作。，并引入**版本控制（Version Control）**。防止模型在辩论中退化，确保任何优秀的“灵光一现”都能被封存、追溯和回滚。

### Graph State 状态设计 (TypedDict / Pydantic)
State 被划分为四个核心层级：
1. **Core State One (静态 LTM)**：问题理解，初始化后不可篡改。包含：原始问题、数据字典(Data Schema)、核心参考文献(Literature)。
2. **Core State Two (动态 LTM)**：解题方案，由 Clarifier 总结写入，是下游节点执行的唯一凭证。包含：问题假设(Assumptions)、符号表(Nomenclature)、核心公式/目标函数(Equations)。
3. **LTM Archive (记忆封存库 / 历史快照)**：**【⭐核心新增】**一个以列表追加（Append）形式存在的版本库（如 `v1.0`, `v1.1`, `v2.0`）。保存被 `Clarifier` 覆盖前的旧版本LTM。
4. **Control State (控制流与短期上下文)**：仅用于当前阶段计算。包含：候选模型(Top-K Plans)、辩论轮数(debate_round)、创新/可行性评分、代码/图表产物。
    

---


## 二、 详细工作流程与节点定义

整个网络划分为五个带有**“检视与回滚机制”**的核心阶段：

### 阶段一：输入与全局信息初始化
1. **`Problem` (入口)**：接收赛题与数据附件。
2. **`Analysist` (破题者)**：提取赛题核心矛盾，生成初始的 `静态 LTM`。
3. **`Searcher` (检索者)**：根据破题思路，检索高价值 ArXiv 论文和参考模型，将其摘要存入 静态 LTM 的参考文献库中，供后续调用。

### 阶段二：建模核心阶段 —— “先发散，后剪枝”的辩证博弈
1. **`Mathematician` (发散与创新)**：
   - 尽可能头脑风暴，拓展解题思路，体现创新，使模型脱颖而出。
   - **查询权限**：当陷入瓶颈时，有权查询 `LTM Archive (封存库)`，从早期被废弃的灵感中寻找破局点，进行“分支重建（Branching）”。
1. **`Realist` (挑刺与剪枝)**：
   - 从数据、算力、常识三维度对方案进行生存分析。
   - 依据 $Score_{total} = w_1 \cdot Score_{inn} + w_2 \cdot Score_{fea}$ 双轨打分，触发打回修改。砍掉不切实际的方案（Feasibility < 60）；打回平庸方案（Innovation < 60）。促使 Mathematician 修改并提出新方案。
1. **`Arbiter` (仲裁者 - 防死循环与退化拦截)**：
   - 当 `debate_round > 3` 时介入。
   - **历史比对**：Arbiter 会对比当前的“妥协版方案”与封存库中的“初代激进方案”。如果发现“越辩越烂”，Arbiter 有权直接终止辩论，提取某一个历史快照强制设为当前最优解。作为一个冷静的第三方专家，敲定一个折中方案，并触发 Human-in-the-loop 让用户决断。

### 阶段三：方案具体化与 LTM 快照管理
1. **`Clarifier` (总结并得出总体解题思路与知识注入**)**：
   - 提取胜出方案的要素。
   - **快照动作**：在将新方案写入 `Core State Two (当前动态LTM)` **之前**，自动将现有的动态 LTM 整体打包，`Push` 到 `LTM Archive` 中封存，并生成版本号（如 v2.0）。
   - 进行符号查重与公式闭环校验。
2. **`Milestone Reviewer 1` & `HITL 1` (人类架构决断)**：
   - 系统暂停。向人类展示当前 LTM。
   - **人类回滚权**：如果人类觉得当前模型虽然严谨但太过平庸，可以在前端界面直接指令系统：“放弃当前版本，`Checkout` (回滚) 到早期模型，并从该节点重新开始细化”。人类确认无误后，放行进入执行阶段。
1. **`Architect` (架构师**：任务是将解题思路具体化，设计可视化，建立论文每一部分的结构)：
   - 严格基于 LTM，建立论文每一部分的结构骨架（大纲），并为下游的图表和代码制定伪代码/API级别的数据输入输出规范。

### 阶段四：并行执行与自纠错
Architect 的指令分为两路，通过 LangGraph 的并行执行机制同时派发：
1. **`Drawer` (可视化工程师)**：- 根据 Architect 的规范，生成图表代码或调用绘图工具，产出可视化结果路径。
2. **`Coder` (算法工程师)**：
   - 完全屏蔽冗长Context，完全根据 动态 LTM 中的公式和符号表编写 Python 代码。
   - **【回滚触发机制】**：如果代码运行连续报错 3 次（例如：发现非线性规划根本无法在多项式时间内收敛）。通过 Conditional Edge 退回到 Architect 或 Clarifier，并携带错误日志。

### 阶段五：最终整合与成稿
1. **`Writer` (LaTeX 主笔)**：
- 汇集 静态 LTM（文献）、动态 LTM（假设与公式）、Drawer 的图表和 Coder 的结果数据。
- 负责将这一切转化为格式严谨、连贯的最终 LaTeX 源码。
1. **`Milestone Reviewer 2` & `HITL 2` (终稿审查)**：
   - **【触发人工介入】** 评审最终 PDF 效果。如果模型结果差强人意，人类可决断是否通过 Conditional Edge 将图谱状态重置回 Mathematician（回到建模阶段继续打磨）；如果完美，则完成整个闭环。

---

## 三、 LangGraph 工程实现的 3 个核心要素

1. **State Reducers 的精妙配合 (覆盖 vs 追加)**：
   - **Current LTM**：使用 `overwrite` (覆盖) 机制。保证 `Coder/Drawer` 永远只能看到唯一、纯净的当前设定，不会产生幻觉和设定冲突。
   - **LTM Archive**：使用 `append` (追加) 机制。不断累积历史快照列表，供 `Arbiter`、`Mathematician` 或人类进行 `read_only`（只读）查询。
2. **动态 System Prompt 注入 (无废话模式)**：
   - - 在 Coder、Drawer 和 Writer 的节点函数中，禁止传入完整的 history messages。它们的 System Prompt 必须动态拼接 State["LTM"]，指令明确：“严格服从以下变量约定，绝不能自行编造设定。”
1. **强大的 Conditional Edges (带时光机的状态路由)**：
    - Mathematician ↔ Realist (条件：分数达标 或 Arbiter介入)
        
    - Coder ➔ Architect (条件：代码执行失败超过 3 次)
        
    - Reviewer ➔ HITL (条件：关键节点拦截)
   - 增加一条特殊的边：`rollback_edge`。当 `Arbiter` 或 `HITL` 决定回滚时，路由器将状态中的 `current_knowledge_base` 替换为 `archived_knowledge_bases[-1]`，并将流程重置回 `Architect` 节点。

---

## 四、 优秀论文表达学习层（Exemplar Learning System）

在 LTM 体系之上增加一层「表达知识」，解决「怎么把论文写好」的问题：
输入过往优秀论文，提炼为结构化表达知识，运行时检索并分级注入下游节点，
让整体表达（论文结构、可视化图表、行文风格）随用户积累的优秀样本持续进化。

### 4.1 三层知识结构

1. **L1 单篇卡片（ExemplarPaper）**：每篇论文提炼为结构化卡片——章节骨架、
   图表清单与风格、文风特征、亮点与雷区、短摘录（≤80 字）。只记录「怎么说」，
   不保存公式、数值与具体结果。
2. **L2 题型指南（TypeStyleGuide）**：按 (题型, 赛事) 分组聚合，只有
   **≥3 篇共有**的特征才进入共性字段，个性亮点留在卡片层作为弱信号。
3. **L3 全局偏好（GlobalStyleProfile）**：用户个人审美（配色、图表偏好、行文习惯），
   独立于优秀论文，始终作为最上层软约束注入。

### 4.2 运行时注入与防过拟合

- 新增 `exemplar_loader` 节点：位于 `Searcher` 与 `Mathematician` 之间，
  先做题型判定（规则关键词 + LLM 兜底），再按 TF-IDF / 字符 n-gram 相关性
  检索 Top-K 卡片与题型指南；低于相关性阈值时不注入，流程与无知识库时完全一致。
- **分级注入**：结构（强）→ Architect；图表（中）→ Drawer；
  文风（中/弱）→ Writer；结构完整性审查（弱）→ Milestone Reviewer 1。
- **强度与 Dropout**：`style_injection` 数值作为各层注入概率，
  writing 层额外按 `style_dropout_rate` 随机关闭，防止风格同质化。
- **防抄袭护栏**：所有注入模板明确禁止复制示例的句子/公式/数值/图表数据；
  Writer 输出与示例库做 n-gram 重合检测，超阈值写入警告。
- **反馈回写**：HITL 终审支持 `approve score <0-100>`，以滑动平均更新
  卡片与指南的质量权重并持久化，形成「越用越懂」的闭环。
- **留一验证**：剔除单篇卡片后检查其亮点是否仍被同题型其余卡片覆盖，
  量化每篇样本的独有贡献，作为知识库增删与参数调优的依据。

详细设计文档见 `docs/exemplar_learning_plan.md`。
