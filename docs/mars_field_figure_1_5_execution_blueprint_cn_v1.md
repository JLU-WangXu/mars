# MARS-FIELD Figure 1-5 执行蓝图（中文）v1

## 这份文档解决什么问题

这份文档专门回答三件事：

1. 每张图到底要讲什么结论
2. 每个子图具体应该用哪些真实数据文件
3. 每个子图到底应该用 `PSE/结构图` 还是 `统计图`

目标不是“让图看起来复杂”，而是：

- 图像形式服务科学结论
- 不该用结构图的地方坚决不用结构图
- 该用结构图的地方就用高质量 PSE / PyMOL 资产把视觉做满

---

## Figure 1

### 图的总任务

告诉读者：

`MARS-FIELD` 的主对象不是候选集合，而是共享 residue field。

### 整图的一句话结论

多源证据不是平行投票，而是共同参数化共享 residue field，再由 controller 和 decoder 在这个 field 上做决策。

### 子图设计

#### Figure 1A

**要说什么**

- 五类证据流进入系统
- 它们是 encoder input，不是独立 scoring branch

**建议内容**

- 几何结构
- 进化 profile
- ASR / lineage
- retrieval / motif memory
- 环境 / 工程条件

**实际数据来源**

- 方法规格文档：
  - `F:\4-15Marsprotein\mars_stack\docs\mars_field_figure1_spec_v2.md`
- 代码模块：
  - `F:\4-15Marsprotein\mars_stack\marsstack\field_network\encoders.py`
  - `F:\4-15Marsprotein\mars_stack\marsstack\evolution.py`
  - `F:\4-15Marsprotein\mars_stack\marsstack\ancestral_field.py`
  - `F:\4-15Marsprotein\mars_stack\marsstack\retrieval_memory.py`

**图型建议**

- 必须是 `示意图 / schematic`
- 不适合用 PSE

**原因**

- 这里讲的是“算法输入模态”，不是单个蛋白的空间结构
- 用结构图会分散注意力

#### Figure 1B

**要说什么**

- 中心对象是 residue field
- 包括 site-wise `U(i,a)` 和 pairwise `C(i,j,a,b)`

**建议内容**

- 中央主视觉
- residue-position × amino-acid 的 field 板块
- 叠加 pairwise coupling arcs / local coupling slab
- 旁边写公式：
  - `E(x) = sum_i U(i, x_i) + sum_(i,j) C(i, j, x_i, x_j)`

**实际数据来源**

- 真实 field 对象的代表文件：
  - `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\position_fields.json`
  - `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\pairwise_energy_tensor.json`
  - `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_position_fields.json`
  - `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_pairwise_energy_tensor.json`

**图型建议**

- 必须是 `示意图 + 数学对象图`
- 不能用 PSE 结构图替代

**原因**

- 这里讲的是抽象决策空间，不是结构外观

#### Figure 1C

**要说什么**

- field 向右流向两个东西：
  - neural controller
  - structured decoder

**建议内容**

- 左侧是 shared field
- 右侧上方是 calibrated selector
- 右侧下方是 structured decoder
- 最右边是 ranked designs / case-study outputs / benchmark outputs

**实际数据来源**

- 代码模块：
  - `F:\4-15Marsprotein\mars_stack\marsstack\field_network\neural_model.py`
  - `F:\4-15Marsprotein\mars_stack\marsstack\field_network\neural_generator.py`
  - `F:\4-15Marsprotein\mars_stack\marsstack\decoder.py`
  - `F:\4-15Marsprotein\mars_stack\scripts\run_mars_pipeline.py`

**图型建议**

- 必须是 `示意图`
- 不适合 PSE

#### Figure 1D

**要说什么**

- 这个 field 在真实 target 上是可实例化的，不是概念空壳

**建议内容**

- 拿 1LBT 或 TEM1 做小 inset
- 展示：
  - 一个位点的 top residue field
  - 一条 pairwise edge
  - 一个 decoder 候选如何从 field 里走出来

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\position_fields.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_position_fields.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_decoder_preview.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_field_runtime_summary.json`

**图型建议**

- `小型统计/示意 inset`
- 不是 PSE 主图

---

## Figure 2

### 图的总任务

告诉读者：

这个方法在 broad benchmark 上站住了，而且 neural decoder 真实在工作。

### 整图的一句话结论

最终 controller 在 twelvepack 上总体稳定，且 neural decoder 在主路径里贡献了真实的 proposal generation。

### 子图设计

#### Figure 2A

**要说什么**

- final 相对 incumbent 的 paired policy shift

**建议内容**

- 每个 target 一行
- 横轴：
  - `policy_selection_score_delta_final_minus_current`
- 正值绿色，负值红色

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`

重点字段：

- `target`
- `policy_selection_score_delta_final_minus_current`

**图型建议**

- 必须是 `统计图`
- 最合适的是：
  - horizontal lollipop
  - dot-and-stick paired delta plot

**绝对不要**

- 用 PSE
- 用结构图

**原因**

- 这是 panel-level benchmark claim，必须一眼看出 9/12 positive、3/12 negative

#### Figure 2B

**要说什么**

- 最浓缩的 headline benchmark metrics

**建议内容**

- `9/12 improved`
- `3/12 decreased`
- `mean delta ~ -0.001`
- `34 retained neural-decoder candidates`

**实际数据来源**

- 由下面两份汇总得到：
  - `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`
  - `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

**图型建议**

- 必须是 `数值卡片 / metric cards`
- 不适合 PSE

#### Figure 2C

**要说什么**

- neural decoder 在各 target 上到底做了多少事

**建议内容**

- preview / retained / rejected 的分解
- 强调 retained 的 target：
  - `tem1_1btl`
  - `petase_5xh3`
  - `t4l_171l`
  - `subtilisin_2st1`
  - `sod_1y67`

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

重点字段：

- `target`
- `neural_decoder_generated_count`
- `neural_decoder_novel_count`
- `neural_decoder_rejected_count`
- `neural_decoder_injected`

**图型建议**

- 必须是 `统计图`
- 最好是 stacked horizontal bar

**原因**

- 这里要表达利用率与选择性，不是空间构象

#### Figure 2D

**要说什么**

- family-level 的 broad transfer / prior regime

**建议内容**

- family × metric heatmap
- 指标：
  - `mean_overall_score`
  - `mean_best_learned_score`
  - `family_prior_targets`

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\family_summary.csv`

**图型建议**

- 必须是 `heatmap / matrix statistics`
- 不能用 PSE

### Figure 2 中文图注初稿

`图2 | MARS-FIELD 在 twelve-target benchmark 上的主结果。A，final controller 相对 incumbent benchmark 的 paired policy delta。大多数 target 呈现正向增益，说明方法收益是 panel-level 而非个别案例。B，headline benchmark metrics，总结了改进 target 数、下降 target 数、平均 paired delta 以及 neural-decoder retained 候选总数。C，neural decoder utilization，显示 neural decoder 在全部 target 上运行，但仅在部分 target 上保留 novel candidates，表明 decode-time generation 是选择性而非盲目的。D，family-level summary，说明该方法在多种蛋白 family 和记工程先验设置下都能运行。`

---

## Figure 3

### 图的总任务

告诉读者：

为什么方法有效，以及它还在哪些地方不够。

### 整图的一句话结论

MARS-FIELD 的表现不是黑箱偶然结果；主要驱动力和剩余失败模式都具有明确的机制解释。

### 子图设计

#### Figure 3A

**要说什么**

- 哪些 evidence component 真正重要

**建议内容**

- no oxidation
- no surface
- no evolution

每个 ablation 展示：

- changed top candidates
- mean full-minus-ablation score

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\ablation_summary.csv`

**图型建议**

- 必须是 `统计图`
- 最适合 bar + line 双轴总结

**绝对不要**

- 结构图

#### Figure 3B

**要说什么**

- neural branch 在不同 target 上依赖不同 evidence mix

**建议内容**

- target × branch 的 heatmap
- branch：
  - geometry
  - phylogeny
  - ancestry
  - retrieval
  - environment

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

重点字段：

- `neural_gate_geom`
- `neural_gate_phylo`
- `neural_gate_asr`
- `neural_gate_retrieval`
- `neural_gate_environment`

**图型建议**

- 必须是 `heatmap`
- 不适合 PSE

#### Figure 3C

**要说什么**

- 当前 regression 是集中而不是泛滥的

**建议内容**

- 只画 3 个 regression target：
  - `CLD_3Q09_NOTOPIC`
  - `CLD_3Q09_TOPIC`
  - `subtilisin_2st1`

展示：

- incumbent policy score
- final policy score
- delta

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`

重点字段：

- `target`
- `policy_selection_score_current`
- `policy_selection_score_final`
- `policy_selection_score_delta_final_minus_current`

**图型建议**

- 必须是 `简洁统计图`
- 可以是 slope plot

#### Figure 3D

**要说什么**

- decoder 是 selective，不是盲目生成

**建议内容**

- x 轴：preview count
- y 轴：retained novel / preview
- retained target 高亮

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

重点字段：

- `target`
- `neural_decoder_generated_count`
- `neural_decoder_novel_count`

**图型建议**

- 必须是 `统计图`
- scatter / bubble plot 最合适

### Figure 3 中文图注初稿

`图3 | MARS-FIELD 的机制与限制。A，ablation 表明 oxidation 和记 evolution 是当前系统最强的约束来源。B，neural gate heatmap 显示不同 target 对 geometry、phylogeny、ancestry、retrieval 和记 environment 的依赖不同。C，剩余 regression 主要集中在少数 calibration-limited targets，说明系统失败是局部而非系统性的。D，neural decoder 的 retained ratio 说明 decode-time generation 是受控且选择性的。`

---

## Figure 4

### 图的总任务

告诉读者：

方法在真实 target 上不是一个模式，而是不同 case 有不同工作模式。

### 整图的一句话结论

MARS-FIELD 会根据不同 target 的证据结构表现出保守型、稳定改进型、复现型和 stress-test 型等不同控制行为。

### 子图设计

#### Figure 4A：1LBT

**要说什么**

- 安全保守模式
- incumbent `M298L` 被保留

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\combined_ranked_candidates.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_decoder_preview.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\viz_bundle\scene.pml`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\viz_bundle\palette.json`

**图型建议**

- `PSE/结构图` 最适合
- 需要局部 close-up

#### Figure 4B：TEM1

**要说什么**

- incumbent 稳定
- neural decoder 提供 learned alternative

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\combined_ranked_candidates.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\neural_decoder_generated_candidates.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\viz_bundle\scene.pml`
- `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\viz_bundle\palette.json`

**图型建议**

- `PSE/结构图`
- 必须是多位点 close-up

#### Figure 4C：PETase

**要说什么**

- 5XFY / 5XH3 两个结构上可复现 canonical redesign

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\petase_5xfy_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\petase_5xh3_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\petase_5xfy_pipeline\viz_bundle\scene.pml`
- `F:\4-15Marsprotein\mars_stack\outputs\petase_5xh3_pipeline\viz_bundle\scene.pml`

**图型建议**

- `PSE/结构图`
- 两个结构并排展示

#### Figure 4D：CLD

**要说什么**

- topic vs no-topic
- incumbent 与 neural top alternative 的 tension

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_topic_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_notopic_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_topic_pipeline\combined_ranked_candidates.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_notopic_pipeline\combined_ranked_candidates.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_topic_pipeline\viz_bundle\scene.pml`
- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_notopic_pipeline\viz_bundle\scene.pml`

**图型建议**

- 主体必须是 `PSE/结构图`
- 辅助可加一个很小的文字/数值 inset

### Figure 4 中文图注初稿

`图4 | 代表性 case studies。A，1LBT 展示了保守安全控制：系统保留已知稳定的 M298L，同时不让弱 neural-decoder 候选进入最终 shortlist。B，TEM1 展示了 incumbent 稳定但 neural decoder 可贡献有价值 learned alternative 的情形。C，PETase 在两个相关结构上复现了同一 canonical redesign，体现了跨结构一致性。D，CLD 展示了 topic-conditioned 与 no-topic 条件下 controller 行为的差异，并暴露了当前系统在局部高工程信号与最终稳定选择之间的校准张力。`

---

## Figure 5

### 图的总任务

告诉读者：

这个方法的价值边界是什么，应该怎么总结，不要让 Discussion 只有文字。

### 整图的一句话结论

MARS-FIELD 当前最强的价值是“统一 controller + 可运行 neural decoder + panel-level 稳定性”，而不是“已经彻底解决所有 target 的 fully joint end-to-end final model”。

### 子图设计

#### Figure 5A

**要说什么**

- source transition
- current / learned / final policy 的 source 变化

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

重点字段：

- `overall_source`
- `best_learned_source`
- `policy_source`

**图型建议**

- 必须是 `统计图`

#### Figure 5B

**要说什么**

- engineering consistency
- 哪些 gain target 是合理 gain，哪些 failure 是 calibration issue

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`

**图型建议**

- 必须是 `统计图`

#### Figure 5C

**要说什么**

- prior regime map
- 哪些 target 开了 family prior / ASR prior / template weighting

**实际数据来源**

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

重点字段：

- `family_prior_enabled`
- `asr_prior_enabled`
- `template_weighting_enabled`

**图型建议**

- 必须是 `heatmap / matrix`

#### Figure 5D

**要说什么**

- 方法当前的局限与下一步

**建议内容**

- `works now`
- `still limited`
- `what full joint training would require`

**实际数据来源**

- 不主要依赖原始数据
- 依赖方法总结文档：
  - `F:\4-15Marsprotein\mars_stack\docs\mars_field_technical_report_cn_v1.md`
  - `F:\4-15Marsprotein\mars_stack\docs\mars_field_intro_discussion_draft_v1.md`

**图型建议**

- `概念总结 panel`
- 不是结构图

### Figure 5 中文图注初稿

`图5 | 方法定位与边界。A，final controller 在不同 source 之间的选择迁移。B，gain target 与 regression target 的 engineering consistency。C，不同 target 的 prior regime 分布。D，MARS-FIELD 当前已实现能力、剩余限制与 fully joint training 路线图。`

---

## 最后一句最实用建议

### 哪些 panel 必须是统计图

- Figure 2A-2D
- Figure 3A-3D
- Figure 5A-5D

### 哪些 panel 适合用 PSE / PyMOL 结构图

- Figure 4A-4D

### 哪些 panel 可以是示意图

- Figure 1A-1D

### 最不该做的事

- 用结构图去替代 benchmark claim
- 用花哨的 3D 装饰去替代方法逻辑
- 用卡片式大数字去代替真正的数据图

如果你的目标是漂亮又像论文，
最稳妥的分工就是：

- Figure 1：高质量 schematic
- Figure 2 / 3 / 5：高质量统计主图
- Figure 4：高质量 PSE / PyMOL 结构主图
