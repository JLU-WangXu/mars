# MARS-FIELD 主图逐图逐面板说明书 v2（只描述内容与数据，不作图）

## 这份文档怎么用

这份文档的目标非常明确：

1. 不再讨论“怎么画得好看”这种抽象问题。
2. 直接告诉你 Figure 1 到 Figure 5 每个子图要表达什么。
3. 直接告诉你每个子图该用什么真实数据文件。
4. 直接告诉你哪些 panel 必须是统计图，哪些 panel 适合用 PSE / PyMOL 结构图。

如果你要自己做图，最重要的是看每个 panel 下的四件事：

- `一句话结论`
- `应该画什么`
- `实际数据文件`
- `图上必须出现的字`

---

## Figure 1

### 图题建议

`MARS-FIELD integrates heterogeneous evidence into a shared residue field`

### 整图要回答的问题

这个方法到底是什么算法对象？

### 整图一句话结论

`MARS-FIELD` 不是多个 generator 的投票系统，而是把结构、进化、祖先、retrieval 和记环境条件统一投射到共享 residue field 中，再由 controller 与 decoder 在这个 field 上完成决策。

### Figure 1A

#### 一句话结论

五类输入不是独立模块，而是统一证据流。

#### 应该画什么

- 左侧五个输入块
  - Geometric encoder
  - Phylo-sequence encoder
  - Ancestral lineage encoder
  - Retrieval memory encoder
  - Environment hypernetwork

每个块只需要极简 icon 和 2 到 4 个关键词。

#### 实际数据 / 信息来源

这是示意图，不需要直接从某个 csv 画。
但内容必须和代码一致，对应这些模块：

- `F:\4-15Marsprotein\mars_stack\marsstack\field_network\encoders.py`
- `F:\4-15Marsprotein\mars_stack\marsstack\evolution.py`
- `F:\4-15Marsprotein\mars_stack\marsstack\ancestral_field.py`
- `F:\4-15Marsprotein\mars_stack\marsstack\retrieval_memory.py`

#### 图上必须出现的字

- geometry-conditioned compatibility
- phylogenetic adaptation statistics
- ancestral posterior constraints
- motif memory retrieval
- environment-conditioned modulation

#### 图型建议

- 必须是 `schematic`
- 不能用结构图替代

---

### Figure 1B

#### 一句话结论

这个方法真正操作的对象不是 candidate list，而是 residue field。

#### 应该画什么

- 中间一个共享 field 对象
- 两层：
  - site-wise residue energy `U(i,a)`
  - pairwise coupling `C(i,j,a,b)`

可以是：

- 一个 residue-position × amino-acid 的 energy matrix / slab
- 旁边叠一个 pairwise coupling 小图

#### 实际数据文件

可以用这些真实文件中的字段作为内容依据：

- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\position_fields.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\pairwise_energy_tensor.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_position_fields.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_pairwise_energy_tensor.json`

如果你要让这个 panel 更“真”，可以直接从 `1LBT` 的 `249 / 251 / 298` 三个位点抽 top residues 和一条 pairwise edge 做 inset。

#### 图上必须出现的字

- `U(i, a)`
- `C(i, j, a, b)`
- `shared residue field`
- `E(x) = Σ_i U(i, x_i) + Σ_(i,j) C(i, j, x_i, x_j)`

#### 图型建议

- 必须是 `示意图 + 数学对象图`
- 不适合 PSE

---

### Figure 1C

#### 一句话结论

field 同时驱动 candidate controller 和 structured decoder。

#### 应该画什么

- residue field 向右连到两个块：
  - calibrated selector / controller
  - structured decoder

然后再连到：

- ranked designs
- benchmark outputs
- case-study structure bundles

#### 实际数据 / 信息来源

这仍然是示意层，但必须和现有代码路径一致：

- `F:\4-15Marsprotein\mars_stack\marsstack\field_network\neural_model.py`
- `F:\4-15Marsprotein\mars_stack\marsstack\field_network\neural_generator.py`
- `F:\4-15Marsprotein\mars_stack\marsstack\decoder.py`
- `F:\4-15Marsprotein\mars_stack\scripts\run_mars_pipeline.py`

#### 图上必须出现的字

- structured decoder
- calibrated selector
- target-wise normalization
- prior consistency
- safety gating
- final policy

#### 图型建议

- 必须是 `schematic`

---

### Figure 1D

#### 一句话结论

共享 field 不是抽象概念，它在真实 target 上可以被实例化。

#### 应该画什么

- 小 inset
- 选 1LBT 最合适
- 展示：
  - 249 位点 top residues
  - 251 位点 top residues
  - 298 位点 top residues
  - 249-251 的 pairwise edge

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\position_fields.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\pairwise_energy_tensor.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_position_fields.json`
- `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_decoder_preview.json`

#### 图上必须出现的字

- `1LBT example`
- `249 / 251 / 298`
- `field instantiation`

#### 图型建议

- 小型 inset
- 不是主结构图

---

## Figure 2

### 图题建议

`The final MARS-FIELD controller remains benchmark-stable while activating neural decode-time generation`

### 整图要回答的问题

这个方法在 broad benchmark 上站住了吗？

### 整图一句话结论

final controller 在 twelvepack 上总体稳定，且 neural decoder 真实参与了 proposal generation。

### Figure 2A

#### 一句话结论

大多数 target 的 paired policy delta 为正。

#### 应该画什么

- 每个 target 一行
- 横轴：`policy_selection_score_delta_final_minus_current`
- 0 为中线
- 正值绿色，负值红色

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`

#### 需要用的字段

- `target`
- `policy_selection_score_delta_final_minus_current`

#### 图上必须出现的字

- `9/12 improved`
- `3/12 decreased`

#### 图型建议

- 必须是 `统计图`
- 最合适：horizontal lollipop / slope-free delta chart

---

### Figure 2B

#### 一句话结论

面板级 headline metrics 支持主 benchmark 结论。

#### 应该画什么

- 4 个小数值块
- 建议内容：
  - `9/12 targets improved`
  - `3/12 targets decreased`
  - `mean paired delta ~ -0.001`
  - `34 retained neural-decoder candidates`

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

#### 图型建议

- 必须是 `metric summary`
- 不能变成花里胡哨的 PPT 卡片

---

### Figure 2C

#### 一句话结论

neural decoder 是活跃且选择性的，不是盲目生成。

#### 应该画什么

- 每个 target 的 decoder preview / retained / rejected 分解
- 突出 retained > 0 的 target

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

#### 需要用的字段

- `target`
- `neural_decoder_generated_count`
- `neural_decoder_novel_count`
- `neural_decoder_rejected_count`
- `neural_decoder_injected`

#### 图上必须出现的字

- `decoder utilization`
- `preview`
- `retained novel`
- `rejected`

#### 图型建议

- 必须是 `统计图`
- stacked bar 最合适

---

### Figure 2D

#### 一句话结论

这个 benchmark 跨 family、跨 prior regime，不是单一任务。

#### 应该画什么

- family × metrics 的 heatmap / matrix
- 建议指标：
  - mean final score
  - mean best learned score
  - family prior targets

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\family_summary.csv`

#### 图型建议

- 必须是 `heatmap / matrix`

---

### Figure 2 中文图注初稿

`图2 | twelve-target benchmark 上的主结果。A，final controller 相对 incumbent benchmark 的 paired policy delta。大多数 target 呈现正向增益，说明方法收益是 panel-level 而非个别案例。B，headline benchmark metrics，总结了改进 target 数、下降 target 数、平均 paired delta 以及 neural-decoder retained 候选总数。C，neural decoder utilization，显示 neural decoder 在全部 target 上运行，但仅在部分 target 上保留 novel candidates，表明 decode-time generation 是选择性而非盲目的。D，family-level summary，说明该方法在多种蛋白 family 和记工程先验设置下都能运行。`

---

## Figure 3

### 图题建议

`Ablations and neural diagnostics reveal what drives MARS-FIELD and where it still fails`

### 整图要回答的问题

为什么方法有效？剩余失败在哪里？

### 整图一句话结论

MARS-FIELD 的收益不是黑箱偶然结果；主要驱动力和失败模式都具有明确机制解释。

### Figure 3A

#### 一句话结论

oxidation 和 evolution 是最强约束。

#### 应该画什么

- 三个 ablation：
  - no oxidation
  - no surface
  - no evolution
- 展示：
  - changed top candidates 数量
  - mean score effect

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\ablation_summary.csv`

#### 图型建议

- 必须是 `统计图`
- bar + line 组合最合适

---

### Figure 3B

#### 一句话结论

不同 target 对不同 evidence branch 的依赖不同。

#### 应该画什么

- target × branch heatmap
- branch：
  - geom
  - phylo
  - asr
  - retrieval
  - env

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

#### 需要用的字段

- `neural_gate_geom`
- `neural_gate_phylo`
- `neural_gate_asr`
- `neural_gate_retrieval`
- `neural_gate_environment`

#### 图型建议

- 必须是 `heatmap`

---

### Figure 3C

#### 一句话结论

当前 regression 是集中、局部、可解释的。

#### 应该画什么

- 只画 3 个 regression target：
  - `CLD_3Q09_NOTOPIC`
  - `CLD_3Q09_TOPIC`
  - `subtilisin_2st1`
- 展示 incumbent vs final 的 policy score 差异

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`

#### 需要用的字段

- `target`
- `policy_selection_score_current`
- `policy_selection_score_final`
- `policy_selection_score_delta_final_minus_current`

#### 图型建议

- 必须是 `简洁统计图`
- slope plot 最合适

---

### Figure 3D

#### 一句话结论

decoder 的 retained novelty 是 selective，不是 blind exploration。

#### 应该画什么

- x 轴：preview count
- y 轴：retained ratio
- retained 的 target 高亮

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

#### 需要用的字段

- `target`
- `neural_decoder_generated_count`
- `neural_decoder_novel_count`

#### 图型建议

- 必须是 `统计图`
- bubble / scatter 最合适

---

### Figure 3 中文图注初稿

`图3 | 方法机制与限制。A，ablation 表明 oxidation 和记 evolution 是当前系统最强的约束来源。B，neural gate heatmap 显示不同 target 对 geometry、phylogeny、ancestry、retrieval 和记 environment 的依赖不同。C，剩余 regression 主要集中在少数 calibration-limited targets，说明系统失败是局部而非系统性的。D，neural decoder 的 retained ratio 说明 decode-time generation 是受控且选择性的。`

---

## Figure 4

### 图题建议

`Representative case studies reveal distinct controller regimes`

### 整图要回答的问题

在具体 target 上，这个 controller 到底是怎么工作的？

### 整图一句话结论

MARS-FIELD 在不同 target 上表现出不同控制模式：保守安全、稳定 incumbent、跨结构复现、以及 calibration stress test。

### Figure 4A：1LBT

#### 一句话结论

这是保守安全控制模式。

#### 应该画什么

- overview：整体 scaffold
- design_window：249/251/298 的 close-up
- 旁边只保留少量文字：
  - policy winner = `M298L`
  - neural decoder active but filtered

#### 实际数据文件

- 结构图资产：
  - `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\1LBT\overview.png`
  - `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\1LBT\design_window.png`
  - `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\1LBT\figure_session.pse`
- 候选与解释：
  - `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\pipeline_summary.md`
  - `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\combined_ranked_candidates.csv`
  - `F:\4-15Marsprotein\mars_stack\outputs\1lbt_pipeline\neural_decoder_preview.json`

#### 图型建议

- 必须是 `PSE / PyMOL 结构图`

---

### Figure 4B：TEM1

#### 一句话结论

这是“incumbent 稳定 + neural decoder 给出 learned alternative”的模式。

#### 应该画什么

- overview
- design_window close-up
- 文字只点这三点：
  - policy = `H153N;M155L;W229F;M272L`
  - best learned = neural-decoder-derived alternative
  - retained neural-decoder candidates = 5

#### 实际数据文件

- 结构图资产：
  - `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\tem1_1btl\overview.png`
  - `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\tem1_1btl\design_window.png`
  - `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\tem1_1btl\figure_session.pse`
- 候选与解释：
  - `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\pipeline_summary.md`
  - `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\combined_ranked_candidates.csv`
  - `F:\4-15Marsprotein\mars_stack\outputs\tem1_1btl_pipeline\neural_decoder_generated_candidates.csv`

#### 图型建议

- 必须是 `PSE / PyMOL 结构图`

---

### Figure 4C：PETase

#### 一句话结论

这是跨相关结构可复现 redesign 的模式。

#### 应该画什么

- `5XFY` overview + close-up
- `5XH3` overview + close-up
- 少量文字：
  - same canonical aromatic redesign across two structures

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\petase_5xfy\overview.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\petase_5xfy\design_window.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\petase_5xh3\overview.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\petase_5xh3\design_window.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\petase_5xfy\figure_session.pse`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\petase_5xh3\figure_session.pse`

#### 图型建议

- 必须是 `PSE / PyMOL 结构图`

---

### Figure 4D：CLD

#### 一句话结论

这是 calibration stress-test 模式。

#### 应该画什么

- `CLD_3Q09_TOPIC` overview + close-up
- `CLD_3Q09_NOTOPIC` overview + close-up
- 旁边点出：
  - incumbent remains stable
  - neural branch keeps favoring nearby alternative

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\CLD_3Q09_TOPIC\overview.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\CLD_3Q09_TOPIC\design_window.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\CLD_3Q09_NOTOPIC\overview.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\CLD_3Q09_NOTOPIC\design_window.png`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\CLD_3Q09_TOPIC\figure_session.pse`
- `F:\4-15Marsprotein\mars_stack\outputs\paper_bundle_v1\structure_panels\CLD_3Q09_NOTOPIC\figure_session.pse`
- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_topic_pipeline\pipeline_summary.md`
- `F:\4-15Marsprotein\mars_stack\outputs\cld_3q09_notopic_pipeline\pipeline_summary.md`

#### 图型建议

- 必须是 `PSE / PyMOL 结构图`

---

### Figure 4 中文图注初稿

`图4 | 代表性 case studies。A，1LBT 展示了保守安全控制：系统保留已知稳定的 M298L，同时不让弱 neural-decoder 候选进入最终 shortlist。B，TEM1 展示了 incumbent 稳定但 neural decoder 可贡献有价值 learned alternative 的情形。C，PETase 在两个相关结构上复现了同一 canonical redesign，体现了跨结构一致性。D，CLD 展示了 topic-conditioned 与 no-topic 条件下 controller 行为的差异，并暴露了当前系统在局部高工程信号与最终稳定选择之间的校准张力。`

---

## Figure 5

### 图题建议

`Method positioning, deployment scope, and remaining research boundary`

### 整图一句话结论

MARS-FIELD 当前最强的价值是统一 controller + 可运行 neural decoder + broad-panel 稳定性，而不是“已经完成 fully joint end-to-end final model”。

### Figure 5A

#### 一句话结论

source 的选择迁移是受控的，不是盲目从旧分支跳到新分支。

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

字段：

- `overall_source`
- `best_learned_source`
- `policy_source`

#### 图型建议

- 统计图

### Figure 5B

#### 一句话结论

大多数 gain 是 engineering-consistent，而 regression 是集中且有限的。

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`
- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\compare_current_vs_final.csv`

#### 图型建议

- 统计图

### Figure 5C

#### 一句话结论

方法跨多种 prior regime 工作，不是单一任务场景。

#### 实际数据文件

- `F:\4-15Marsprotein\mars_stack\outputs\benchmark_twelvepack_final\benchmark_summary.csv`

字段：

- `family_prior_enabled`
- `asr_prior_enabled`
- `template_weighting_enabled`

#### 图型建议

- heatmap / matrix

### Figure 5D

#### 一句话结论

当前系统已经是 unified controller-decoder，但还不是 fully joint training 终版。

#### 实际数据来源

- `F:\4-15Marsprotein\mars_stack\docs\mars_field_technical_report_cn_v1.md`
- `F:\4-15Marsprotein\mars_stack\docs\mars_field_methods_full_v2.md`
- `F:\4-15Marsprotein\mars_stack\docs\mars_field_related_work_full_v2.md`

#### 图型建议

- 概念总结 panel

---

## 最后一条最重要的执行原则

### 必须用统计图的

- Figure 2A-2D
- Figure 3A-3D
- Figure 5A-5D

### 最适合用 PSE / PyMOL 结构图的

- Figure 4A-4D

### 必须是 schematic 的

- Figure 1A-1D

### 千万不要再做的

- 用结构图去替代 benchmark claim
- 用大数字卡片去替代真正的数据图
- 用 dashboard 风格把 panel 做散
