# MARS-FIELD Figure 1-4 中文逐图说明书 v1

## 这份文档怎么用

这份文档是给你快速分析、汇报、画示意图、和对接设计师/插画师用的。

每张图我都按下面几个问题来写：

1. 这张图到底要回答什么科学问题？
2. 这张图一句话要让读者记住什么？
3. 每个 panel 应该画什么？
4. 每个 panel 不应该画什么？
5. 这张图的图注应该怎么讲？

如果你要自己画草图，可以直接按每张图的 `面板脚本` 来画。

## Figure 1

### 图题建议

`MARS-FIELD integrates heterogeneous evidence into a shared residue field`

中文理解：

`MARS-FIELD 将多模态证据流投射到共享 residue field 中`

### 这张图要回答的问题

`MARS-FIELD` 到底是不是一个统一算法？

### 这张图一句话要传达的结论

`MARS-FIELD` 不是几个 generator 的投票系统，
而是一个把结构、进化、祖先、retrieval、环境条件统一映射到 residue decision space 的 field controller。

### 面板脚本

#### Panel A：五类证据流输入

画什么：

- 左边五个输入模块
  - geometry
  - phylogeny
  - ancestry
  - retrieval memory
  - environment

每个模块内部只画最关键的图标：

- geometry：backbone / local graph
- phylogeny：MSA / family tree style strip
- ancestry：ancestor node + posterior distribution
- retrieval：motif atlas / memory bank
- environment：oxidation / stress / engineering context

想表达什么：

- 外部方法不是一个个独立投票器
- 而是 evidence stream

别这样画：

- 不要画成一个很长的软件流程图
- 不要把 MPNN / ASR / Foldseek logo 堆在一起
- 不要画成“先 A 后 B 后 C 后 D”的流水线

#### Panel B：共享 residue field

画什么：

- 整张图最中心
- 一个 layered residue field object
- 用两层表示：
  - site-wise `U(i, a)`
  - pairwise `C(i, j, a, b)`

可以画成：

- 中间一个 residue-position × amino-acid 的 energy sheet
- 上面叠一层 pairwise coupling arcs / heatmap

想表达什么：

- 所有证据流最后都被投射到同一个 residue decision space
- 最终方法真正操作的对象不是 candidate list，而是 residue field

别这样画：

- 不要只画成 candidate box
- 不要把 field 画成纯装饰性的 3D 方块
- 不要没有数学对象

#### Panel C：decoder + controller

画什么：

- residue field 向右流向两个模块
  - structured decoder
  - calibrated selector / controller

structured decoder 上建议写：

- neural field decoder
- constrained beam decoding
- energy-guided sequence search

selector 上建议写：

- target-wise calibration
- prior consistency
- safety gating
- final policy

想表达什么：

- 这个系统不只是评分
- 它既能 decode，又能 select

#### Panel D：方法公式和一句话 claim

画什么：

- 小公式块：
  - `E(x) = sum_i U(i, x_i) + sum_(i,j) C(i, j, x_i, x_j)`
- 一句小字：
  - evidence streams parameterize a shared residue field rather than a generator vote

想表达什么：

- 给 reviewer 一个非常明确的“方法主对象”

### 图注中文草稿

`图1 | MARS-FIELD 方法框架。MARS-FIELD 将几何结构证据、系统发育序列证据、祖先谱系证据、retrieval-based motif memory 和记工程环境条件统一投射到共享 residue field 中。该 residue field 由位点级残基能量和位点对耦合能量共同定义，并进一步驱动受约束的序列解码和校准后的最终选择。该框架的核心不是多个 generator 的投票，而是多模态证据在统一 residue decision space 中的融合。`

## Figure 2

### 图题建议

`The final MARS-FIELD controller remains benchmark-stable while activating neural decode-time generation`

中文理解：

`最终版 MARS-FIELD 控制器在保持 benchmark 稳定的同时，引入了真实的 neural decode-time generation`

### 这张图要回答的问题

方法到底有没有在 broad benchmark 上站住？

### 这张图一句话要传达的结论

在 twelvepack 上，MARS-FIELD final controller 没有因为 neural decoder 的引入而整体崩溃；
相反，它在多数 target 上保持或提升了 policy 表现，同时 neural decoder 真实参与了 proposal generation。

### 这张图是最重要的主结果图

这张图不能再像过去那样只是一些零散 bar/scatter。

它必须变成：

- 一眼看出 benchmark 结论
- 一眼看出 decoder 真的在工作
- 一眼看出 gains 和 failures 的分布

### 面板脚本

#### Panel A：paired delta 主面板

画什么：

- 每个 target 一行
- x 轴是 `policy_selection_score_delta_final_minus_current`
- 用 lollipop / horizontal dot plot
- 正值一侧是暖色
- 负值一侧是冷色

行上直接标：

- target name
- family

想表达什么：

- 9/12 positive
- 3/12 negative
- gains 是 panel-level，不是少数 case

别这样画：

- 不要再用松散 scatter
- 不要没有 target label
- 不要让读者自己去猜哪几个 target 好、哪几个不好

#### Panel B：headline metrics summary

画什么：

- 4 个紧凑 summary box
  - `9/12 improved`
  - `3/12 negative`
  - `mean delta ~ -0.001`
  - `neural decoder enabled 12/12`

想表达什么：

- reviewer 一眼能抓到主结果

#### Panel C：neural decoder utilization

画什么：

- 每个 target 一个 stacked bar
  - preview
  - retained
  - rejected

或者分三层短条：

- neural decoder preview count
- retained novel count
- rejected count

特别高亮 retained 的 5 个 target：

- TEM1
- PETase 5XH3
- T4L
- subtilisin
- SOD

想表达什么：

- neural branch 不是摆设
- 不是所有 target 都盲目注入 novelty
- decoder 是 selective、calibrated 的

#### Panel D：family-level summary

画什么：

- family × metric 的简洁矩阵
- metrics 可包括：
  - mean final score
  - best learned score
  - family prior active
  - ASR prior active

想表达什么：

- benchmark 是 heterogeneous 的
- 方法不是只在单一家族上工作

### 图注中文草稿

`图2 | twelve-target benchmark 上的主结果。A，final controller 相对 incumbent benchmark 的 paired policy delta，显示 12 个 target 中 9 个为正增益，3 个为负增益。B，panel-level headline metrics，总结了 paired gain 分布与全局稳定性。C，neural field decoder 的利用率统计，显示 neural decoder 在全部 target 上启用，但仅在部分 target 上保留 novel decoded candidates，说明其行为是选择性且受控的。D，family-level summary，表明该 benchmark 跨越多种蛋白 family 和记工程先验设置。`

## Figure 3

### 图题建议

`Ablations and neural diagnostics reveal what drives MARS-FIELD and where it still fails`

中文理解：

`ablation 与 neural diagnostics 揭示了 MARS-FIELD 为什么有效，以及它还在哪里失效`

### 这张图要回答的问题

为什么这个方法有效？
它的限制又是什么？

### 这张图一句话要传达的结论

MARS-FIELD 的表现不是黑箱偶然结果；
它依赖明确的 oxidation / evolution / prior 约束，
而剩余失败也集中在可解释的 calibration-limited cases 上。

### 这张图是 Nature 级叙事最关键的一张

因为它决定这篇文章看起来像不像成熟方法学工作。

### 面板脚本

#### Panel A：ablation summary

画什么：

- 三个 ablation：
  - no oxidation
  - no surface
  - no evolution
- 每个 ablation 画两个数：
  - changed top candidates
  - mean score effect

最好用：

- compact summary bars
- 或者两列数值加极简图标

想表达什么：

- oxidation 是最强的工程约束之一
- evolution 是最强的基础约束之一
- surface 是辅助项，不是主导项

#### Panel B：neural gate composition

画什么：

- 每个 target 的 gate composition
- geometry / phylo / ancestry / retrieval / environment

建议：

- heatmap
- 或者 stacked bar matrix

想表达什么：

- neural branch 不是黑箱
- 不同 target 确实调用不同 evidence regime

#### Panel C：failure / limitation panel

画什么：

- 只展示 3 个 regression target：
  - CLD_3Q09_NOTOPIC
  - CLD_3Q09_TOPIC
  - subtilisin_2st1

每个 target 一行，展示：

- incumbent policy
- final policy
- delta
- 一个简短 failure tag

例如：

- over-exploration
- local engineering optimum over-promoted
- calibration-limited replacement

想表达什么：

- 失败是集中且可解释的
- 不是系统性不可靠

#### Panel D：decoder selectivity

画什么：

- retained vs rejected 的 target-level分布
- 强调：
  - neural decoder enabled 12/12
  - retained 5/12

想表达什么：

- decoder 的行为是 selective，不是 blind generation

### 图注中文草稿

`图3 | 方法机制与限制。A，component ablations 表明 oxidation 和记 evolution 是当前系统最重要的约束来源。B，neural gate composition 显示不同 target 对 geometry、phylogeny、ancestry、retrieval 和记 environment 的依赖不同。C，剩余 regression 主要集中在少数 calibration-limited cases，说明系统的失败是局部而非系统性的。D，neural decoder 的 retained/rejected 分布表明 decode-time generation 是选择性触发的，而不是无约束的候选爆炸。`

## Figure 4

### 图题建议

`Representative case studies reveal distinct operating regimes of the controller`

中文理解：

`代表性案例揭示了控制器在不同 target 上的不同工作模式`

### 这张图要回答的问题

在具体蛋白上，这个系统到底在做什么？

### 这张图一句话要传达的结论

MARS-FIELD 不是“所有 target 都一个模式”；
它会根据不同 target 的证据结构，表现出保守型、稳定改进型、复现型和 stress-test 型等不同工作模式。

### 面板脚本

#### Panel A：1LBT

要讲什么：

- 保守安全控制
- incumbent `M298L` 被保留
- neural decoder 虽然活跃，但没有强行注入不可靠候选

画什么：

- 结构 close-up
- 298 位点
- incumbent 与 decoder 候选关系
- 一条小注释：
  - safety-preserving controller behavior

#### Panel B：TEM1

要讲什么：

- incumbent 保持稳定
- neural decoder 贡献 learned alternative
- 体现“保留 top winner + surfacing useful alternatives”

画什么：

- 4 位点 close-up
- incumbent vs best learned candidate
- neural_decoder retained 候选标记

#### Panel C：PETase

要讲什么：

- 两个结构上的可复现 canonical redesign
- 说明系统不是为了 novelty 而 novelty

画什么：

- 5XFY / 5XH3 并排
- 同一组 aromatic redesign
- 高度统一的局部结构配图

#### Panel D：CLD

要讲什么：

- topic / no-topic 对照
- incumbent 与 neural top alternative 的 tension
- 这是方法限制和研究潜力的集中体现

画什么：

- topic vs no-topic 两个小窗口
- 局部氧化壳 close-up
- residue substitutions 的简洁标签

### 图注中文草稿

`图4 | 代表性 case studies。A，1LBT 展示了保守安全控制：系统保留已知稳定的 M298L，同时不让弱 neural-decoder 候选进入最终 shortlist。B，TEM1 展示了 incumbent 稳定但 neural decoder 能贡献有价值 learned alternative 的情况。C，PETase 在两个相关结构上复现了相同的 canonical redesign，说明该控制器具有跨结构一致性。D，CLD 展示了 topic-conditioned 与 no-topic 条件下的控制器行为差异，并揭示了当前系统在局部高工程信号与最终稳定选择之间的校准张力。`

## 最后给你一句最简画图提示

如果你只想快速手绘一版草图，记住下面四句话就够了：

- Figure 1：讲“方法是什么”
- Figure 2：讲“方法 broadly 有用吗”
- Figure 3：讲“为什么有用 / 哪里还不够”
- Figure 4：讲“在具体 target 上到底发生了什么”
