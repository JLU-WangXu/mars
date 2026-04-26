# MARS-FIELD 中文技术报告 v1

## 一、这份报告的用途

这份文档不是论文正文，而是给内部快速统一认知用的中文技术报告。

它的目标有三个：

1. 用中文把 `MARS-FIELD` 目前到底是什么算法讲清楚。
2. 用中文把现在 benchmark 的结果到底说明了什么讲清楚。
3. 给后续画示意图、改主图、写论文、对外汇报提供统一口径。

## 二、现在这个项目到底做到了哪一步

一句话概括：

`MARS-FIELD` 现在已经不是“几个方法拼起来投票”的工程系统，而是一个以共享 residue field 为核心、同时具备 neural controller 和 neural decoder 的统一证据到序列控制框架。

但也必须诚实补一句：

它现在已经足够支撑一篇方法学论文和比较完整的实验故事，
但还不是那个“proposal generator、field、decoder 三者完全 joint training 的最终研究终版”。

所以最准确的表述应该是：

- 作为论文主方法原型：已经到位
- 作为工程上可运行、可 benchmark、可讲清楚的方法体系：已经到位
- 作为 fully joint training 的最终模型：还差最后一步

## 三、MARS-FIELD 的核心算法主线

### 3.1 传统问题的痛点是什么

传统蛋白工程计算流程通常是这样：

- 一部分模块负责根据结构给建议
- 一部分模块负责根据 MSA 或进化给建议
- 一部分模块负责根据 ASR 给建议
- 一部分模块负责根据 Foldseek / motif 邻域给建议
- 最后再用一套手工规则或经验分数把这些候选排一下

这种做法最大的问题不是“没效果”，而是：

- 信号是分裂的
- 模块之间语义不统一
- 候选是先生成后比较，不是统一在一个 decision space 里做决策
- 很容易变成“人工规则堆叠”

所以如果要往高水平论文去讲，主语不能是：

- “我们用了 MPNN”
- “我们用了 ASR”
- “我们用了 Foldseek”

而应该是：

- 我们把这些证据流吸收到一个共享 residue decision space 里
- 最后由统一控制器去完成 proposal、decode、selection

### 3.2 MARS-FIELD 的基本对象是什么

不是 candidate list。

它真正的核心对象是：

- 位点级残基能量 `U(i, a)`
- 位点对耦合能量 `C(i, j, a, b)`

最终一个序列 `x` 的能量可以写成：

`E(x) = sum_i U(i, x_i) + sum_(i,j) C(i, j, x_i, x_j)`

这句话很重要，因为它决定了整个方法的叙事方向：

- 外部所有证据流都不是“投票器”
- 它们是用来参数化这个 residue field 的

### 3.3 五类证据流怎么进入系统

当前版本的 `MARS-FIELD` 一共吸收五类证据：

1. 几何结构证据
   - backbone 条件
   - 局部几何
   - 保护位点 / 可设计位点上下文
   - 氧化热点、表面暴露、柔性信息

2. 系统发育 / 进化证据
   - homolog profile
   - conservation
   - family differential preference
   - template-aware weighting

3. 祖先谱系证据
   - ASR posterior
   - posterior entropy
   - lineage confidence
   - ancestor-informed residue preference

4. retrieval / motif memory 证据
   - motif atlas
   - prototype memory
   - local structural neighbors
   - residue-level support prototypes

5. 环境 / 工程条件证据
   - oxidation pressure
   - flexible-surface burden
   - target-specific engineering context
   - prior availability flags

### 3.4 这些证据流在模型里不是平行投票，而是共享编码

当前实现里，证据不是各算一个分数然后硬加和。

它们经历的是：

1. 各自进入 encoder / branch
2. ancestry branch 和 retrieval branch 再通过 memory bank 做融合
3. environment branch 对其他分支做 modulation
4. 所有分支汇入共享 residue representation
5. 再生成 unary field 和 pairwise field

这一步是整个系统从“工程规则系统”升级到“统一网络框架”的关键。

### 3.5 candidate controller 是怎么工作的

当前系统不只是对 residue field 做解码，还对 candidate 做神经化控制。

candidate controller 看的不只是 sequence 本身，还看：

- residue embedding
- pairwise summary
- source type
- support count
- mutation burden
- engineering components
- selector prior
- gap-to-best 信息

输出的是三个层级的预测：

- selection prediction
- engineering prediction
- policy prediction

这使得 controller 不只是“跟着 mars_score 走”，而是有能力学习：

- 什么候选虽然局部高分但整体不可靠
- 什么候选虽然来自 decoder，但不该被放大
- 什么候选虽然和 incumbent 不完全一样，但值得保留

### 3.6 neural field decoder 是这轮升级最关键的地方

这是当前版本与过去最大的不一样之处。

过去 neural branch 更像：

- 先有候选
- 再 rerank

现在的 neural branch 已经变成：

- 先构造 neural field
- neural field 直接生成 `neural_position_fields`
- neural field 直接生成 `neural_pairwise_energy_tensor`
- 然后由 decoder 在这个 learned field 上解码
- 生成 `neural_decoder` 候选
- 再回到统一 ranking / selection 路径

也就是说：

现在 neural branch 已经不是只“评分”，而是已经在主流程里“参与生成”了。

这也是为什么现在可以更有底气讲：

`MARS-FIELD` 已经是一个 evidence-conditioned neural controller-decoder system，
而不只是后置 reranker。

### 3.7 为什么最后还保留 hybrid final policy

因为系统虽然已经很强，但少数 hard targets 仍然会暴露 calibration 问题。

所以当前最诚实、最稳妥、最适合对外发布的默认策略是：

- neural rerank 开启
- neural decoder 开启
- final selection 用 hybrid policy

这表示：

- 神经分支真实参与决策
- 但不会在 marginal case 上盲目覆盖一个更稳的 incumbent

这个设计非常重要，因为它体现的是：

- 不过度宣传
- 把稳定性作为方法的一部分
- 把 controller 设计成一个工程上可部署的系统，而不是只追求局部 flashy 结果

## 四、当前 benchmark 最关键的结果

当前最应该反复记住的数字是：

- benchmark 面板：12 个 target，10 个 family
- neural decoder enabled：`12/12`
- retained novel neural-decoder candidates：`34`
- 有 retained neural-decoder 候选的 target：`5/12`
- paired policy score improved：`9/12`
- paired policy score negative：`3/12`
- mean paired delta：约 `-0.001`

### 4.1 这些数字各自说明什么

#### neural decoder enabled: 12/12

说明：

- end-to-end neural generation 不是只能在 showcase case 上工作
- 它对整个 benchmark panel 都具备可执行性

这证明的是：

- 方法的普适运行能力
- 不是“演示级”系统，而是“面板级”系统

#### retained novel neural-decoder candidates: 34

说明：

- neural field 不是只在重复已有候选
- 它确实生成了新候选
- 这些候选还通过了后续工程与安全筛选

这证明的是：

- neural decoder 有真实新增设计价值
- 神经分支不是形式上的存在

#### improved on 9/12

说明：

- 方法的正向收益不是一两个 target 的偶然事件
- 是在大多数 target 上都有体现

这证明的是：

- 方法收益具有面板级广泛性

#### negative on 3/12

说明：

- 系统仍然有边界
- 但边界是集中、明确、可分析的

这证明的是：

- 我们可以诚实地写 limitation
- 不是全局不稳定，而是少数 hard cases 仍有 calibration 不足

#### mean paired delta about -0.001

这个数字很容易被误读。

它不表示：

- “方法没用”

它真正表示的是：

- 在引入 neural field decoder 和更复杂 controller 后
- 系统整体没有崩
- 面板级全局仍然稳定

因为同时还有：

- `9/12` 为正

所以这个组合说明的是：

- 系统是“多数 target 有 gain，少数 target 有 regression”
- regression 抵消了 gain 后，均值接近 0

这是一种非常典型、也非常合理的方法学系统状态：

- 说明它已经足够成熟可以写论文
- 但还没成熟到“所有 target 全面压制 incumbent”

### 4.2 当前正向案例和负向案例分别应该怎么讲

#### 正向案例

现在适合重点讲的正向 target 包括：

- `1LBT`
- `adk_1s3g`
- `esterase_7b4q`
- `petase_5xfy`
- `petase_5xh3`
- `sfgfp_2b3p`
- `sod_1y67`
- `t4l_171l`
- `tem1_1btl`

这些案例可以分别代表：

- incumbent strengthening
- engineering improvement
- reproducible redesign
- decoder-assisted policy shift

#### 负向案例

现在适合诚实写进 limitation 的 target 包括：

- `CLD_3Q09_NOTOPIC`
- `CLD_3Q09_TOPIC`
- `subtilisin_2st1`

这几个 target 非常有价值，因为它们能说明：

- controller 仍然会在某些复杂证据场景下低估 incumbent 的稳定性
- neural branch 在某些局部高工程分场景下仍可能 over-explore

Nature 级别的文章不是不能有失败，
而是要把失败写成：

- 有边界
- 有原因
- 有机制解释
- 有后续改进方向

## 五、当前 ablation 说明了什么

当前补出来的 ablation 很有用，因为它能把论文从“只有一张 benchmark 表”推进到“有机制支撑”。

### 5.1 no_oxidation

结果：

- 改变了 `10/12` 个 target 的 top candidate
- 平均分数下降很大

说明：

- oxidation-aware engineering 不是边缘项
- 是系统最强的约束之一

这意味着论文里可以明确说：

- 氧化压力建模是 MARS-FIELD 的核心 engineering signal

### 5.2 no_surface

结果：

- 只改变 `2/12` 个 target 的 top candidate

说明：

- 当前 panel 上 surface term 的全局影响不如 oxidation 和 evolution 强
- 但它仍然在局部 case 上决定选择边界

这意味着：

- 表面信息是辅助稳定项，不是主导项

### 5.3 no_evolution

结果：

- 改变 `6/12` 个 target 的 top candidate
- 对 score landscape 影响很大

说明：

- evolution / profile prior 仍然是系统最关键的 backbone 之一

这很重要，因为它让我们可以反驳一种错误解读：

- 不是“neural 分支把所有传统信息都取代了”

而是：

- neural controller 把这些证据统一起来了
- 但 evolution 仍然是最重要的基础约束之一

## 六、当前 family 层面的结论

family 分层现在可以讲出几件事：

1. 这个 benchmark 不是集中在一个 family 上
2. family prior / ASR prior 不是所有 target 都开，而是异质配置
3. 方法是在多种 evidence regime 下工作的

当前可以直接写进 paper 的点包括：

- total families：10
- family prior active：3 个 target
- ASR prior active：2 个 target
- template-aware weighting：12/12

这组数字的意义是：

- benchmark 不是 homogeneous benchmark
- 方法不是只在“某一种任务设置”上有效
- 它更像一个通用 engineering controller

## 七、这篇论文目前最强的科学主张应该是什么

不要把论文主张写成：

- “我们比某个单模型更强”
- “我们用了更多模块”

更好的主张应该是：

### 主张 1

`MARS-FIELD` 提供了一个统一的 evidence-to-sequence 控制框架，
把结构、进化、祖先、retrieval 和记工程条件投射到共享 residue field 中。

### 主张 2

当前实现已经不只是后置 reranker，
而是在主 benchmark 路径里实现了 neural field decoder，
因此方法已经具有真实的 end-to-end controller-decoder 性质。

### 主张 3

这个统一控制器在 12-target panel 上保持了全局稳定，
同时在多数 target 上获得正向 paired gain，
并在部分 target 上引入了 retained novel neural-decoder proposals。

### 主张 4

方法的收益与限制都是可解释的：

- gains 可以从 field、prior、decoder 协同来解释
- failures 也集中在少量 calibration-limited case 上

## 八、当前还不够 Nature 级别的地方到底是什么

这部分必须明确，不然内部会不断误判状态。

### 8.1 图还不够强

现在最大的短板不是代码，而是图的科学叙事还不够狠。

现有图的问题是：

- 说明性强，结论性弱
- 偏 dashboard，不够 claim-driven
- 缺少 mechanism + limitation 联动

### 8.2 主问题还没压成 2 到 3 个狠结论

Nature 级稿件的文章主线通常非常紧：

- 一个大问题
- 两到三个非常明确的结论
- 四张左右主图把它压实

而现在材料虽然很多，
但主张还没有压得足够狠。

### 8.3 还没有真正 fully joint training

这是最大的学术边界。

当前系统虽然已经是：

- end-to-end controller-decoder

但还不是：

- proposal generator / field / decoder 三者共同优化

所以论文里一定不能写成：

- “fully joint end-to-end protein design foundation model”

更准确的说法应该是：

- neuralized unified controller-decoder architecture
- end-to-end engineering controller
- but not yet fully joint proposal-generator-field training

## 九、现在最适合的论文组织方式

我建议整篇论文按下面的结构来组织：

### Figure 1

讲：

- 方法原理
- evidence streams -> shared field -> controller/decoder

### Figure 2

讲：

- panel-level benchmark claim
- paired delta
- decoder utilization
- family summary

### Figure 3

讲：

- mechanism + limitation
- ablation
- neural diagnostics
- failure cases

### Figure 4

讲：

- 代表性 case studies
- 1LBT / TEM1 / PETase / CLD

这样四张主图就能形成闭环：

1. 方法是什么
2. 方法有用吗
3. 方法为什么有用 / 什么时候会失败
4. 代表性 target 上具体发生了什么

## 十、接下来最正确的动作是什么

不是继续无限补工程。

而是：

1. 重构 Figure 1-4 主图
2. 完成 Introduction + Methods + Results + Discussion
3. 再决定 submission 前还要不要补 supplementary ablation

换句话说，
项目已经从“系统开发主导期”进入“论文组织主导期”。

现在最该做的是把：

- 方法主张
- 主图逻辑
- benchmark 证据
- case study 叙事

压成一套能投稿的故事。
