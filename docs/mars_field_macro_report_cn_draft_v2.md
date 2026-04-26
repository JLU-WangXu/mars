# MARS-FIELD 项目宏观架构说明报告（中文草稿 v2）

## 摘要

`MARS-FIELD` 的目标，不是把几个现有蛋白设计工具简单串联起来，而是建立一套统一的证据到序列（evidence-to-sequence）蛋白工程框架。该框架面向火星相关多压力场景，以及更一般的极端环境蛋白工程问题，尝试把结构条件、系统发育信息、祖先重建信息、结构检索记忆与环境条件约束整合到共享的残基决策空间中，再在显式工程约束下生成、筛选并校准候选序列。

从项目现状来看，这套系统已经不再是“若干脚本加若干规则”的原型拼接，而是形成了较完整的研究型工程体系。一方面，我们已经围绕 `Cld`、`AresG`、`DrwH` 等目标建立了专题路线，推进祖先重建、结构映射、局部保护模块设计和第一轮候选输出；另一方面，我们也已经把这些思路抽象为 `MARS-FIELD / MarsStack` 平台，实现了单 target 流程、十二目标 benchmark、family/held-out 统计、ablation、paper bundle 和结构案例输出。

需要强调的是，当前版本最准确的表述仍然是：**一个统一证据场驱动的蛋白工程研究原型**。它已经实现了场式（field-style）的设计逻辑、神经化的候选控制和混合策略选择，但还不是 fully joint、fully learned 的终版端到端模型。因此，这份报告的重点不是夸大系统完成度，而是清楚说明：我们已经做成了什么、它的算法流程如何运行、代码如何组织、它目前的边界在哪里，以及下一步如何继续演化。

## 一、项目背景：为什么要做 MARS-FIELD

这个项目最初来自一个很具体的问题：面对火星相关环境或近似环境，包括低温、干燥-复水、冻融、强氧化、辐射诱导损伤、高盐和高氯酸盐压力，常规的蛋白工程方法很难同时整合多种信息来源，也很难稳定地产生可解释、可复用的工程候选。

传统做法通常是这样的：先用结构生成工具给出一批候选序列，再参考同源序列、家族保守位点或少量祖先信息做人工筛选，最后再用若干经验规则排除明显危险的突变。这种做法在单个案例上可以奏效，但一旦问题变得更复杂，就会暴露出几个根本局限。

第一，信息是割裂的。结构、进化、祖先序列、局部 motif、环境约束分别存在于不同脚本或不同工具中，缺乏统一表达。第二，很多重要决策依赖人工经验，缺少稳定的数学对象，导致流程难以迁移、难以复用。第三，候选通常是“先生成、后修补”，而不是从一开始就在统一设计空间内做约束下的选择。第四，如果想把系统扩大到多个蛋白 family 或多个 target，很容易失去统一 benchmark 和统一叙事。

因此，项目后续逐步收敛为两个彼此关联的目标。第一个目标是生物学与工程目标：围绕 `Cld` 这类火星相关关键功能酶，以及 `AresG`、`DrwH` 这类保护模块，建立多压力条件下的功能韧性工程路线。第二个目标是方法学目标：把这些专题路线中的共同计算需求，抽象成一套统一的 `MARS-FIELD` 框架，使系统不依赖于某一个特定蛋白，也不依赖于某一条单独工具链。

换句话说，`MARS-FIELD` 想解决的不只是“某个蛋白该怎么改”，而是“多源证据如何进入同一个设计决策系统”这个更底层的问题。

## 二、项目总体结构：专题路线与平台路线并行推进

如果从整体工作区来看，当前项目可以分成三层。

第一层是专题生物学路线。这部分主要围绕火星相关压力条件和重点靶标展开，包括：

- `Cld` 路线：关注高氯酸盐/亚氯酸盐相关解毒功能、祖先骨架筛选、结构映射和抗氧化改造建议。
- `AresG` 路线：关注应激条件下形成可逆保护微环境的蛋白模块，不再强调直接 DNA 结合，而更强调冻干、脱水、低温和高渗场景中的保护能力。
- `DrwH` 路线：关注小型紧凑保护域，作为 `cargo-cap` 或局部蛋白保护帽使用，更适合保护酶活 cargo，而不是做 DNA tail。

第二层是方法平台路线，也就是 `MARS-FIELD / MarsStack` 本体。它负责把结构特征、同源序列 profile、祖先后验、retrieval motif 和记忆信息、环境条件、候选解码、混合排序等内容组织到同一条计算链中。

第三层是结果交付层。包括单目标 pipeline 输出、多目标 benchmark、family 和 held-out 汇总、ablation 分析、paper bundle、图形化案例、以及结构可视化资产。这一层很重要，因为它保证系统不仅“理论上能运行”，而且“真正留下了可复查、可汇报、可写作的产物”。

## 三、我们目前已经做成了什么

### 3.1 专题路线方面

在 `Cld` 方向，项目已经完成了第一轮可执行 ASR、候选祖先面板构建、结构映射和抗氧化改造建议，已经能够形成像 `Node007 / Node012 / Node006` 这样的优先骨架候选，用于后续结构预测、表达和活性比较。

在 `Ares` 方向，第一轮 seed constructs 和 AF3 结果已经表明，`AresG` 比 `AresR` 更值得继续推进，因此当前主线从“直接抗辐射/抓 DNA”转向“在应激条件下形成可逆保护微环境”。Round 2 已经围绕 linker 长度和 tail 长度给出了多个版本。

在 `DrwH` 方向，已经完成 `WHy-domain` 家族整理和第一轮祖先面板构建，并提出了更适合的工程定位：它不是 DNA shielding tail，而是一个小而稳的局部保护帽，用于酶活 cargo 或其他脆弱蛋白的应激保护。

### 3.2 平台方法方面

`MARS-FIELD` 当前已经明确不是一个单一模型，而是一套统一证据场框架。核心能力包括：

- 单目标 pipeline，可对一个目标蛋白完整地完成输入解析、证据构建、候选生成、场构建、排序与输出。
- 多目标 benchmark，可在多个 family 和多个 scaffold 上统一运行、统一出表。
- 多源证据整合，包括结构特征、同源序列 profile、family prior、ASR prior、retrieval memory 和环境条件代理。
- 显式场构建，包括 position field 和 pairwise tensor。
- 显式 decoder，包括约束式 beam search。
- learned fusion ranker、neural reranker 和 hybrid selection policy。

### 3.3 结果资产方面

目前平台已经具备 `triplet`、`sixpack`、`ninepack` 和 `twelvepack` 等 benchmark 配置，并且已经产生：

- target-level summary
- family summary
- held-out family units
- ablation summary
- neural comparison summary
- paper bundle
- benchmark figures
- case-study structure assets

这意味着系统已经跨过了“单蛋白试玩”的阶段，进入了“研究原型平台”的阶段。

## 四、算法核心思想：从分散证据到共享残基决策场

`MARS-FIELD` 当前最重要的概念，不是某一个具体生成器，也不是某一个具体打分项，而是“共享残基决策场”。

传统流程的典型问题在于：结构工具给出一个候选、MSA 给出一个候选、ASR 给出一个候选、检索工具给出一个候选，最后大家再投票或相加。这种思路的问题是，每个工具都活在自己的表述空间里，没有真正进入同一个决策对象。

而在 `MARS-FIELD` 中，系统试图把这些不同来源的证据都投影到同一个残基级别的决策空间中。对每个设计位点 `i`，系统不再只问“哪个工具建议哪个氨基酸”，而是问：“在当前多模态证据条件下，这个位点对不同氨基酸的偏好分布是什么？”进一步地，系统还问：“如果两个位点同时发生变化，它们是否兼容？是否会产生组合上的额外风险或收益？”

因此，当前系统的主计算对象可以理解为：

- 位点级的残基偏好场
- 位点对级的耦合张量

这两个对象共同定义了后续 decoder 和 selector 的工作边界。也正是在这一步，项目从“多个工具的拼接”走向了“统一设计场”的方法形态。

## 五、算法数据流程：从输入到输出的完整链路

从可执行实现来看，当前流程可以分成六个阶段。

### 5.1 输入层

系统首先读取目标结构、链 ID、野生型序列、设计位点、保护位点和基础配置。每个 target 都通过独立的 `yaml` 文件定义，这些文件记录：

- 蛋白名与 `PDB` 路径
- 设计位点
- 保护位点和催化位点
- 候选生成参数
- 结构约束参数
- evolution / ASR 数据路径
- topic-aware scoring 是否启用

benchmark 则通过更高一层的配置把这些单 target 组合成统一实验面板。

### 5.2 证据层

系统会为每个 target 构建多源证据。

结构分支从目标 `PDB` 中提取局部理化信息，包括：

- `SASA`
- `B-factor`
- 与保护位点距离
- 二硫键成员关系
- 潜在糖基化 motif

然后系统根据这些特征识别：

- 氧化热点位点
- 柔性表面位点

进化分支则从 homolog 或 family 数据集中提取：

- 基本 profile prior
- family differential prior
- 模板感知的 position weighting

祖先分支会把 ASR 结果转成显式的 ancestral field，保留：

- posterior
- entropy
- confidence
- recommendation

retrieval 分支则从本地结构集合或 motif atlas 中检索局部结构近邻，把 motif 级支持写回位点级 evidence。

环境分支目前更多还是通过 context token、score hook 和 topic-aware 规则进入系统，但已经开始作为一类独立条件被显式建模。

### 5.3 场构建层

系统把上述证据合并到共享 residue field 中。当前代码里，这一步主要由 evidence aggregation 与 field network 完成。

这里的关键不是单纯“给每个位点一个分数”，而是为每个位点保留一个由多条证据共同支持的候选氨基酸集合及其支持强度。同时，系统还会构造 pairwise 兼容性张量，用于描述位点组合时的相互影响。

这一层是整个项目最关键的抽象跃迁：外部所有信息不再各说各话，而是变成统一的可解码对象。

### 5.4 候选生成层

当前系统保留多条候选生成支路，而不是只赌一个模型。主要包括：

- 人工理性候选
- `ProteinMPNN` 候选
- 可选 `ESM-IF` 候选
- local chemistry-aware proposal
- `fusion_decoder` 候选
- `neural_decoder` 候选

这种设计的好处是当前版本仍然兼顾稳定性和探索性：既保留工程上较稳的 local branch，也允许 neural field decoder 在已有场上提出真正新的 candidate。

### 5.5 排序与选择层

生成后的候选不会直接输出，而是进入统一排序层。

第一重是 `MarsScore`，它负责工程意义上的基础评价，当前主要包括：

- 氧化热点硬化奖励
- 柔性表面水化奖励
- 高风险氧化替换惩罚
- evolution / ASR / family prior
- topic-aware score
- mutation burden penalty

第二重是 learned fusion ranker。它会结合候选来源、支持来源、局部证据、pairwise summary 和 selector prior 进行再排序。

第三重是 neural reranker 与 hybrid selection policy。当前默认最稳妥的策略不是纯神经 takeover，而是 hybrid policy，也就是让神经分支积极参与，但在 hard targets 上保留更保守的安全边界。

### 5.6 输出层

系统最终输出的不只是“一个序列”，而是一整套设计资产，包括：

- 候选 `csv`
- shortlist `fasta`
- summary `markdown`
- benchmark 汇总表
- family 与 held-out 表
- ablation 表
- 神经分支比较结果
- paper bundle
- 结构图和 PyMOL 资产

从研究工程角度说，这一点非常重要，因为它把计算流程和后续分析、汇报、写作真正连成了闭环。

## 六、代码结构：如何阅读这个仓库

如果要把代码布局讲清楚，最合适的方式不是按文件大小排序，而是按“用户怎么理解和运行系统”来讲。

### 6.1 `configs/`

这是系统的边界定义层。读者首先应该从这里理解：

- 系统支持哪些 target
- 每个 target 的设计位点在哪里
- benchmark 如何组织
- 当前默认 policy 是什么

### 6.2 `marsstack/`

这是算法核心层。

- `structure_features.py`：结构特征提取
- `evolution.py`：同源 profile 与 family prior
- `ancestral_field.py`：祖先后验场
- `retrieval_memory.py`：motif atlas 与 retrieval memory
- `mars_score.py`：通用工程打分
- `topic_score.py`：`Cld`、`DrwH`、`AresG` 等专题目标
- `decoder.py`：约束式解码
- `fusion_ranker.py`：混合重排与校准

其中 `field_network/` 是更高层抽象，负责定义 contracts、dataset、neural model、generator 与 unified system，是当前方法形态最核心的代码区域。

### 6.3 `scripts/`

这是最接近用户执行入口的层。

- `run_mars_pipeline.py`：单目标主流程
- `run_mars_benchmark.py`：benchmark 主入口
- `build_structure_motif_atlas.py`：结构 motif 图谱构建
- `render_*` / `build_*bundle.py`：图、表、论文资产和文稿生成

### 6.4 `outputs/`

这是运行结果的资产池。对于评估一个研究代码库是否真正成熟，`outputs/` 往往比 `src/` 还更重要，因为它直接体现：

- 是否真的跑过
- 是否形成 benchmark 痕迹
- 是否有可复查的中间结果
- 是否已经能服务论文和汇报

## 七、当前 benchmark 面板与它的意义

当前 `benchmark_twelvepack_final` 并不是最终生物学主线本身，而是平台级验证面板，用来证明系统在不同 protein family、不同折叠、不同设计窗口上都有工作的能力。

其中代表性 target 包括：

- `CALB / 1LBT`
- `TEM-1`
- `PETase`
- `sfGFP`
- `T4 lysozyme`
- `subtilisin`
- `ADK`
- `esterase`
- `SOD`
- `Cld` topic / no-topic 对照

这些 target 的作用不是都去讲火星故事，而是作为平台的跨家族压力测试集，帮助我们验证：

- local rational branch 与 neural branch 的关系
- family prior 是否有作用
- topic-aware scoring 是否有用
- hybrid policy 是否比纯 neural 更稳

也就是说，十二目标 benchmark 主要是方法论证资产，而 `Cld / AresG / DrwH` 才是更接近生物学主线的对象。

## 八、当前版本的边界与不足

虽然系统已经比较完整，但仍有几个边界必须诚实说明。

第一，当前还不是 fully joint 的 end-to-end 模型。proposal generator、field 和 decoder 之间还没有做到完全联合训练。

第二，虽然我们已经引入了 neural field、neural reranker 和 neural decoder，但 geometry、retrieval、ancestry、environment 这几个分支中，仍然有不少 engineered feature 和 prior 的成分，不是纯 learned latent branch。

第三，environment branch 目前更接近“显式约束条件 + proxy 分数项”，还不是成熟的环境 hypernetwork。

第四，专题路线和平台路线虽然正在靠拢，但还没有完全统一。当前状态更像“平台已经有了，专题正在逐步接上去”，而不是从第一天起就在一个完整统一模型里共同演化。

第五，最关键的实验闭环还没完全形成。当前系统已经为 wet-lab 提供了良好的候选与分析资产，但像：

- ROS 挑战后剩余活性
- 低温活性
- 冻融恢复
- 干燥复水恢复
- 高氯酸盐条件下功能保留

这些 readout 还没有完全变成方法训练和选择闭环的一部分。

## 九、下一步改进方向

未来建议从四个方向继续推进。

### 9.1 强化神经化程度

继续把 retrieval、ancestry、pairwise head 和 environment branch 从工程近似升级为更完整的 learned module。

### 9.2 强化 proposal branch

当前 local rational branch 仍然很强，下一步要让 learned generator 真正在更多 target 上具备竞争力，而不只是做 rerank 或补充分支。

### 9.3 推进专题与平台更紧密耦合

让 `Cld`、`AresG`、`DrwH` 不只是平台外部的应用案例，而是平台内部设计目标和先验建模的一部分。

### 9.4 建立计算与实验闭环

最终真正决定项目高度的，不是加多少新术语，而是能不能让实验 readout 进入系统迭代。长期目标应该是让真实实验结果反过来指导：

- 目标函数设计
- policy 选择
- 环境条件建模
- 候选保留与淘汰

## 十、结论

`MARS-FIELD` 当前最准确的定位，是一个**面向极端环境蛋白工程的统一证据场研究原型**。它已经把结构、进化、祖先、retrieval 和记忆、环境约束、候选解码与混合排序整合到同一个系统里，并且已经形成了单目标流程、多目标 benchmark 和论文级结果资产。

但它最有价值的地方，不在于“已经全部完成”，而在于：它已经把原本分散的工程思路，提升成了一个可以继续扩展、继续神经化、继续和实验闭环对接的统一方法平台。这也是为什么当前项目最应该坚持的口径，不是“又加了一个工具”或“又修了一个脚本”，而是：**我们正在把极端环境蛋白工程，逐步推进成一个统一证据场驱动的设计系统。**
