# MARS-FIELD 项目宏观说明报告（中文草稿 v1）

## 一、项目背景与问题来源

这个项目最初不是为了单纯“做几个更稳定的突变体”，而是想回答一个更大的问题：在火星相关环境、极端氧化环境、低温、冻融、干燥复水、高盐或高氯酸盐等复杂压力下，我们能不能建立一套统一的蛋白工程方法，把分散的结构信息、进化信息、祖先序列信息和环境条件整合起来，更系统地提出可执行的设计方案。

过去常见的做法通常是把几个工具串起来使用。例如先用结构生成工具给出候选序列，再参考同源序列做人工筛选，再结合经验规则去掉明显危险的突变。这种流程并不是完全无效，但问题也很明显：第一，信息是割裂的，结构、进化、祖先和环境约束之间没有统一表达；第二，很多决策停留在人工经验层面，不容易规模化复用；第三，不同 target 之间很难形成统一 benchmark，也难以沉淀成一套可解释、可复现、可继续演化的方法框架。

因此，我们后面逐步把目标收敛为两件事。第一，围绕火星相关多压力场景，针对像 `Cld` 这样的关键功能酶，以及 `AresG`、`DrwH` 这类保护模块，建立更系统的工程路线。第二，在方法层面做一个统一平台，也就是现在的 `MARS-FIELD` / `MarsStack`，把结构、进化、ASR、检索记忆、环境条件和候选解码整合成一个完整流程。现在这套代码已经不再只是几个脚本的集合，而是一个研究原型级的蛋白工程系统。

## 二、我们目前已经做了什么

从工程实现上看，项目已经完成了三层内容。第一层是专题路线本身，包括 `Cld` 的 ASR、结构映射、抗氧化改造建议，以及 `AresG`、`DrwH` 等保护模块的设计和筛选准备。第二层是方法平台，即 `MarsStack`，负责把不同证据源组织到同一条设计链路里。第三层是结果交付层，也就是 benchmark、报告、结构图、论文图和打包产物。

在专题路线方面，我们已经有家族抓取、序列清洗、祖先重建、候选祖先面板输出等脚本；已经形成了 `Cld` 的祖先候选、结构映射结果和第一轮工程建议；也已经形成了 `Ares` 系列的 Round 1、Round 2 方案和 AF3 分析结果。这说明项目已经从概念探索推进到了“早期工程化验证”阶段。

在方法平台方面，`MARS-FIELD` 的核心定位已经比较清楚：它不是一个单独的逆折叠模型，也不是单纯的打分器，而是一个统一的 evidence-to-sequence 框架。它接收目标蛋白结构、野生型序列、可设计位点、保护位点、同源序列、祖先信息和环境条件等输入，然后在内部构造统一的 residue field，再结合 pairwise energy、decoder 和 selector 输出候选序列。

在结果交付方面，平台已经可以对单个 target 运行完整 pipeline，也可以运行多 target benchmark。目前代码里已经有 `triplet`、`sixpack`、`ninepack` 和 `twelvepack` 等基准配置，能输出目标级 summary、family summary、held-out family 结果、ablation 分析、neural comparison，以及论文图和结构案例图。也就是说，这个仓库已经具备了“可运行、可评估、可汇报、可写文章”的基本形态。

## 三、算法与数据流程

当前流程可以概括为“输入层、证据层、场构建层、候选层、排序层、报告层”六步。

第一步是输入层。系统读取目标蛋白的 `PDB` 结构、链 ID、野生型序列、设计位点和保护位点，并解析可选的同源序列、family 数据集、祖先序列输入和模板信息。这部分主要由配置文件控制，每个 target 都有独立的 `yaml` 文件，benchmark 再统一调用这些配置。

第二步是证据层。代码会先对结构进行分析，提取 `SASA`、`B-factor`、距保护位点距离、二硫键、糖基化 motif 等基础特征，并检测氧化热点位点和柔性表面位点。随后系统读取 homolog 或 family 数据构建 profile prior，并根据需要加载 family differential prior 与 ASR profile。对于祖先信息，代码会把祖先后验分布转成 position-wise ancestral field，保留 posterior、entropy、confidence 和 recommendation。对于 retrieval 分支，系统会从本地结构集合构建 motif atlas 或 memory bank，并为每个位点检索近邻结构原型，形成 retrieval recommendation。环境分支则把氧化、低温、低剪切、高氯酸盐等工程条件编码成 context token 或 score hook。

第三步是场构建层。这里是当前系统最核心的抽象。代码不会把不同来源的证据简单相加成一个总分，而是把它们投影到共享 residue decision space 中，形成每个位点对不同氨基酸的候选支持度，也就是 position field。同时系统还会构建 pairwise tensor，表示不同位点突变之间的兼容性或耦合关系。这个步骤对应了 `evidence -> field` 的核心思想，也是 `MARS-FIELD` 和传统脚本堆叠最大的区别。

第四步是候选层。系统目前同时保留多条候选生成支路，包括人工理性候选、基于 `ProteinMPNN` 的候选、可选的 `ESM-IF` 候选、本地 chemistry-aware proposal，以及 field decoder 生成的 `fusion_decoder` 和 `neural_decoder` 候选。也就是说，当前版本不是“纯神经直接生成一切”，而是多路 proposal 并存，然后在统一框架下比较。

第五步是排序层。系统会用 `MarsScore` 和 learned fusion ranker 对候选进行重排。`MarsScore` 当前包括氧化热点硬化奖励、表面水化奖励、氧化危险位点惩罚、进化 prior、ASR prior、family prior、topic-aware score 以及 mutation burden penalty 等项。之后 learned fusion ranker 再结合 source、support、candidate feature、pairwise summary、selector prior 等信息进行二次排序。最新版本里还加入了 neural reranker 和 neural field decoder，并通过 hybrid policy 做最后选择，避免纯神经路径在 hard target 上出现不稳定回归。

第六步是报告层。系统会把结果输出成 `csv`、`json`、`fasta`、`summary.md`、benchmark 汇总、ablation 表、family 表、held-out 表，以及论文图和 PyMOL 结构面板。这样一来，算法输出不只是一个候选序列表，而是完整的研究资产。

## 四、代码结构与执行细节

如果从代码布局来看，建议在 README 或 Word 报告中按下面顺序介绍。

第一部分是 `configs/`。这里存放单 target 配置和 benchmark 配置，是整个系统的调度入口。读者先看配置，就能理解系统支持哪些 target、哪些 benchmark、哪些 policy。

第二部分是 `marsstack/`。这是核心算法库。`structure_features.py` 负责结构特征提取；`evolution.py` 负责 homolog profile、family differential prior 和 template-aware weighting；`ancestral_field.py` 负责 ancestral posterior field；`retrieval_memory.py` 负责 motif atlas 和 retrieval 分支；`mars_score.py` 负责统一工程打分；`topic_score.py` 负责 `Cld`、`DrwH`、`AresG` 等专题加权；`decoder.py` 负责约束式 beam decoding；`fusion_ranker.py` 负责 learned fusion ranking。`field_network/` 则是更高一层的抽象，定义 evidence contract、neural dataset、neural model、generator 和 unified system。

第三部分是 `scripts/`。这里是最适合用户直接运行的入口。`run_mars_pipeline.py` 是单目标主流程；`run_mars_benchmark.py` 是 benchmark 主入口；`build_structure_motif_atlas.py` 负责 motif atlas 构建；各类 `render_*.py` 和 `build_*bundle.py` 则负责把结果转成图、表和文稿。

第四部分是 `outputs/`。这里保存每次 pipeline 或 benchmark 的实际结果，也是最能体现系统是否真正跑通的地方。对于一个研究型工程系统来说，`outputs/` 的存在很重要，因为它表明我们不仅写了方法，还保留了可复查的运行痕迹。

## 五、当前版本的边界与不足

虽然现在的系统已经比较完整，但它还不是最终形态。第一，目前 proposal generator、field 和 decoder 还没有做到 fully joint training。第二，geometry、retrieval、ancestry、environment 这些分支虽然已经进入统一框架，但其中不少仍带有 engineered prior 的成分，而不是纯 learned latent branch。第三，环境分支还偏“proxy + hook”，距离真正成熟的环境条件 hypernetwork 还有距离。第四，当前 benchmark 虽然已经能支撑论文级叙事，但在更大规模 family generalization 和真实实验闭环上还需要加强。

## 六、下一步计划与改进方向

下一步我们准备从四个方向推进。第一，继续加强神经化程度，把 retrieval、ancestry、pairwise head 和 environment branch 从工程近似进一步升级为更完整的 learned module。第二，强化 proposal branch，让 learned generator 真正在更多 target 上能和 local rational branch 正面竞争，而不只是做 rerank 辅助。第三，把 `Cld`、`AresG`、`DrwH` 这些专题路线和 `MARS-FIELD` 平台更紧密地对接，减少“专题在外、平台在内”的分离状态。第四，把 wet-lab 需要的 readout 指标真正纳入方法闭环，例如 ROS 挑战后剩余活性、低温活性、冻融恢复、干燥复水恢复和高氯酸盐条件下功能保留，让系统从“计算研究原型”走向“计算与实验共同闭环”的工程平台。

## 七、建议的 Word 页面布局

如果这份内容后面要转成 Word，我建议按以下页面布局排版：

1. 封面页：标题、版本、日期、项目一句话定位。  
2. 项目背景：问题来源、为什么要做、和传统流程的差别。  
3. 当前完成内容：专题路线、方法平台、交付层三部分。  
4. 算法总流程图：输入、证据、场、候选、排序、输出六步。  
5. 模块说明：按 `configs / marsstack / scripts / outputs` 逐页讲。  
6. benchmark 与结果资产：列出 twelvepack、family、held-out、ablation、paper bundle。  
7. 当前边界与不足：诚实说明没有 fully end-to-end 的部分。  
8. 下一步计划：方法升级、专题对接、实验闭环。  

这版草稿的目的不是一次性定稿，而是先把项目的大框架、代码逻辑和后续方向讲顺。等你提意见后，我们可以再把它改成更像 README、答辩汇报稿，或者直接生成 Word 版正式文档。
