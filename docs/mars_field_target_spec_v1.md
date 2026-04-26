# MARS-FIELD 目标文档 v1

## 1. 文档目的

这份文档不是阶段性想法备忘，而是后续所有工程改动、实验设计、论文写作的统一目标锚点。

从现在开始，`MarsStack` 的开发不再以“加一个新脚本 / 修一个样例 / 让某个 target 更漂亮”为导向，而是以：

**把当前工程系统推进成一个自洽、统一、可发表的 `MARS-FIELD` 证据到序列网络框架**

为唯一主目标。

每次代码改动都需要回答三个问题：

1. 这次改动对应本目标文档的哪一个模块或缺口？
2. 这次改动让系统更接近“统一网络”，还是只是临时修补？
3. 这次改动是否带来了新的 benchmark 证据？

## 2. 总目标

### 2.1 论文级目标

将当前 `MarsStack` 工程系统推进为一个可以用计算机/算法论文语言自洽描述的框架：

**MARS-FIELD: a unified evidence-to-sequence network that maps geometric, phylogenetic, ancestral, retrieval-based, and environment-conditioned signals into a shared residue energy field, and decodes calibrated protein designs under explicit engineering constraints.**

### 2.2 工程级目标

把代码从当前的“多来源候选 + field 近似 + ranker/calibration”推进到更接近以下形态：

- 输入主对象不是 `candidate list`
- 主对象是位点级和位点对级能量：
  - `U(i, a)`: residue unary energy
  - `C(i, j, a, b)`: pairwise coupling energy
- decoder 直接基于 `U + C` 解码
- selector 只做 calibration 和 safety，不再承担主算法职责

## 3. 非目标

以下内容不是当前阶段主目标：

- 单个 target 的漂亮案例优先于统一框架
- 再增加更多“工具名分支”作为论文卖点
- 把现有脚本继续堆成更复杂的候选池
- 用大量 ad hoc 规则掩盖框架本体的不足

## 4. 统一框架定义

### 4.1 Geometric Encoder

目标定义：

- 输入 backbone、局部几何、保护位点、设计位点 mask
- 输出结构条件隐表示 `h_i^geom`

当前工程近似：

- `structure_features.py`
- `analyze_structure`
- `oxidation_hotspots`
- `flexible_surface_positions`

离目标差距：

- 仍是手工结构特征，不是真正的几何编码器

### 4.2 Phylo-Sequence Encoder

目标定义：

- 输入 homolog MSA、family contrast、保守性、适应方向统计
- 输出进化隐表示 `h_i^phylo`

当前工程近似：

- `evolution.py`
- homolog profile
- family differential prior
- structure-aware evolution weighting

离目标差距：

- 仍偏统计型 prior，不是统一编码器

### 4.3 Ancestral Lineage Encoder

目标定义：

- 输入 ASR posterior、祖先深度、祖先不确定性
- 输出祖先隐表示 `h_i^asr`

当前工程近似：

- `ancestral_field.py`
- ASR posterior/confidence/recommendation

离目标差距：

- 已成为显式对象，但还未真正主导 field 的构造

### 4.4 Retrieval Memory Encoder

目标定义：

- 输入位点局部结构描述子
- 从结构记忆库中检索相似 motif / stress exemplar
- 输出检索记忆表示 `h_i^retr`

当前工程近似：

- `retrieval_memory.py`
- 当前以本地结构模板库自动构建 `motif atlas / prototype memory`
- 使用局部几何描述子检索 motif prototype
- motif 检索结果作为 `retrieval_memory` 证据写回 residue field
- `scripts/build_structure_motif_atlas.py` 可单独构建和导出 atlas

离目标差距：

- 还没有更大规模、独立于当前 benchmark 的结构模板库
- prototype 仍然是自动聚类近似，不是 learned memory tokens
- 仍然是工程检索，不是 learned memory encoder

### 4.5 Environment Hypernetwork

目标定义：

- 输入 stress context token
- 输出环境条件调制向量 `z_env`

当前工程近似：

- 仍主要由 `mars_score` 和 topic/rule 系统间接体现

离目标差距：

- 没有真正显式的环境条件网络

### 4.6 Unified Residue Energy Field

目标定义：

- 把结构、进化、ASR、检索、环境证据映射到统一 residue field
- 产出：
  - `U(i, a)`
  - `C(i, j, a, b)`

当前工程近似：

- `evidence_field.py`
- `energy_head.py`
- `field_network/*`

离目标差距：

- `U(i, a)` 仍然是 evidence aggregation 的工程近似
- `C(i, j, a, b)` 目前来自高分序列共现与结构距离，不是 learned pairwise head
- `run_mars_pipeline.py` 已开始通过 canonical `MarsFieldSystem` 主路径构场，但仍未完全脱离脚本 orchestration

### 4.7 Structured Decoder

目标定义：

- 基于 `U + C` 做约束解码
- 满足 protected mask / mutation budget / safety constraints

当前工程近似：

- `decoder.py`
- constrained beam decoder

离目标差距：

- 仍是启发式 beam search，不是更强的结构化解码器

### 4.8 Calibrated Selector

目标定义：

- 做 target-wise calibration
- prior consistency
- safety gating

当前工程近似：

- `fusion_ranker.py`
- factorized ranker + calibration + penalties

离目标差距：

- 仍承担较多主排序职责，未来应逐步退回 calibration head

## 5. 当前开发阶段判断

当前系统位置：

- 已经不是简单脚本流水线
- 已有 `evidence field / ancestral field / retrieval memory / pairwise energy / decoder / calibrated selector`
- 但还没有成为真正论文版的 end-to-end field network

当前阶段可定义为：

**MARS-FIELD engineering approximation v1**

## 6. 必须达成的实验标准

### 6.1 Benchmark 标准

至少完成以下稳定证据：

1. `benchmark_ninepack`
   - `v20 decoder_off`
   - `v21 decoder_on`
   - `v21 calibrated`

2. `held-out family` 视角
   - 不只给 target-level winners
   - 还要看 family-level aggregate

3. `ablation`
   - 去掉 retrieval
   - 去掉 ASR field
   - 去掉 pairwise energy
   - 去掉 calibration

### 6.2 论文前最低结论标准

只有在以下条件同时满足时，才可以进入正式投稿准备：

1. 全量 benchmark 上多数 target 稳定提升
2. 不再出现明显不合理的 winner
   - 例如无意义 `WT` 回退
   - 明显坏 hotspot 替换却高分
3. family-level 不出现系统性退化
4. 至少一个代表案例可以给出生物学可解释的结果

## 7. 当前最该推进的三件事

### 优先级 A

让 `retrieval_memory` 从“自动聚类的 motif atlas”升级到“learned prototype memory / motif token bank”。

### 优先级 B

让 `evidence_field` 从工程聚合进一步逼近 prototype-conditioned residue field。

### 优先级 C

让 `energy_head` 和 decoder 更紧密，逐步降低 `fusion_ranker` 的主排序职责。

## 8. 每轮开发必须更新的内容

每次有实质改动，必须同步更新：

1. `mars_field_process_log.md`
2. 本文档中的“当前阶段判断”或“当前最该推进的三件事”

如果改动没有办法映射到本文件中的某个模块或目标，就说明这次改动大概率是偏离主线的。

## 9. 当前版本的判断

当前版本不再适合描述成：

- 多方法投票
- 候选池 rerank 系统

而应该描述成：

**一个统一证据场的工程近似网络框架**

这是后续所有实现、图示、论文文本都必须坚持的口径。
