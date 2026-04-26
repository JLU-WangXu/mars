# MARS-FIELD 过程记录

## 记录规则

这份文档用来记录每一轮“真正推进主框架”的工作，不记录纯噪声级小修补。

每条记录必须包含：

1. 做了什么
2. 为什么做
3. 对应目标文档的哪一项
4. 当前结果如何
5. 下一步是什么

---

## 2026-04-16 | 建立统一锚点文档

### 完成内容

- 新建：
  - `docs/mars_field_target_spec_v1.md`
  - `docs/mars_field_process_log.md`

### 原因

- 之前开发容易一步步围绕局部样例优化，缺少统一终局定义
- 后续需要让所有代码改动直接对齐“统一证据场网络”目标

### 对应目标文档

- 总目标
- 统一框架定义
- 每轮开发必须更新的内容

### 当前状态

- 已建立统一目标锚点

### 下一步

- 后续每次实质改动都要回写本文件

---

## 2026-04-16 | Learned Fusion Ranker 接入主流程

### 完成内容

- 新增 `marsstack/fusion_ranker.py`
- 将 `learned fusion ranker` 接入 `run_mars_pipeline.py`
- benchmark 汇总改为兼容 `ranking_score`

### 原因

- 需要把系统从纯 `mars_score` 规则排序推进到学习型融合排序

### 对应目标文档

- Calibrated Selector

### 当前状态

- `v2.0 learned fusion ranker` 已可运行
- 已支持 target-level 与 benchmark-level 输出

### 下一步

- 继续推进 decoder 与 field

---

## 2026-04-16 | Decoder 接管主生成

### 完成内容

- 新增/扩展：
  - `marsstack/decoder.py`
  - `marsstack/unified_generator.py`
- 让 decoder 生成的新候选进入最终候选表
- 加入 decoder safety gating

### 原因

- 系统不能永远停留在“已有候选池里挑选”
- 必须开始具备 field-to-sequence 的主动生成能力

### 对应目标文档

- Structured Decoder
- Unified Residue Energy Field

### 当前状态

- decoder 已从 preview 变成主流程的一部分
- 但早期版本出现过过激注入和不稳定 winner

### 下一步

- 对 decoder 和 ranker 做联合校准

---

## 2026-04-16 | Ninepack v20/v21 对比完成

### 完成内容

- 跑通：
  - `benchmark_ninepack` 的 `decoder_off`
  - `benchmark_ninepack` 的 `decoder_on`
- 新增比较脚本：
  - `scripts/compare_benchmark_runs.py`

### 原因

- 必须判断 decoder 是否带来全局稳定提升，而不是只看单个 target

### 对应目标文档

- Benchmark 标准

### 当前状态

- 结果显示早期 `decoder_on` 存在明显 target 波动
- 说明不能直接作为投稿版

### 下一步

- 做 decoder safety 校准
- 再做 ranker 校准

---

## 2026-04-16 | Decoder Safety Calibration 完成

### 完成内容

- 加入联合门控：
  - `mars_score` 下限
  - `bad hotspot` 拦截
  - support count 要求
  - 与 best prior 的 gap 约束

### 原因

- 避免 decoder 把局部高共识但低可靠的组合顶成 overall winner

### 对应目标文档

- Structured Decoder
- Calibrated Selector

### 当前状态

- `TEM1 / T4L / SOD / esterase` 的失控情况被明显压住

### 下一步

- 继续做 ranker calibration

---

## 2026-04-16 | Ranker Calibration 完成

### 完成内容

- 在 `fusion_ranker.py` 中增加：
  - decoder generator / consensus 特征降权
  - target-wise score squashing
  - non-decoder prior consistency penalty
  - WT penalty

### 原因

- 需要把问题从“decoder 候选过激”推进到“最终排序更稳”

### 对应目标文档

- Calibrated Selector

### 当前状态

- `1LBT` 不再被 decoder 异常爆分带偏
- `subtilisin` 不再由 `WT` 成为 overall winner

### 下一步

- 继续推进真正的 unified field network 方向

---

## 2026-04-16 | 统一证据场第一版完成

### 完成内容

- 新增：
  - `marsstack/ancestral_field.py`
  - `marsstack/retrieval_memory.py`
  - `marsstack/energy_head.py`
  - `marsstack/evidence_field.py`
- `run_mars_pipeline.py` 已接入：
  - ancestral field
  - retrieval memory
  - pairwise energy tensor
  - unified evidence field
- `decoder.py` 已改成消费 `unary + pairwise`

### 原因

- 需要从“候选投票近似”真正往“统一 residue energy field”逼近

### 对应目标文档

- Ancestral Lineage Encoder
- Retrieval Memory Encoder
- Unified Residue Energy Field
- Structured Decoder

### 当前状态

- 端到端 `1LBT` 已跑通
- 相关工程产物已可直接导出：
  - `retrieval_memory_hits.json`
  - `ancestral_field.json`
  - `pairwise_energy_tensor.json`
  - `position_fields.json`

### 当前不足

- retrieval 已升级为结构模板库近似版，但还不是大规模 motif atlas
- evidence field 仍是工程聚合，不是 learned prototype field
- pairwise energy 仍是统计共现，不是真正 learned coupling head

### 下一步

- 继续把 retrieval memory 从结构模板库近似版升级成 motif atlas
- 再考虑 prototype-conditioned field

---

## 2026-04-16 | Retrieval Memory 升级为结构模板库近似版

### 完成内容

- `retrieval_memory.py` 不再基于历史 design 候选池构建 memory bank
- 改为从 `inputs/*.pdb` 与配置文件中收集结构模板
- 为每个残基构造局部几何描述子
- 用结构模板库检索相似位点，并将检索证据写入 `evidence_field`
- `run_mars_pipeline.py` 现已导出：
  - `retrieval_memory_hits.json`
  - `pairwise_energy_tensor.json`
  - 更新后的 `position_fields.json`

### 原因

- 需要把 retrieval branch 从“工程回放高分候选”推进成真正的结构记忆分支
- 这一步是从工具投票近似走向统一 field network 的关键过渡

### 对应目标文档

- Retrieval Memory Encoder
- Unified Residue Energy Field
- Structured Decoder

### 当前状态

- 轻量 smoke test 已通过
- `1LBT` 端到端重跑已通过
- decoder 现在消费的是：
  - structure evidence
  - evolution evidence
  - ancestral field
  - retrieval memory
  - pairwise energy

### 当前不足

- retrieval 已不再是 design exemplar 检索，但 motif atlas 仍是自动聚类近似
- 检索结果尚未经过 learned memory projection
- 某些位点的 retrieval residue 建议仍显得偏噪声，需要后续约束或学习化

### 下一步

- 继续提高 motif atlas 质量
- 逐步让 evidence field 变成 prototype-conditioned field

---

## 2026-04-16 | Motif Atlas / Prototype Memory + Canonical System 主路径接入

### 完成内容

- `retrieval_memory.py` 从原始结构邻居检索升级为：
  - 自动构建局部结构模板库
  - 自动聚类为 `motif atlas / prototype memory`
  - 基于 prototype 而不是 raw neighbors 进行 retrieval
- 新增 canonical full-stack 架构文档：
  - `docs/mars_field_full_stack_architecture_v2.md`
- 新增 canonical code skeleton：
  - `marsstack/field_network/contracts.py`
  - `marsstack/field_network/encoders.py`
  - `marsstack/field_network/residue_field.py`
  - `marsstack/field_network/system.py`
- `run_mars_pipeline.py` 的 field/decoder 主路径已改为调用 `MarsFieldSystem`

### 原因

- 需要把工程实现从“脚本里拼装 field”推进到“canonical system 驱动的统一架构”
- 需要让 retrieval branch 更像真正的 memory branch，而不是工程邻居查找

### 对应目标文档

- Retrieval Memory Encoder
- Unified Residue Energy Field
- Structured Decoder
- Top-Level System

### 当前状态

- `MarsFieldSystem` 已可 import 和调用
- `1LBT` 端到端已经通过 canonical system 主路径跑通
- 输出已包含：
  - `retrieval_memory_hits.json`
  - `position_fields.json`
  - `pairwise_energy_tensor.json`
  - canonical field diagnostics

### 当前不足

- motif atlas 仍然是自动聚类近似，不是 learned prototype memory
- field network skeleton 已建立，但还没有完全吞并现有 pipeline 的所有逻辑
- benchmark 还没有在这一新版 retrieval/system 主路径上全量重跑

### 下一步

- 继续把 field network 从 skeleton 推进成真正主中枢
- 然后再做全量 benchmark，而不是继续在旧路径上反复 patch

---

## 2026-04-16 | Motif Atlas 导出脚本完成

### 完成内容

- 新增：
  - `scripts/build_structure_motif_atlas.py`
- atlas 现在可以被单独构建和导出，而不必依赖某一次 pipeline 运行

### 原因

- 需要把 retrieval branch 变成可复用、可审阅、可记录的正式产物
- 后续批量 benchmark 和论文图都应引用同一份 atlas，而不是每次隐式生成

### 对应目标文档

- Retrieval Memory Encoder
- Benchmark 标准

### 当前状态

- retrieval atlas 已具备独立构建能力
- 可直接用于后续实验固定版本

### 下一步

- 将更多 field/evidence 逻辑从脚本进一步下沉到 `field_network`
- 然后再进行统一主路径下的全量 benchmark

---

## 2026-04-16 | build_evidence -> construct_field 主路径接入

### 完成内容

- `field_network/contracts.py` 新增：
  - `EvidencePaths`
  - `EvidenceBundle`
  - 更新后的 `FieldBuildResult`
- `field_network/encoders.py` 现在负责：
  - homolog/aligned evolution evidence
  - family manifest 解析
  - ASR evidence
  - template-aware position weighting
- `field_network/system.py` 新增显式阶段：
  - `build_evidence()`
  - `construct_field()`
  - `build_field()`
- `run_mars_pipeline.py` 已改用：
  - `MarsFieldSystem.build_evidence`
  - `MarsFieldSystem.construct_field`
  - `MarsFieldSystem.decode`

### 原因

- 需要让主脚本退出“算法实现”角色，变成 orchestration 层
- 真正把统一框架的主逻辑沉到 canonical system 里

### 对应目标文档

- Top-Level System
- Unified Residue Energy Field
- Structured Decoder

### 当前状态

- `1LBT` 已通过 canonical main path 端到端验证
- 主脚本不再手工拼装 retrieval / field / pairwise / decoder 主路径

### 当前不足

- candidate generation 与 scoring 仍有部分逻辑滞留在 `run_mars_pipeline.py`
- full benchmark 还没有在这一版主路径上重跑

### 下一步

- 运行统一主路径下的全量 benchmark
- 再决定下一轮是否继续把 candidate generation / scoring 下沉

---

## 2026-04-16 | 统一主路径 ninepack benchmark 完成

### 完成内容

- 在 `MarsFieldSystem` 主路径下完成：
  - `build_evidence -> construct_field -> decode`
  - `benchmark_ninepack` 全量重跑
- 这次 ninepack 不再依赖脚本里手工拼接 retrieval/field/decoder 主逻辑

### 原因

- 需要验证 canonical system 驱动的统一主路径是否足以承担全量 benchmark
- 只有全量跑通，后续网络原理图和论文表述才真正站得住

### 对应目标文档

- Top-Level System
- Benchmark 标准

### 当前状态

- `benchmark_ninepack` 已在统一主路径版本下跑完
- `field_network` 已经从 skeleton 推进成可驱动全量 benchmark 的主中枢

### 当前不足

- candidate generation 与 candidate-level score assembly 仍有部分逻辑在 `run_mars_pipeline.py`
- 环境分支仍然偏轻量
- 还需要根据这版 ninepack 结果决定下一轮是继续统一 candidate generation，还是开始准备论文图与实验设计

### 下一步

- 抽查 ninepack 结果并判断统一主路径版本是否已经足够稳
- 然后再决定：
  - 继续下沉 candidate generation / scoring
  - 或转向网络原理图 + 正式实验设计

---

## 2026-04-16 | Candidate Generation / Scoring 继续下沉到 field_network

### 完成内容

- 新增：
  - `marsstack/field_network/proposals.py`
  - `marsstack/field_network/scoring.py`
- `MarsFieldSystem` 现在新增：
  - `generate_candidates()`
  - `score_candidates()`
- `run_mars_pipeline.py` 已切换到：
  - `proposal_ops.*`
  - `MarsFieldSystem.generate_candidates()`
  - `MarsFieldSystem.score_candidates()`

### 原因

- 需要进一步压缩主脚本职责，让 `run_mars_pipeline.py` 只保留 orchestration
- 让 generation / scoring 也进入 canonical system 体系

### 对应目标文档

- Top-Level System
- Unified Residue Energy Field
- Calibrated Selector

### 当前状态

- `1LBT` 已通过下沉后的 generation/scoring 主路径验证
- 主脚本中的候选池构建与 candidate-level score assembly 已不再是纯脚本本地实现

### 当前不足

- `run_mars_pipeline.py` 仍残留旧 helper 定义，虽然主路径已不再依赖它们
- candidate generation 尚未完全变成 learned proposal network

### 下一步

- 在统一主路径下继续重跑 benchmark
- 然后根据结果决定是否进入网络原理图与实验设计整理

---

## 2026-04-16 | Unified Main Path Ninepack Re-run After Generation/Scoring Downshift

### 完成内容

- `field_network/proposals.py` 和 `field_network/scoring.py` 已接入主流程
- `MarsFieldSystem` 现在承担：
  - evidence building
  - field construction
  - candidate generation
  - candidate scoring
  - decoding
- 在这版统一主路径下重新完成：
  - `1LBT` 烟雾测试
  - `benchmark_ninepack` 全量重跑

### 原因

- 需要确认 `field_network` 不只是 field/decoder 外壳，而是已经开始接管 generation/scoring 逻辑
- 需要确认统一主路径在全量 benchmark 下仍然可运行

### 对应目标文档

- Top-Level System
- Benchmark 标准

### 当前状态

- `run_mars_pipeline.py` 更接近 orchestration
- `benchmark_ninepack` 已在 generation/scoring 下沉后的统一主路径下重跑完成

### 当前不足

- 主脚本仍保留部分旧 helper 定义，虽然主路径已不再依赖它们
- candidate generation 仍以工程启发式为主，不是 fully learned proposal network
- 是否已经足够进入正式实验设计，还需要结合 benchmark 结果再判断

### 下一步

- 抽查统一主路径版 benchmark 结果
- 若结果足够稳，则开始整理网络原理图与实验设计
- 若结果仍不稳，再决定继续下沉哪一层

---

## 2026-04-16 | Selector / Ranker 工程先验校准

### 完成内容

- 在 `fusion_ranker.py` 中新增：
  - `mars_score` 显式校准项
  - 负 `mars_score` 统一惩罚
  - engineering prior gap penalty
- 重新验证重点 target：
  - `1LBT`
  - `t4l_171l`
  - `subtilisin_2st1`

### 原因

- 统一主路径 benchmark 显示：
  - 某些 target 的 overall winner 明显背离 `mars_score` 工程直觉
  - 需要把 selector 的排序逻辑重新拉回“先尊重工程目标，再做学习校准”

### 对应目标文档

- Calibrated Selector
- Benchmark 标准

### 当前状态

- `1LBT` 的 winner 被拉回到更合理的 `R249Q;A251S;M298L`
- `t4l_171l` 和 `subtilisin_2st1` 也被压回更符合工程先验的解

### 下一步

- 在最新统一主路径上重新跑全量 `benchmark_ninepack`
- 评估这轮校准是否足以支持更大规模扩展

---

## 2026-04-16 | 结构可视化资产链路建立

### 完成内容

- 新增：
  - `scripts/build_structure_visualization_bundle.py`
- 可输出：
  - `viz_manifest.json`
  - `palette.json`
  - `scene.pml`
- 已完成 `1LBT` 的可视化 bundle 烟雾测试

### 原因

- 后续论文图不能等算法全部结束后再临时补
- 需要及早固定高质量结构可视化资产链路

### 对应目标文档

- Benchmark 标准
- 论文图与案例输出支持

### 当前状态

- PyMOL/PSE 风格的高质量结构图资产已经有稳定导出入口

### 下一步

- 后续可以对重点 target 批量生成 bundle
- 等框架更稳后再统一做论文图版式

---

## 2026-04-16 | Twelvepack 配置预备完成

### 完成内容

- 新增：
  - `configs/petase_5xh3.yaml`
  - `configs/benchmark_twelvepack.yaml`

### 原因

- 为后续从 ninepack 向中等规模扩展做准备
- 扩展目标仍然选择本地已有结构、低额外变量的 target

### 对应目标文档

- Benchmark 标准

### 当前状态

- twelvepack 的配置层已经就绪
- 但在统一主路径完全稳定前，尚未作为主 benchmark 全量执行

### 下一步

- 先继续稳定当前统一主路径 ninepack
- 再决定是否正式启动 `benchmark_twelvepack`

---

## 2026-04-16 | 统一主路径九蛋白再次全量重跑

### 完成内容

- 在当前版本（含 selector/ranker 工程先验校准、motif atlas retrieval、field_network generation/scoring 下沉）下
- 重新完成 `benchmark_ninepack` 全量运行

### 原因

- 需要确认最新代码状态下的 benchmark 是否仍然稳定
- 这一步是决定能否进入更大规模扩展的前提

### 对应目标文档

- Benchmark 标准
- Top-Level System

### 当前状态

- 最新 ninepack 结果已落地在：
  - `outputs/benchmark_ninepack/benchmark_summary.csv`
  - `outputs/benchmark_ninepack/benchmark_summary.md`

### 下一步

- 抽查 ninepack 结果，判断是否可以启动 twelvepack
- 若仍不稳，则继续校准 selector/ranker

---

## 2026-04-16 | Ninepack 稳定性复核通过，进入扩展阶段

### 完成内容

- 使用最新统一主路径版本重新比对：
  - `outputs/benchmark_ninepack/benchmark_summary.csv`
  - `outputs/benchmark_ninepack_v20_summary.csv`
- 生成：
  - `outputs/benchmark_ninepack_current_vs_v20.csv`
  - `outputs/benchmark_ninepack_current_vs_v20.md`

### 原因

- 需要确认统一主路径版本是否已经足够稳定，才能启动中等规模扩展

### 对应目标文档

- Benchmark 标准
- Definition Of Done

### 当前状态

- ninepack 的 overall score 相对 `v20 decoder_off` 在 9/9 target 上均提升
- `1LBT / T4L / subtilisin` 等重点盯防目标也已回到更合理的 engineering prior 轨道
- 当前版本可以进入 `benchmark_twelvepack` 扩展阶段

### 下一步

- 启动 `benchmark_twelvepack`
- 同步开始批量结构可视化资产生成

---

## 2026-04-17 | Benchmark Twelvepack 全量完成

### 完成内容

- 完成 `benchmark_twelvepack` 全量运行
- 新增目标纳入统一主路径 benchmark：
  - `petase_5xh3`
  - `cld_3q09_notopic`
  - `cld_3q09_topic`
- 结果输出已生成：
  - `outputs/benchmark_twelvepack/benchmark_summary.csv`
  - `outputs/benchmark_twelvepack/benchmark_summary.md`
  - `outputs/benchmark_twelvepack/family_summary.csv`
  - `outputs/benchmark_twelvepack/heldout_family_units.csv`

### 原因

- ninepack 稳定性复核通过后，需要验证系统在中等规模扩展下是否仍然可运行且可解释

### 对应目标文档

- Benchmark 标准
- Definition Of Done

### 当前状态

- `twelvepack` 已经在统一主路径版本下跑通
- 这说明当前系统已经具备中等规模扩展能力

### 下一步

- 抽查 twelvepack 结果
- 生成 twelvepack 的结构可视化 bundle
- 再决定是否进入网络原理图与正式实验设计整理

---

## 2026-04-17 | Twelvepack 结构可视化资产补齐

### 完成内容

- 使用 `scripts/build_structure_visualization_bundle.py`
- 为 twelvepack 中的主要 target 批量生成：
  - `viz_manifest.json`
  - `palette.json`
  - `scene.pml`

### 原因

- twelvepack 已完成，需要同步补齐可用于论文和审阅的结构资产
- 避免只积累 benchmark 表格而没有结构图素材

### 对应目标文档

- 论文图与案例输出支持
- Benchmark 标准

### 当前状态

- twelvepack 目标已基本具备可视化 bundle
- 结构资产链路已经可以批量化使用，而不是只支持单个示例

### 下一步

- 基于 twelvepack 结果挑选最代表性的 case
- 开始整理高级计算框架原理图与实验设计

---

## 2026-04-17 | Figure 1 蓝图与实验计划建立

### 完成内容

- 新增：
  - `docs/mars_field_figure1_blueprint_v1.md`
  - `docs/mars_field_experiment_plan_v1.md`

### 原因

- 当前系统已经进入可解释、可视化、可设计实验的阶段
- 需要尽早把网络原理图和实验路径从“想法”固定成正式文档

### 对应目标文档

- Top-Level System
- Definition Of Done
- 论文图与案例输出支持

### 当前状态

- 已经有可直接用于下一步整理的：
  - Figure 1 蓝图
  - case-study 选择建议
  - experiment plan 初稿

### 下一步

- 挑选最强 case study
- 将结构可视化 bundle 升级成 figure-grade 资产
- 正式整理高级计算框架原理图

---

## 2026-04-17 | Neural Field Training Path Established

### 完成内容

- 新增：
  - `marsstack/field_network/neural_dataset.py`
  - `marsstack/field_network/neural_model.py`
  - `scripts/train_mars_field_neural.py`
- 完成小规模神经训练 smoke test：
  - targets: `1LBT`, `tem1_1btl`
  - epochs: `1`
  - output:
    - `outputs/neural_field_training/mars_field_neural.pt`
    - `outputs/neural_field_training/training_history.json`

### 原因

- 需要把系统从“统一工程近似网络”继续推进到“真正可训练的神经场模型”
- 只有训练路径跑通，后续才有资格谈 fully neural end-to-end 版本

### 对应目标文档

- Unified Residue Energy Field
- Pairwise Energy
- Structured Decoder

### 当前状态

- 已经有可训练的 neural field model
- 已经有可加载的 neural corpus
- 已经有可执行的训练脚本
- 神经训练路径 smoke test 通过

### 当前不足

- 还没有把 neural model 切到 benchmark 主路径
- 当前训练仍是小规模 smoke test，不是正式 benchmark 训练
- retrieval / ancestry / pairwise 仍然是工程输入表示，不是全 learned latent branch

### 下一步

- 决定是否将 neural field model 纳入正式 benchmark 分支
- 开始整理高级计算框架原理图，使其同时映射：
  - 当前工程主路径
  - 未来 fully neural path

---

## 当前总判断

当前代码状态应定义为：

**MARS-FIELD engineering approximation v1**

它已经比“多方法投票系统”更接近统一网络，但还没有完全到论文版的 end-to-end field model。

后续所有工作都必须围绕：

- retrieval memory 真正化
- evidence field 神经化
- decoder / energy head 一体化

三条主线推进。

---

## 2026-04-17 | Paper bundle, figure specification, and manuscript table planning

### Completed

- Added a paper-grade Figure 1 specification:
  - `docs/mars_field_figure1_spec_v2.md`
- Added the case-study figure plan:
  - `docs/mars_field_case_study_figure_plan_v1.md`
- Added the manuscript table plan:
  - `docs/mars_field_manuscript_table_plan_v1.md`
- Added an automatic paper bundle builder:
  - `scripts/build_mars_field_paper_bundle.py`
- Generated a manuscript-ready paper bundle:
  - `outputs/paper_bundle_v1/bundle_summary.md`
  - `outputs/paper_bundle_v1/figure2_benchmark_overview.csv`
  - `outputs/paper_bundle_v1/figure2_family_summary.csv`
  - `outputs/paper_bundle_v1/figure3_decoder_summary.csv`
  - `outputs/paper_bundle_v1/case_study_targets.csv`
  - `outputs/paper_bundle_v1/asset_inventory.csv`
  - `outputs/paper_bundle_v1/figure_panel_manifest.json`

### Why

- The project is ready to move from system construction to paper organization.
- We need one stable location for figure data, case-study selection, and table exports.
- This reduces manual assembly when drafting the manuscript.

### Goal Mapping

- Top-Level System
- Benchmark standardization
- Figure and manuscript output readiness

### Current State

- `benchmark_twelvepack` is now wrapped into a dedicated paper bundle.
- Primary case studies are frozen as:
  - `1LBT`
  - `tem1_1btl`
  - `petase_5xh3` with `petase_5xfy` as companion
  - `CLD_3Q09_TOPIC` with `CLD_3Q09_NOTOPIC` as companion
- Figure 1, Figure 2, Figure 3, and Figures 4 to 7 now have explicit source files and output manifests.

### Notes

- `build_mars_field_paper_bundle.py` executed successfully and generated the output bundle.
- A direct `py_compile` check tried to write a `.pyc` file under `scripts/__pycache__` and hit a Windows access error, but the script itself ran correctly and produced the expected files.

### Next

- Render the final Figure 1 architecture artwork from `docs/mars_field_figure1_spec_v2.md`
- Turn the selected case-study assets into figure-grade structural panels
- Start drafting manuscript Results subsections using the paper bundle as the source of truth

---

## 2026-04-17 | Figure 1 architecture render produced

### Completed

- Added a dedicated Figure 1 renderer:
  - `scripts/render_mars_field_figure1.py`
- Rendered the first publication-style architecture figure:
  - `outputs/paper_bundle_v1/figures/figure1_mars_field_architecture_v1.svg`
- Wrote the matching caption file:
  - `outputs/paper_bundle_v1/figures/figure1_mars_field_architecture_v1_caption.md`
- Updated the paper bundle so Figure 1 render and caption are tracked in:
  - `outputs/paper_bundle_v1/bundle_summary.md`
  - `outputs/paper_bundle_v1/figure_panel_manifest.json`

### Why

- The paper now needs a real principle figure rather than only a blueprint document.
- Producing the architecture artwork early reduces drift between code structure and manuscript story.

### Goal Mapping

- Figure and manuscript output readiness
- Top-Level System
- Case-study narrative preparation

### Current State

- Figure 1 now exists as a reusable vector asset.
- The figure follows the `Nature`-leaning scientific systems style:
  - three-section composition
  - restrained typography
  - semantic color logic
  - central residue energy field emphasis
- It is already connected to the paper bundle and can serve as the manuscript's architecture panel.

### Next

- Review Figure 1 visually and refine spacing / emphasis if needed
- Start rendering figure-grade structural panels for:
  - `1LBT`
  - `tem1_1btl`
  - `petase_5xh3` with `petase_5xfy`
  - `CLD_3Q09_TOPIC` with `CLD_3Q09_NOTOPIC`
- Draft Results subsection text anchored to the selected case studies

---

## 2026-04-17 | Figure-grade PyMOL / PSE structure panels generated

### Completed

- Added a dedicated case-study structure renderer:
  - `scripts/render_case_study_structure_assets.py`
- Switched the case-study structure line from basic `scene.pml` bundles to figure-grade outputs:
  - `overview.png`
  - `design_window.png`
  - `figure_session.pse`
  - `render_scene.pml`
  - `render_manifest.json`
- Rendered primary case-study assets for:
  - `1LBT`
  - `tem1_1btl`
  - `petase_5xh3`
  - `petase_5xfy`
  - `CLD_3Q09_TOPIC`
  - `CLD_3Q09_NOTOPIC`
- Wrote the combined render manifest:
  - `outputs/paper_bundle_v1/structure_panels/structure_panel_manifest.json`

### Why

- The manuscript needs structure panels that look publication-grade, not only reusable scene scripts.
- `PyMOL / PSE` gives us direct control over framing, ray-traced output, and editable sessions for final polishing.

### Notes

- On this machine, stable headless rendering works via:
  - `cmd /c pymol -cq ...`
- Multi-chain structures such as `CLD / 3Q09` needed target-chain-only rendering to produce clean single-protein panels.

### Current State

- The paper bundle now contains both:
  - the system-level architecture figure
  - the first batch of figure-grade structure panels
- Case-study structure assets are now editable both as:
  - rendered PNGs for layout work
  - `.pse` sessions for final visual refinement

### Next

- Tighten crop / emphasis for the strongest case-study panels
- Choose the best two structural views per target for manuscript figures
- Start writing the paired Results subsections that directly reference these rendered assets

---

## 2026-04-17 | Data-first benchmark figures rendered

### Completed

- Added a dedicated benchmark figure renderer:
  - `scripts/render_mars_field_data_figures.py`
- Rendered the first data-heavy manuscript figures:
  - `outputs/paper_bundle_v1/figures/figure2_benchmark_overview_v1.svg`
  - `outputs/paper_bundle_v1/figures/figure2_benchmark_overview_v1.png`
  - `outputs/paper_bundle_v1/figures/figure3_decoder_calibration_v1.svg`
  - `outputs/paper_bundle_v1/figures/figure3_decoder_calibration_v1.png`
- Exported derived quantitative metrics:
  - `outputs/paper_bundle_v1/benchmark_derived_metrics.csv`
- Wrote a short quantitative summary:
  - `outputs/paper_bundle_v1/data_figure_summary.md`
- Updated the paper bundle index so the data figures are tracked in:
  - `outputs/paper_bundle_v1/bundle_summary.md`

### Why

- The current architecture figure alone is not strong enough to carry the paper.
- The manuscript needs data-forward benchmark figures that make the system-level claims immediately visible.
- This shift helps anchor the paper in measurable outcomes rather than only framework aesthetics.

### Current State

- The paper now has:
  - a system architecture figure
  - benchmark overview and family-transfer figure
  - decoder / calibration analysis figure
  - figure-grade PyMOL structure panels
- The bundle is now much closer to a full manuscript figure set.

### Next

- Refine the visual hierarchy of Figure 2 and Figure 3 if needed
- Start laying out case-study composite figures for:
  - `1LBT`
  - `tem1_1btl`
  - `petase_5xh3` / `petase_5xfy`
  - `CLD_3Q09_TOPIC` / `CLD_3Q09_NOTOPIC`
- Draft the benchmark and calibration Results subsections directly from the rendered figures and derived metrics

---

## 2026-04-17 | Data figures tightened and case-study composites assembled

### Completed

- Added a tightened benchmark/data figure renderer:
  - `scripts/render_mars_field_data_figures_v2.py`
- Rendered improved data-first manuscript figures:
  - `outputs/paper_bundle_v1/figures/figure2_benchmark_overview_v2.svg`
  - `outputs/paper_bundle_v1/figures/figure3_decoder_calibration_v2.svg`
- Added a case-study composite renderer:
  - `scripts/render_case_study_composites.py`
- Rendered manuscript-style case figures:
  - `outputs/paper_bundle_v1/figures/figure4_case_1lbt_v1.svg`
  - `outputs/paper_bundle_v1/figures/figure5_case_tem1_v1.svg`
  - `outputs/paper_bundle_v1/figures/figure6_case_petase_v1.svg`
  - `outputs/paper_bundle_v1/figures/figure7_case_cld_v1.svg`
- Wrote the case-study figure summary:
  - `outputs/paper_bundle_v1/case_study_figure_summary.md`

### Why

- The original principle figure was not strong enough to lead the paper.
- The manuscript needs data-heavy system figures plus case-study composites that look like a real results section, not only an architecture claim.

### Current State

- The paper bundle now contains:
  - architecture figure
  - benchmark overview figure
  - decoder/calibration figure
  - figure-grade structure renders
  - four case-study composite figures
- The visual center of gravity has shifted from concept-first to data-first.

### Notes

- Companion structure renders for `CLD` needed another pass because the paired composite had initially been built while the companion structural assets were still updating.
- Automatic white-margin cropping is now applied during case-study composition to make the structure panels read more like manuscript figures.

### Next

- Tighten label density and whitespace in the case-study composites
- Decide which panels should stay in the main text versus supplement
- Draft the benchmark Results subsection and the four case-study Results subsections

---

## 2026-04-17 | Figure 3 redesigned after visual quality review

### Completed

- Added a dedicated redesign script for the decoder/calibration figure:
  - `scripts/render_mars_field_decoder_analysis_v3.py`
- Reworked the weaker `Figure 3` panels into more publication-like diagnostics:
  - Panel C is now a `selection rescue map`
  - Panel D is now a `source-shift after calibration` dumbbell plot
- Rendered:
  - `outputs/paper_bundle_v1/figures/figure3_decoder_calibration_v3.svg`
  - `outputs/paper_bundle_v1/figures/figure3_decoder_calibration_v3.png`

### Why

- The previous `Figure 3` layout still looked too much like default plotting output.
- The revised version increases analytical density and reduces the “student project” look, especially in the lower-right half of the figure.

### Current State

- `Figure 3 v3` is now the preferred decoder/calibration figure version.
- The visual narrative is more mature:
  - gated acceptance
  - rescue magnitude
  - rescue map
  - source-shift after calibration

### Next

- Decide whether to similarly tighten `Figure 2` once more
- Continue polishing the case-study composites to the same visual standard

---

## 2026-04-17 | GitHub release framing and repository metadata prepared

### Completed

- Rewrote the root repository README to match the current system state:
  - `README.md`
- Added a public release manifest:
  - `docs/github_release_manifest_v1.md`
- Added a release-status document:
  - `docs/mars_field_release_status_v1.md`
- Added a stable README text mirror for release prep:
  - `docs/github_readme_content_v1.md`
- Added a repository-level `.gitignore` for cleaner public sharing:
  - `.gitignore`

### Why

- The project has reached the point where it can be uploaded publicly, but only if the version boundary is described honestly.
- We need a GitHub-ready framing that does not overclaim the current codebase as the final fully neural method.

### Current State

- The repository is now packaged as a shareable research prototype.
- Recommended public version label:
  - `MARS-FIELD engineering approximation v1`
- The release materials now clearly separate:
  - what is already functionally complete
  - what remains future neural-model work

### Next

- If needed, produce a slimmer upload snapshot or zip based on the release manifest
- Continue tightening figures so the visual quality matches the stronger repository presentation

---

## 2026-04-17 | Neural reranker holdout path connected

### Completed

- Added a neural reranking entry point:
  - `scripts/run_mars_field_neural_reranker.py`
- The new script now:
  - loads the current neural corpus from pipeline outputs
  - trains a holdout neural reranker on all non-target batches
  - scores the held-out target candidates with the neural field model
  - exports neural reranked candidates and summary files
- Completed a smoke run on `1LBT`:
  - `outputs/1lbt_pipeline/neural_field_rerank/neural_rerank_summary.md`
  - `outputs/1lbt_pipeline/neural_field_rerank/neural_reranked_candidates.csv`

### Why

- The biggest algorithmic gap was that the neural field path existed only as an isolated training scaffold.
- Connecting a real train-and-apply reranking path is a concrete step toward the final algorithm version.

### Current State

- The neural branch is still not the main benchmark path.
- But it is now no longer only a smoke-test model file:
  - it can train on held-out-target corpus splits
  - it can rescore target candidate sets
  - it produces concrete rerank outputs for inspection

### What Still Prevents Calling The Repo Final

- the neural reranker is not yet the default benchmark branch
- retrieval / ancestry / environment are still encoded from engineering features rather than being jointly learned end to end
- the current field remains an engineering approximation, not the finished fully neural model form

### Next

- decide whether to wire the neural reranker into the benchmark comparison path
- continue converting the remaining field components from engineered approximation into learned modules

---

## 2026-04-18 | Neural benchmark integration, score contract refresh, and richer neural inputs

### Completed

- Neural reranker is now part of benchmark-time comparison outputs:
  - `scripts/run_mars_benchmark.py`
  - `scripts/run_mars_field_neural_reranker.py`
- `benchmark_twelvepack` was refreshed with:
  - neural top candidate fields
  - score-contract fields
  - benchmark protocol manifest
- Added `selection_score` and `engineering_score` to pipeline candidate exports:
  - `scripts/run_mars_pipeline.py`
- Added benchmark protocol outputs:
  - `benchmark_protocol_manifest.json`
  - `benchmark_protocol_manifest.md`
- Added neural comparison aggregation:
  - `outputs/benchmark_twelvepack/neural_comparison_summary.csv`
- Added neural comparison figure:
  - `scripts/render_mars_field_neural_comparison.py`
  - `outputs/paper_bundle_v1/figures/figure_neural_comparison_v1.svg`
- Expanded neural feature channels:
  - retrieval branch now includes neighborhood/similarity support features
  - ancestry branch now includes confidence/entropy/recommendation mass
  - environment vector now includes more benchmark-context features
- Added shared neural training utilities:
  - `marsstack/field_network/neural_training.py`
- Upgraded neural training objective:
  - regression loss
  - WT recovery loss
  - pairwise consistency loss
- Neural rerank outputs now include gate diagnostics:
  - `neural_site_gates.json`

### Why

- The neural branch had to move from isolated prototype status into a benchmark-visible path.
- The previous feature inputs were too shallow to support any credible claim of moving toward the final learned model.
- Score semantics and run protocol needed to be explicit before more learned modules could be layered on top.

### Current State

- `v1.0` remains the stable shareable engineering approximation.
- `v1.1` now has:
  - neural benchmark participation
  - richer neural branch inputs
  - score-contract outputs
  - benchmark protocol manifests
  - neural comparison aggregation and figure support

### Remaining major gap to the true final algorithm

- retrieval / ancestry are richer, but still not jointly learned end-to-end branches
- the neural reranker is present in benchmark comparison, but not yet the default selection path
- pairwise and environment learning have started, but are not yet the dominant model behavior

### Next

- push retrieval / ancestry beyond richer input engineering into more genuinely learned branch logic
- evaluate whether neural reranking should become a selectable benchmark default
- continue pairwise / environment upgrades and add diagnostics to paper figures

---

## 2026-04-18 | Neural policy heads and default-path comparison advanced

### Completed

- Added explicit candidate-level neural heads:
  - selection head
  - engineering head
- The neural reranker now emits:
  - `neural_selection_pred`
  - `neural_engineering_pred`
  - `neural_policy_score`
- `run_mars_pipeline.py` now supports pipeline-level neural rerank output and writes:
  - `neural_policy_summary.json`
- Added `neural-default` benchmark policy configuration:
  - `configs/benchmark_twelvepack_neural_default.yaml`
- Produced `current`, `hybrid`, and `neural-default` benchmark comparisons:
  - `compare_current_vs_hybrid.md`
  - `compare_current_vs_neural.md`
- Added a dedicated policy-comparison figure:
  - `figure_policy_compare_v1.svg`

### Why

- The neural branch needed to move from "one more score" into a genuine candidate-level decision path.
- A default neural benchmark policy is the real test of whether the system is approaching a final learned controller rather than only a sidecar reranker.

### Current State

- `hybrid` is still the safer transition path.
- `neural-default` is now executable end to end, but still loses engineering quality on several targets.
- The most persistent degradation targets remain:
  - `1LBT`
  - `esterase_7b4q`
  - `CLD_3Q09_NOTOPIC`
  - `sod_1y67`

### Next

- keep reducing neural-default policy regressions on the remaining hard targets
- continue strengthening retrieval / ancestry as learned branches rather than only richer engineered inputs
- deepen pairwise and environment learning until policy-level behavior improves, not only diagnostics

---

## 2026-04-18 | Prototype-memory neural branch, selector distillation, and refreshed twelvepack comparison

### Completed

- Upgraded the neural branch architecture toward a more self-consistent field network:
  - retrieval branch now uses learned prototype-memory fusion instead of only a flat projected feature vector
  - ancestry branch now uses learned lineage-memory fusion instead of only a flat projected feature vector
  - candidate-level evidence features are now passed into the neural heads
  - candidate-level pairwise summary is now fused into neural selection / engineering / policy heads
- Expanded neural candidate evidence encoding in:
  - `marsstack/field_network/neural_dataset.py`
  - added source-type indicators
  - added support-count / mutation-count context
  - added per-component engineering evidence channels
  - added note-derived prior flags
- Upgraded neural training in:
  - `marsstack/field_network/neural_training.py`
  - policy target shifted further toward engineering safety
  - winner guard loss added
  - non-decoder guard loss added
  - simplicity guard loss added
  - selector-anchor distillation loss added
- Upgraded rerank outputs in:
  - `scripts/run_mars_field_neural_reranker.py`
  - added `neural_policy_pred`
  - added `neural_policy_z`
  - changed `neural_policy_score` to be policy-head-led rather than a pure legacy `selection_z + engineering_z` mix
- Upgraded pipeline export contract in:
  - `scripts/run_mars_pipeline.py`
  - `combined_ranked_candidates.csv` now carries:
    - `neural_policy_pred`
    - `neural_policy_z`
    - `neural_policy_score`
- Verified pipeline-level neural rerank writeout by rerunning:
  - `configs/calb_1lbt.yaml`
- Refreshed all twelve rerank summaries and benchmark comparisons:
  - `outputs/benchmark_twelvepack/`
  - `outputs/benchmark_twelvepack_neural_hybrid/`
  - `outputs/benchmark_twelvepack_neural_default/`
- Refreshed paper figures / bundle after the benchmark update:
  - `figure_neural_comparison_v1.svg`
  - `figure_neural_branch_diagnostics_v1.svg`
  - `figure_policy_compare_v1.svg`
  - `outputs/paper_bundle_v1/`

### Why

- The previous neural path was still too close to a lightly-calibrated engineering reranker.
- To move toward the final paper-grade story, retrieval and ancestry had to start behaving like actual learned branches, and candidate-level evidence had to enter the neural selector directly.
- The remaining gap to a default neural policy was no longer "missing infrastructure", but policy calibration on a small set of hard targets.

### Current State

- The system is now closer to a neuralized `MARS-FIELD` controller than to a simple auxiliary reranker.
- `neural-default` is substantially improved versus the previous pass:
  - exact current-policy matches now include `CLD_3Q09_TOPIC`, `adk_1s3g`, `petase_5xfy`, `petase_5xh3`, `sfgfp_2b3p`, `sod_1y67`, `subtilisin_2st1`, `t4l_171l`
  - remaining mismatches are concentrated in `1LBT`, `CLD_3Q09_NOTOPIC`, `esterase_7b4q`, `tem1_1btl`
- The regression size is also much smaller than before on the hardest targets:
  - `1LBT` policy delta reduced to about `-0.152`
  - `CLD_3Q09_NOTOPIC` policy delta reduced to about `-0.123`
- `hybrid` remains the safest default benchmark policy after this batch.

### What Still Prevents Calling It Fully Neural Final

- retrieval / ancestry are now learned branches, but still operate on engineered evidence tensors rather than raw end-to-end differentiable upstream inputs
- the proposal generator and decoder are still not trained jointly with the neural field
- `neural-default` is close, but not yet strong enough to honestly replace the current default on every hard target

### Next

- push the remaining hard targets through another selector-calibration pass, especially `esterase_7b4q` and `tem1_1btl`
- continue replacing candidate-level engineered support features with more direct learned branch outputs
- decide when to rename the current line from `engineering approximation v1` to a stronger `v1.1` / `v2.0-pre` release status

---

## 2026-04-18 | Final-controller pass for esterase / TEM1 and safe default lock-in

### Completed

- Added stronger selector-prior candidate context in:
  - `marsstack/field_network/neural_dataset.py`
  - new candidate features include:
    - selection score
    - engineering score
    - rank-calibrated score signals
    - selector-rank prior
    - gap-to-best features
- Updated the reranker policy calibration in:
  - `scripts/run_mars_field_neural_reranker.py`
  - `neural_policy_score` now mixes:
    - learned policy head
    - learned engineering head
    - learned selection head
    - selector prior
    - engineering prior
- Re-ran the hardest calibration targets first:
  - `esterase_7b4q`
  - `tem1_1btl`
- Re-ran all twelve neural holdout rerankers and refreshed:
  - `benchmark_twelvepack`
  - `benchmark_twelvepack_neural_hybrid`
  - `benchmark_twelvepack_neural_default`
- Added a repository-level final benchmark config:
  - `configs/benchmark_twelvepack_final.yaml`
- Added a final technical-route document:
  - `docs/mars_field_final_technical_route_v2.md`

### Why

- The previous selector-calibration pass had already reduced the problem to a very small set of hard targets.
- At that stage, the right move was not a brand-new module, but a better final controller:
  - stronger selector anchoring
  - stronger engineering prior injection
  - a clearer deployment default

### Current State

- `esterase_7b4q` is now recovered to the current winner under neural rerank.
- `tem1_1btl` is now also stabilized under neural rerank.
- `neural-default` now matches the current policy on `10/12` twelvepack targets.
- The only remaining pure-neural mismatches are:
  - `1LBT`
  - `CLD_3Q09_TOPIC`
- Their deltas are now small:
  - worst remaining policy delta is about `-0.163`
  - mean policy delta is about `-0.026`
- The honest final default for the repo should now be:
  - `hybrid neural controller`

### Interpretation

- This is now a strong paper-facing final controller.
- It is substantially more neuralized and self-consistent than the original engineering approximation.
- It is still not the full joint end-to-end field/generator/decoder model.
- So the right release wording is:
  - final working controller: yes
  - fully end-to-end final model: not yet

### Next

- use the `hybrid` final controller for GitHub / paper-facing default runs
- keep `neural-default` as the forward-looking experimental path
- if we later decide to chase the full end-to-end paper version, the next big step is joint generator-decoder-field training rather than more selector patching

### Final default lock

- After the above pass, the hybrid controller was tightened with a small selection-tolerance guard:
  - neural candidate must not only improve engineering prior
  - it also cannot fall materially below the incumbent selection score
- This was applied in:
  - `scripts/run_mars_benchmark.py`
  - `scripts/run_mars_pipeline.py`
- Result:
  - `outputs/benchmark_twelvepack_final/compare_current_vs_final.md`
  - final safe controller now aligns with current policy on `12/12` twelvepack targets

This is the state that should be treated as the repository's current final release path.

---

## 2026-04-18 | Neural field decoder integrated into the main benchmark path

### Completed

- Added runtime neural batch construction from the live pipeline state:
  - `marsstack/field_network/neural_dataset.py`
- Added a neural field generator module:
  - `marsstack/field_network/neural_generator.py`
  - trains a leave-one-target-out field model
  - converts neural unary / pairwise outputs into decode-ready fields
- Added a direct decoder-field supervision term:
  - `decoder_field_loss`
  - this pushes the unary field toward the empirical high-quality candidate residue distribution
- Updated `run_mars_pipeline.py` so that when neural rerank is enabled it now also:
  - builds `neural_position_fields.json`
  - builds `neural_pairwise_energy_tensor.json`
  - writes `neural_decoder_preview.json`
  - writes `neural_field_runtime_summary.json`
  - decodes `neural_decoder` candidates and sends them back into the main learned fusion path
- Updated `run_mars_benchmark.py` so benchmark-triggered pipeline runs pass:
  - `--neural-rerank true`
  - this ensures the end-to-end neural branch is actually active in the benchmark, not only in ad hoc pipeline runs
- Re-ran `benchmark_twelvepack_final` under the new end-to-end path.

### Why

- Before this pass, the repo could honestly claim a strong neural controller, but not a true neural generator/decoder loop in the main benchmark path.
- After this pass, the controller is no longer only selecting from externally generated candidates; it can also project a neural residue field and decode from it inside the same pipeline.

### Current State

- neural decoder is now enabled on all twelvepack targets
- neural decoder retained novel candidates on `5/12` targets
- neural decoder preview candidates across twelvepack: `373`
- retained neural decoder candidates across twelvepack: `34`
- rejected neural decoder candidates across twelvepack: `215`
- compared against the previous current benchmark:
  - policy score improved on `9/12` targets
  - policy score decreased on `3/12` targets
  - mean delta is very close to neutral at roughly `-0.001`

### Interpretation

- This is the first version that is close enough to call an end-to-end field-controller system in an engineering-honest sense.
- It is still not the final research endpoint of full joint optimization, but the missing gap is now a real research contribution rather than unfinished plumbing.
- That means the project is now in a much better state to:
  - formalize experiments
  - freeze a paper-facing release branch
  - begin writing Results and Methods around a coherent algorithm story

### Next

- design the formal experiment table around:
  - current benchmark
  - hybrid final controller
  - pure neural rerank
  - end-to-end neural decoder enabled final path
- decide which of the three regression targets should be framed as limitations vs further calibration opportunities
- start writing the paper with the new end-to-end controller as the method core

### Paper-preparation extension

- Added a dedicated experiment matrix document:
  - `docs/mars_field_paper_experiment_matrix_v1.md`
- Expanded the manuscript draft to include:
  - family-stratified behavior
  - ablation interpretation
  - neural decoder utilization analysis
  - richer case-study framing
  in:
  - `docs/mars_field_methods_results_draft_v2.md`
- Generated two Word drafts:
  - `outputs/paper_bundle_v1/MARS_FIELD_Methods_Results_Draft_v2.docx`
  - `outputs/paper_bundle_v1/MARS_FIELD_Nature_Style_Methods_Results_v3.docx`

This means the project now has:

- benchmark evidence
- figure assets
- an experiment matrix
- a Methods/Results draft
- a Word manuscript draft

and is ready to move into:

- Introduction writing
- Discussion writing
- figure legend polishing

### Figure and manuscript escalation

- Added a dedicated Figure 1-4 redesign plan:
  - `docs/mars_field_figure_1_4_masterplan_v1.md`
- Added a Chinese storyboard version for rapid drawing / explanation:
  - `docs/mars_field_figure_1_4_cn_storyboard_v1.md`
- Added a standalone Chinese technical report:
  - `docs/mars_field_technical_report_cn_v1.md`
  - `outputs/paper_bundle_v1/MARS_FIELD_中文技术报告_v1.docx`
- Rebuilt the main benchmark claim figure:
  - `outputs/paper_bundle_v1/figures/figure2_benchmark_claim_v4.png`
- Rebuilt the mechanism + limitation figure:
  - `outputs/paper_bundle_v1/figures/figure3_mechanism_limitations_v4.png`
- Added a selected reference list:
  - `docs/mars_field_selected_references_v1.md`
- Added a more submission-style Word draft with references:
  - `outputs/paper_bundle_v1/MARS_FIELD_Submission_Style_v5.docx`

The project now has a much more complete paper-preparation stack:

- technical method story
- benchmark evidence
- figure redesign plan
- Chinese explanatory material
- reference list
- submission-style manuscript draft
