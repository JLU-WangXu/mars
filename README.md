# MARS-FIELD

## 1. Positioning / 项目定位

**EN**  
`MARS-FIELD` is a protein engineering research prototype. It maps structural, evolutionary, ancestral, retrieval-based, and environment-conditioned evidence into a shared residue decision field, then decodes and calibrates candidate designs under explicit engineering constraints.

**中文**  
`MARS-FIELD` 当前的准确定位是一个蛋白工程研究原型。它把结构、进化、祖先重建、结构检索记忆和环境条件等证据统一投到共享残基决策场中，再在显式工程约束下生成、筛选和校准候选序列。

Current public status / 当前对外状态：

- benchmarked on `ninepack` and `twelvepack`
- supports config-driven target pipelines
- supports arbitrary `PDB -> analyze/design`
- produces paper-style figures and structure assets
- should be described as **engineering approximation v1**

Do **not** describe it as:

- a completed fully neural end-to-end field model
- a final production package
- a fully joint generator-decoder-field foundation model

不要对外写成：

- fully finished end-to-end 最终模型
- 生产级成品系统
- fully joint 的最终基础模型

---

## 2. What Works / 当前已经完成什么

**EN**

What already works:

- config-driven single-target pipeline
- benchmark runner with family / held-out / ablation outputs
- structure / evolution / ancestry / retrieval / environment field integration
- local proposal generation, `ProteinMPNN`, optional `ESM-IF`, decoder, and calibrated selector
- topic overlays for `Cld`, `DrwH`, and `AresG`
- paper bundle, figures, and structure-render assets
- raw `PDB -> analyze/design` entrypoint

What is not final:

- not fully joint end-to-end training
- not zero-annotation biology
- not a zero-configuration production deployment
- some branches still rely on optional external components

**中文**

已经完成的部分：

- 基于配置文件的单目标设计流程
- 多目标 benchmark，包括 family / held-out / ablation 输出
- 结构 / 进化 / 祖先 / retrieval / 环境条件的统一 field 化
- local proposal、`ProteinMPNN`、可选 `ESM-IF`、decoder 和 selector
- `Cld` / `DrwH` / `AresG` 的专题加权
- paper bundle、图表和结构资产
- 任意 `PDB -> analyze/design` 新入口

还不是最终状态的部分：

- 还不是 fully joint end-to-end 训练
- 还不是零注释生物学系统
- 还不是零配置生产部署
- 一些分支仍依赖外部组件

---

## 3. Original Mars Program vs Current Platform / 最初火星任务 vs 当前平台

**EN**

The original Mars-oriented biological program emphasized:

- `Cld` as the main functional enzyme line
- `AresG` / `DrwH` as protection-module lines
- multi-stress Mars-relevant functional resilience as the biological story

The current implementation is stronger in:

- unified algorithmic framing
- benchmark-first engineering validation
- field-style evidence integration
- reusable design infrastructure

Still missing:

- full reintegration of Mars-specific wet-lab readouts
- full biological closed loop
- mature raw-protein usability as a public-facing product

**中文**

最初火星路线强调的是：

- `Cld` 作为主功能酶
- `AresG` / `DrwH` 作为保护模块
- 火星多压力条件下的功能韧性作为生物学主线

当前实现更强的是：

- 统一算法框架
- benchmark 优先的工程验证
- field-style 证据整合
- 可复用的平台化基础设施

还缺的是：

- 火星专题 wet-lab readout 重新接回平台
- 生物学闭环
- 面向任意蛋白的成熟公开产品体验

See also / 另见：

- [`docs/mars_gap_analysis_v1.md`](docs/mars_gap_analysis_v1.md)
- [`docs/mars_biology_completion_status_v1.md`](docs/mars_biology_completion_status_v1.md)

---

## 4. Install / 安装

### Recommended: Linux GPU / 推荐 Linux GPU

```bash
cd mars_stack
bash scripts/bootstrap_linux_gpu.sh
```

Main files / 关键文件：

- [`environment.linux-gpu.yml`](environment.linux-gpu.yml)
- [`requirements.txt`](requirements.txt)
- [`scripts/check_mars_runtime.py`](scripts/check_mars_runtime.py)

Default CUDA target is `12.1`.

如需改 CUDA 版本：

```bash
export MARS_CUDA_VARIANT=12.4
bash scripts/bootstrap_linux_gpu.sh
```

### CPU fallback / CPU 兜底

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

CPU 模式适合：

- 浏览已有输出
- 生成 HTML / Word 交付
- 跑基础结构分析
- 跑部分轻量流程

不适合作为完整 proposal stack 的首选环境。

### Runtime Check / 运行检查

```bash
python scripts/check_mars_runtime.py
```

This reports / 会报告：

- Python package status
- optional tool status (`mafft`, `iqtree2`)
- vendor/model presence
- branch usability for `ProteinMPNN`, `ESM-IF`, and neural branches

Cross-machine setup note / 跨机器部署说明：

- [`docs/setup_cross_machine_release_v1.md`](docs/setup_cross_machine_release_v1.md)

---

## 5. Two Usage Modes / 两种使用方式

### A. Config-driven pipeline / 配置驱动流程

Single target:

```bash
python scripts/run_mars_pipeline.py --config configs/tem1_1btl.yaml --top-k 12
```

Benchmark:

```bash
python scripts/run_mars_benchmark.py --benchmark-config configs/benchmark_triplet.yaml --top-k 12
```

这是当前最稳定、最完整的使用路径。

### B. Arbitrary PDB analyze/design / 任意 PDB 分析与设计

Analyze only:

```bash
python scripts/run_mars_autodesign.py analyze --pdb inputs/tem1_1btl/1BTL.pdb
```

Auto-design:

```bash
python scripts/run_mars_autodesign.py design --pdb inputs/t4l_171l/171L.pdb
```

Optional evidence hooks / 可选证据输入：

```bash
python scripts/run_mars_autodesign.py design \
  --pdb inputs/cld_3q09/3Q09.pdb \
  --homologs-fasta ../designs/asr_cld_prb/cld_asr_input.fasta \
  --asr-fasta ../designs/asr_cld_prb/run3/cld_ancestor_recommended_panel.fasta
```

---

## 6. Raw-PDB Auto-Design Defaults / 任意 PDB 默认策略

**EN**  
The raw-PDB entrypoint uses a bounded full-protein exploration policy.

- whole chain is screened
- ligand-adjacent / disulfide / missing-backbone / nonstandard residues are protected
- top `24` positions are reported
- top `12` positions are actually used for bounded design
- mutation burden is kept in a small engineering window
- local proposal is always enabled
- `ProteinMPNN` / `ESM-IF` / neural branches degrade based on runtime availability

**中文**  
任意 PDB 模式不是无限制乱搜，而是“有边界的全蛋白探索”。

- 全链先筛一遍
- 靠近配体、二硫键、缺骨架、非标准聚合残基默认保护
- 报告前 `24` 个候选设计位点
- 实际设计只用前 `12` 个位点
- 默认突变负担限制在较小工程窗口内
- local proposal 永远启用
- `ProteinMPNN` / `ESM-IF` / neural 分支会根据 runtime 能力自动降级

Important limitation / 重要限制：

- this mode does **not** guarantee preservation of unknown biological function
- 这个模式不保证任意未知蛋白的功能保持

---

## 7. Algorithm Flow / 算法流程

`MARS-FIELD` is organized around a shared residue-field controller:

1. Input / 输入
   - target structure
   - target sequence
   - design / protected masks
   - optional homolog / ASR / family evidence
2. Evidence encoders / 证据编码
   - geometric encoder
   - phylo-sequence encoder
   - ancestral lineage encoder
   - retrieval memory encoder
   - environment branch
3. Residue field / 残基场
   - shared residue-wise decision field
4. Pairwise energy / 成对能量
   - site coupling / compatibility
5. Decoder / 解码器
   - constrained sequence search
6. Selector / 选择器
   - calibration and safety gating
7. Outputs / 输出
   - ranked candidates
   - shortlist FASTA
   - field / tensor exports
   - benchmark summaries
   - figure and structure assets

---

## 8. Planned Wet-Lab Gap Closure / 准备要做的实验

这一节只回答一件事：

- 现在实验上还缺什么
- 每组实验是为了验证什么

### 8.1 `Cld` main line / `Cld` 主线

#### Experiment 1: expression + baseline function / 表达 + 基础功能

Targets / 对象：

- `Q47CX0`
- `Node007`
- `Node012`
- `Node006`

Why / 目的：

- validate whether ancestral scaffolds are expressible
- validate heme loading
- validate baseline `chlorite` activity

Readouts / readout：

- soluble expression
- heme loading
- baseline activity

Minimum control / 最小对照：

- `Q47CX0`

Success / 成功标准：

- at least one ancestor passes:
  - soluble expression `>= 0.30 x WT`
  - baseline activity `>= 0.50 x WT`

#### Experiment 2: oxidative challenge / 氧化挑战

Why / 目的：

- test whether the line is truly moving beyond ordinary stability engineering

Readout：

- `H2O2` challenge residual activity

Success：

- at least one ancestor retains higher post-oxidation activity than WT

#### Experiment 3: low-temperature function / 低温功能

Why / 目的：

- validate low-temperature resilience as part of the Mars-relevant story

Readout：

- `4 C`
- `10 C`
- matched standard-condition control

Success：

- at least one ancestor retains better relative low-temperature activity than WT

#### Experiment 4: freeze-thaw / dry-rehydration / 冻融与干燥复水

Why / 目的：

- validate physical-stress resilience, not just oxidation

Readout：

- `FT03`
- optional `LYO01`

Success：

- at least one ancestor beats WT after freeze-thaw or dry-rehydration

#### Experiment 5: conservative rational mutation validation / 保守理性改造验证

Suggested first set / 建议第一组：

- `Node007`
- `Node007 + W67Y`
- `Node007 + H271Q`
- `Node007 + W67Y + H271Q`

Why / 目的：

- validate that the ancestral scaffold is a better carrier for oxidation-aware redesign

Success：

- at least one small mutation set improves post-oxidation activity without strongly harming baseline activity

### 8.2 `AresG` protection-module line / `AresG` 保护模块线

Targets / 对象：

- `AresG2-01_shortlink_shorttail`
- `AresG2-02_midlink_balanced`
- `AresG2-03_longlink_longtail`

#### Experiment 1: expression / solubility / stability screen / 表达、可溶性、稳定性筛选

Why / 目的：

- determine which version is physically buildable and worth carrying forward

Controls / 对照：

- `AresC-00_core_only`

Success：

- narrow to one primary version and one retained comparison version

#### Experiment 2: cargo protection assay / cargo 保护实验

Why / 目的：

- prove that `AresG` is a real protection module, not just a design concept

Scenarios / 场景：

- freeze-dry recovery
- osmotic/dehydration stress
- low-temperature cargo retention

Controls / 对照：

- cargo only
- `AresC core-only`
- traditional excipient system

Success：

- at least one `AresG2` version improves cargo retention in at least one stress regime

### 8.3 `DrwH / AresW` cargo-cap line / `DrwH / AresW` 货物保护帽路线

Targets / 对象：

- `DrwHAnc-01_local_near_Node046`
- `DrwHAnc-02_local_mid_Node054`
- `DrwHAnc-03_deep_ingroup_Node068`

#### Experiment 1: ancestor-domain screen / 祖先域筛选

Why / 目的：

- determine which ancestor behaves most like a compact stable protection cap

Readouts：

- expression
- solubility
- compactness / fold trend

Control：

- native `Q9RUL2`

Success：

- narrow to one or two cap candidates

#### Experiment 2: `AresW` fusion validation / `AresW` 拼接体验证

Targets：

- `AresW-03_C_cap_anc_local`
- `AresW-04_C_cap_anc_mid`
- `AresW-05_C_cap_anc_deep`

Why / 目的：

- determine whether the cap can be attached without collapsing the core

Control：

- `AresW-00_core_only`

Success：

- at least one fusion survives expression / solubility screening without obvious core failure

#### Experiment 3: cargo-cap protection assay / cargo-cap 保护实验

Why / 目的：

- prove that `DrwH` is a real cargo-protection module

Scenarios：

- oxidative stress
- dry-rehydration stress

Controls：

- cargo only
- core only
- cap only

Success：

- at least one cargo-cap condition outperforms cargo alone

### 8.4 Recommended experimental order / 推荐实验顺序

If resources are limited, use this order:

1. `Cld` expression + baseline activity
2. `Cld` oxidative challenge
3. `Cld` low-temperature / freeze-thaw / dry-rehydration
4. `Cld` conservative mutation validation
5. `AresG2` expression narrowing
6. `AresG2` cargo protection
7. `DrwH` ancestor-domain narrowing
8. `AresW` fusion validation
9. `DrwH / AresW` cargo-cap protection

---

## 9. Outputs / 输出内容

### Config-driven pipeline outputs / 配置驱动输出

- `structure_features.csv`
- `feature_summary.json`
- `profile_summary.json`
- `position_fields.json`
- `pairwise_energy_tensor.json`
- `combined_ranked_candidates.csv`
- `shortlist_top.fasta`
- `pipeline_summary.md`

### Raw-PDB outputs / 任意 PDB 输出

- generated config snapshot
- analysis summary JSON / Markdown
- ranked design positions CSV
- field / tensor exports
- ranked candidates CSV
- shortlist FASTA

---

## 10. Limitations / 限制

- not fully joint end-to-end training
- not guaranteed function preservation for arbitrary proteins
- optional external components still matter:
  - `ProteinMPNN`
  - `ESM-IF`
  - `MAFFT`
  - `IQ-TREE`
  - AF3 / external structure prediction services

In one sentence / 一句话结论：

`MARS-FIELD` 已经是可交付的研究原型平台，但火星生物学主线还需要靠 `Cld`、`AresG`、`DrwH` 的实验闭环来真正完成。
