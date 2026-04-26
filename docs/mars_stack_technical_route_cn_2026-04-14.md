# MarsStack 当前技术路线说明

## 1. 项目现在的正确定义

`MarsStack` 现在不应该被定义成“改了一版 ProteinMPNN”。

更准确的定义是：

**一个面向极端环境耐受改造的通用蛋白工程框架。**

它的核心不是单一生成器，而是三层组合：

- `proposal generators`：从结构出发生成候选突变
- `Mars objective`：按火星/极端环境风险函数重排候选
- `evolution prior`：用同源信息限制不合理替换

这个表述比“改 MPNN”更适合后续投稿，因为它天然支持：

- 多生成器并列比较
- 多蛋白跨家族泛化
- retrospective benchmark + prospective validation

## 2. 当前已经落地的算法主线

当前主干流程已经是：

`Structure features -> candidate generators -> MarsScore rerank -> benchmark summary`

其中已经真正接好的模块包括：

- `baseline_mpnn`
- `mars_mpnn`
- `local_proposal`
- `profile prior`
- `overall winner / best learned winner` 双层汇报

接口已经预留但还在打通的模块：

- `esm_if`
- 更强的 `MSA/evolution prior`
- 后续可能的 `ASR / family-conditioned prior`

## 3. 为什么现在的主瓶颈不是打分，而是生成器

六蛋白 benchmark 的现象已经很明确：

- `MarsScore` 能把搜索方向往对的化学答案上推
- 但如果候选池里一开始就没有这些答案，rerank 也救不回来

这也是为什么目前经常出现：

- `local_proposal` 胜过 learned branch
- manual / chemistry-aware control 给出更合理突变

这不是坏消息，反而说明：

- `objective` 是有信息量的
- 真实短板在 `proposal coverage`

所以接下来最重要的事不是继续堆规则，而是补一个更强的 learned generator。

## 4. 为什么下一步优先接 ESM-IF

`ESM-IF1` 适合作为下一步，原因有三点：

- 它本身就是成熟的 inverse folding 模型，适合作为第二 learned branch
- 它和 `ProteinMPNN` 的偏好不完全一样，能提高候选多样性
- 论文上更容易形成“multi-generator + shared Mars objective”的方法叙事

因此下一版方法故事应当是：

**MarsStack = multiple structure-conditioned generators + Mars objective + evolution prior**

而不是：

**MarsStack = modified ProteinMPNN**

## 5. 当前工程状态

已经完成的关键工程包括：

- `TEM-1 / 1BTL` 的 gap-aware template mapping 修复
- `sfGFP / 2B3P` 的非标准残基预处理
- 六蛋白 benchmark 跑通
- 六蛋白 homolog/profile prior 接入
- family-level summary 输出
- `ESM-IF` 单链无 `biotite` 版本 runner 已打通
- `ESM-IF1` 大模型 checkpoint 已落到仓库本地缓存
- `CALB / 1LBT` 已完成一次真实 `ESM-IF -> Mars rerank` 测试

对应主文件：

- `scripts/run_mars_pipeline.py`
- `scripts/run_mars_benchmark.py`
- `scripts/run_esm_if_generator.py`
- `docs/mars_stack_technical_report_v1.md`

## 6. 当前环境结论

截至 2026-04-14，本地环境的结论是：

- `torch` 已可用
- `torch_geometric` 已安装成功
- `gemmi` 已可用
- `biotite` 在 Python 3.14 + Windows 下因为 `biotraj` wheel/编译问题失败

但这里有一个重要判断：

**当前单链 ESM-IF 跑法已经不再把 biotite 作为硬依赖。**

也就是说，现阶段不用继续把时间花在强装 `biotite` 上。

现在真正剩下的阻塞是：

- 把 `ESM-IF` 从单 target 验证扩到多 target benchmark
- 把 `ESM-IF` branch 系统性接入 6-10 个蛋白
- 继续做 family split 和 baseline 对比

默认缓存位置已经约定为：

- `mars_stack/.cache/esm_if1_gvp4_t16_142M_UR50.pt`

## 7. 论文主线应该怎么讲

如果按可投稿的方法逻辑，主线建议固定成下面这条：

1. 定义一个极端环境蛋白工程的统一目标函数
2. 用多个结构条件生成器产生候选
3. 用 chemistry + evolution 约束做统一重排
4. 在多蛋白 benchmark 上证明优于单一 generator 或纯 heuristic
5. 用 held-out family split 证明不是只记住几个蛋白

然后把结果拆成两层：

- `overall winner`
- `best learned winner`

这样既保留工程上最强设计，也不会把方法信号淹掉。

## 8. 未来两周最小闭环

最小可发表的下一阶段闭环，我建议就按这个顺序推进：

1. 先打通 `ESM-IF` learned branch
2. 把 benchmark 从 6 个蛋白扩到 8 到 10 个
3. 引入 held-out family split 统计
4. 做 `ProteinMPNN vs MarsStack vs heuristic` 对比
5. 做 `oxidation / surface / evolution` ablation

在这个闭环成型前，不建议再把主要精力放在：

- 继续魔改规则打分
- 过早重训练完整大模型

## 9. 当前最现实的执行判断

如果只是把现有内容写成论文草稿，已经够写方法草稿和组会汇报。

如果目标是顶会主会，现在还差的是：

- 更强 learned generator 的实证结果
- 更大的 benchmark 面板
- family split 泛化
- 统计显著性与 baseline 矩阵

所以当前阶段最准确的判断是：

**不是想法阶段了，已经进入方法工程化阶段；但距离顶会主会，还差一轮真正的 learned-branch strengthening 和系统 benchmark。**
