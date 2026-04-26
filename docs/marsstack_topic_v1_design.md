# MarsStack Topic-Aware Scoring v1

## 1. 目标

在 `MarsStack v0` 的通用打分器之上，增加一个**专题算法层**，让不同任务对象使用不同的二级目标函数。

当前第一版专题对象：

1. `Cld`
2. `DrwH`
3. `AresG`

专题层的目标不是替代原有 `MarsScore v0`，而是在通用层之上追加：

- `topic_sequence`
- `topic_structure`
- `topic_evolution`

三类分数。

## 2. 总体结构

总分现在由两层组成：

### 2.1 通用层

已有项：

- `oxidation`
- `surface`
- `manual`
- `evolution`
- `burden`

这部分继续负责“极端环境蛋白硬化”的通用偏好。

### 2.2 专题层

新增项：

- `topic_sequence`
- `topic_structure`
- `topic_evolution`

这部分负责把“具体对象是什么蛋白、要承担什么角色”显式写进打分器。

## 3. 为什么需要专题层

如果只用通用层，系统能做“整体更耐受”的候选筛选，但还不能很好回答：

1. `Cld` 的 heme 口袋和功能网络该怎么保守地硬化？
2. `DrwH` 这种小型保护域如何维持紧凑 cargo-cap 特性？
3. `AresG` 这种保护模块如何保持低复杂度、弱装配趋势和不过度失控的平衡？

所以专题层的作用就是：

- 把对象差异显式编码进 scorer
- 让同一平台支持“功能酶”“保护域”“保护模块”三类对象

## 4. 第一版专题算法设计

## 4.1 `CldScore`

输入信号：

- 序列：功能壳层位点是否被激进破坏
- 结构：氧化防护位点、distal gate、proximal 网络、埋藏位点稳定性
- 进化：profile / ASR / family prior 在关键壳层位点上的支持度

第一版规则：

1. 对 `functional_shell_positions` 内的突变做额外约束。
2. 对 `oxidation_guard_positions` 上的安全替换给更强奖励。
3. 对 `distal_gate_positions` 和 `proximal_network_positions` 上的 `P/G` 等破坏性突变给强惩罚。
4. 对关键壳层位点叠加更强的 `ASR`/family/profile 支持项。

适合回答的问题：

- 哪些 `Cld` 祖先或小变体更像“保守硬化”而不是“破坏功能核心”。

## 4.2 `DrwHScore`

输入信号：

- 序列：全序列疏水/极性/电荷窗口
- 结构：埋藏位点核心稳定、表面保护性极性、表面氧化风险
- 进化：近祖先/深祖先与家族 profile 的支持度

第一版规则：

1. 维持小型紧凑保护域所需的疏水-极性平衡。
2. 惩罚埋藏位点被 `P/G/D/E/K/R` 之类残基破坏。
3. 奖励表面极性增强，惩罚表面氧化易感残基增加。
4. 对突变位点叠加 `ASR` 支持。

适合回答的问题：

- 哪一类 `DrwH` 祖先更适合作为 cargo-cap 模块继续推进。

## 4.3 `AresGScore`

输入信号：

- 序列：低复杂度比例、极性比例、疏水比例、电荷幅度
- 结构：表面可逆弱装配倾向、核心不塌的基本要求
- 进化：如果后续加入天然保护片段 panel，可叠加对应 prior

第一版规则：

1. 奖励落在目标窗口内的低复杂度比例。
2. 奖励适度极性与适度疏水并存，而不是单边极端。
3. 惩罚表面氧化易感残基与埋藏位点破坏。
4. 对有进化先验的构建支持 topic_evolution。

适合回答的问题：

- 哪些 `AresG` 版本更像“可控保护模块”，而不是简单变软或变乱。

## 5. 配置接口

新增顶层配置节：

```yaml
topic:
  enabled: true
  name: "cld"
  cld:
    functional_shell_positions: [152, 189, 190, 201, 204, 211, 217, 254, 261]
    oxidation_guard_positions: [189, 190, 261]
    distal_gate_positions: [217]
    proximal_network_positions: [204, 254]
    buried_sasa_max: 20.0
    profile_prior_scale: 0.35
    asr_prior_scale: 0.55
    family_prior_scale: 0.55
```

对应输出列：

- `score_topic_sequence`
- `score_topic_structure`
- `score_topic_evolution`

## 6. 当前边界

第一版专题算法还是**规则驱动**而不是 learned reranker。

优点：

- 可解释
- 便于写技术路线
- 便于后续消融

不足：

- 还没有 learned topic reranker
- 还没有 generator 层的专题感知
- 还没有把融合保护、配方保护和功能酶保护统一进一个多实体模型

## 7. 下一步建议

1. 给 `Cld` 写第一份专题配置并跑小型 retrospective。
2. 给 `DrwH` 写第一份专题配置并比较三层祖先。
3. 给 `AresG` 补第一版结构/序列混合 scoring 配置。
4. 在 `run_mars_benchmark.py` 中增加专题层消融：
   - `no_topic_sequence`
   - `no_topic_structure`
   - `no_topic_evolution`
