# Experiment log

每次保留命令、代码版本、随机种子、环境和结论。不要只保存最好的一次结果。

## E000：脚本专家拾放基线

- 日期：2026-10-14
- 目的：验证环境、专家策略、安全层和评测链路
- 命令：`uv run sim2skill evaluate --episodes 20 --seed 0`
- 环境：macOS 14.7.8，Intel i5 4 核，8GB RAM，CPU

结果：

- 成功率：100%（20/20）
- 平均步数：632.75
- 平均终点误差：0.0366 m
- 安全干预次数：0

### 结论

固定种子范围内基线可以稳定生成示范。曾在 700 步上限下出现 2/20 个 episode
到达目标附近但来不及完成释放，因此将任务上限修正为 800 步；这一改动修复的是任务
时限配置，不是策略能力。

## E001：表面清洁，名义轨迹 vs 真值位姿

- 日期：2026-10-14
- 目的：量化「检测位姿 + 固定擦拭路径」在定位误差下的退化，对应 clean-robot 中
  `two_three_base` 后执行固定清洁轨迹的模式
- 命令：
  - `uv run sim2skill evaluate --task wipe --policy oracle --episodes 8 --seed 0`
  - `uv run sim2skill evaluate --task wipe --policy nominal --pose-noise 0.04 --episodes 8 --seed 0`
- 环境：同上，MuJoCo 3.3.7 CPU

| 策略 | 位姿噪声 | 成功率 | 覆盖率 | 接触丢失率 | 过力率 |
|------|----------|--------|--------|------------|--------|
| oracle | 0 cm | 100% | 0.918 | 0.090 | 0.000 |
| nominal | 4 cm | 0% | 0.620 | 0.435 | 0.013 |

失败分类（nominal, 4 cm）：`low_coverage+contact_loss` 4，`contact_loss` 2，`low_coverage` 2。

### 结论

在没有定位误差时，参数化之字擦拭可以稳定覆盖表面。给名义轨迹 4 cm 的目标位姿噪声后，
成功率从 100% 降到 0%，覆盖率下降约 30 个百分点。这直接说明：原工程里「检测一个参考点，
再走固定路径」的瓶颈在感知误差，而不只是缺一个更大的网络。

## E002：有界残差 MLP 尚未超过名义轨迹

- 日期：2026-10-14
- 目的：打通残差数据、训练和对照评测，而不是先追求刷指标
- 命令：
  - `uv run sim2skill train-residual --episodes 12 --seed 0`
  - `uv run sim2skill compare --episodes 8 --train-episodes 0 --residual artifacts/residual.npz`

结果摘要（8 episode）：

| 策略 | 噪声 | 成功率 | 覆盖率 | 安全干预 |
|------|------|--------|--------|----------|
| nominal | 0 | 100% | 0.918 | 0 |
| residual MLP | 0 | 0% | 0.404 | 225 |
| nominal | 4 cm | 0% | 0.620 | 0 |
| residual MLP | 4 cm | 0% | 0.411 | 201 |

训练损失从 0.098 降到 0.016，说明拟合闭环是通的。部署时覆盖率反而下降，主要原因是：
训练走的是「有界 oracle 修正后的轨迹」，测试时残差加在未修正的名义轨迹上，分布不一致；
12 条轨迹也不足以学到稳定的接触修正。

### 结论

残差训练流水线可以展示，但不能把当前 MLP 说成已经提升了清洁性能。下一步应先用结构化
接触残差做上界，再用更多 episode 或 DAgger 式在线标注缩小分布偏移。

## E003：结构化接触残差

- 日期：2026-10-14
- 命令：`uv run sim2skill evaluate --task wipe --policy structured --pose-noise 0.04 --episodes 8 --seed 0`

在 4 cm 噪声下，覆盖率 0.666 / 成功率 12.5%，相对名义轨迹的 0.620 / 0% 有小幅提升，
但安全层干预次数明显增加。它证明「接触力 + 局部污渍方向」是可用来纠偏的信号，
还不是一个已经调好的控制器。
