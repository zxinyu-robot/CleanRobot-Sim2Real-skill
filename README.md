# sim2skill

一个面向具身操作岗位的 **simulation-first、real-ready** 练习仓库。它承接既有复合清洁机器人项目中的真实系统集成经验，探索如何把硬编码的工具操作与清洁轨迹逐步升级为可学习、可评测的自适应操作能力。

目标不是在当前没有硬件的情况下包装“已完成 Sim-to-Real”，而是持续提供可运行、可测量、可解释的工作证据。

当前版本包含两个可复现任务：

1. **工具拾放**：对应原工程的取放工具。
2. **表面清洁**：对应原工程「检测参考点 + 固定擦拭路径」，并补上覆盖率、接触丢失和过力评测。

同时覆盖：

- 随机化任务环境与确定性复现
- 有限状态机 / 参数化名义轨迹
- 动作限速、工作空间约束和安全干预统计
- 示范采集、episode 级 train/val/test 划分、失败分类
- 有界残差策略接口（结构化修正与 NumPy MLP）

## 为什么这样选

既有工程已经涉及 ROS1、xArm、RealSense、目标检测、TF 坐标变换、移动底盘和清洁执行器，但任务流程和机械臂轨迹主要依赖固定规则与点位。

当前开发机是 Intel i5 / 8GB 的 macOS，适合原生 MuJoCo CPU 仿真，不适合本地训练视觉 Transformer，也无法重新验证真实机器人部署。因此第一阶段优先打通学习侧的工程闭环；视觉模仿学习训练预留给后续 Linux/GPU 环境，真机复测则明确列为未来验证项。

这比从双臂 ACT 或 ROS2 大工程起步更容易形成可信的提交记录和实验结果。

## 快速开始

要求 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev
uv run sim2skill smoke
uv run sim2skill evaluate --episodes 20
uv run sim2skill evaluate --task wipe --policy oracle --episodes 8
uv run sim2skill evaluate --task wipe --policy nominal --pose-noise 0.04 --episodes 8
uv run sim2skill collect --episodes 20 --output artifacts/demos.npz
```

macOS 打开 MuJoCo 可视化窗口：

```bash
uv run mjpython -m sim2skill.cli demo
```

运行质量检查：

```bash
uv run ruff check .
uv run pytest
```

## 系统边界

当前系统由可替换模块组成：

1. `PickPlaceEnv` / `WipeEnv`：仿真、观测、动作和成功条件。
2. `ScriptedExpert` / `WipeExpert`：透明的任务级基线，也用于生成示范。
3. `ActionShield`：独立于策略的安全约束。
4. `rollout`：统一数据采集和评测，避免训练与评测口径漂移。
5. `ResidualPolicy` / `StructuredResidual`：只预测有界修正，不替换名义轨迹。

学习策略只需要实现“观测到动作”的接口，就能与脚本基线使用相同的安全层和评测流程。

## 当前进度

- [x] 可复现的 MuJoCo 拾放环境
- [x] 脚本专家与安全执行层
- [x] 无界面评测和示范数据采集
- [x] 表面清洁任务与覆盖率 / 接触评测
- [x] episode 级数据划分和失败分类
- [x] 有界残差训练闭环（尚未超过名义轨迹）
- [ ] 残差或行为克隆在固定测试集上形成正向对照
- [ ] ROS2 / 真机适配器

详细计划见 [ROADMAP](docs/ROADMAP.md)，新旧项目的能力衔接见 [PROJECT_BRIDGE](docs/PROJECT_BRIDGE.md)，技术取舍见 [ADR-001](docs/adr/001-scope-for-mac.md)。


## 已知限制

- 当前为简化的笛卡尔吸盘机构，不代表真实机械臂动力学。
- 观测以低维状态为主，尚未验证视觉策略。
- 没有硬件就不能声称完成真机部署或 Sim-to-Real。
- 脚本专家是数据来源和工程基线，不是学习算法成果。

## License

[MIT](LICENSE)
