# HimLoco Lab - ZsiBot

## 项目概述

本项目是基于 **HimLoco** 算法的四足机器人步态控制器，代码参考 **HimLoco_lab**项目进行开发 ，使用 **Isaac Lab** 进行强化学习训练，并支持 **Mujoco Sim2Sim** 验证和 **Sim2Real** 真机部署。

针对 **ZsiBot (ZSL1)** 机器人模型进行了适配和优化。

## 项目结构

```
himloco_lab/
├── scripts/                              # 脚本文件
│   ├── himloco_rsl_rl/
│   │   ├── train.py                      # 训练脚本
│   │   ├── play.py                       # 批量推理和策略导出
│   │   ├── play_interactive.py           # 交互式控制脚本
│   │   └── cli_args.py
│   ├── list_envs.py                      # 列出可用环境
│   ├── zero_agent.py
│   └── random_agent.py
│
├── source/himloco_lab/                   # 主要源代码
│   └── himloco_lab/
│       ├── tasks/
│       │   └── locomotion/
│       │       ├── mdp/                  # 观测、动作、奖励、终止项
│       │       └── robots/zsibot/        # ZsiBot 训练配置
│       │
│       ├── rsl_rl/                       # HimLoco 算法实现
│       │   ├── config/                   # 算法配置
│       │   ├── modules/                  # HIMActorCritic, HIMEstimator 网络
│       │   ├── algorithms/               # HIMOnPolicyRunner 训练逻辑
│       │   ├── wrappers/                 # HimlocoVecEnvWrapper 环境适配
│       │   └── env/                      # VecEnv 接口
│       │
│       ├── assets/                       # 机器人资产
│       │   └── zsibot.py                 # ZsiBot URDF 配置
│       │
│       ├── robot/zsl1_description/       # ZSL1 机器人描述文件
│       │   ├── urdf/                     # URDF 文件
│       │   └── meshes/                   # STL 网格文件
│       │
│       └── utils/
│           └── export_policy.py          # JIT 和 ONNX 导出工具
│
└── zsibot_mujoco/                        # Mujoco Sim2Sim
    ├── robot/zsl1_description/          # Mujoco 机器人模型
    │   ├── xml/                          #Mujoco XML 场景文件
    │   ├── urdf/                         #Mujoco URDF 文件
    │   └── meshes/                       # 网格文件
    ├── mujoco_sim2sim/                   # Sim2Sim 推理脚本
    │   ├── sim2sim_zsibot.py             # 位置控制
    │   ├── sim2sim_zsibot_torque.py     # 力矩控制
    │   └── torque_controller.py           # 力矩控制器
    └── deploy/
        └── lowlevel_state_machine_himloco.py  # 真机部署控制器(不能直接运行)


```

## 算法原理

[HimLoco](https://github.com/InternRobotics/HIMLoco/blob/main/projects/himloco/README.md)

![HimLoco](https://github.com/InternRobotics/HIMLoco/blob/main/assets/overview.jpeg)

HimLoco 采用双网络架构设计，结合了 actor-critic 框架和状态估计器，能够在四足机器人上实现稳定的步态控制。

## 安装指南

### 1. 安装 Isaacsim 5.1.0 与 Isaac Lab 2.3.0

按照 [官方安装指南](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)安装Isaacsim；
按照 [官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) 安装 Isaac Lab。
推荐使用 pip 安装方式。

### 2. 克隆本仓库

```bash
git clone https://github.com/your-username/himloco_lab.git
cd himloco_lab
```

### 3. 安装 HimLoco Lab

```bash
python -m pip install -e source/himloco_lab
```

### 4. 依赖说明

本项目使用内置的 ZSL1 机器人描述文件；
- URDF 文件位于 `source/himloco_lab/himloco_lab/robot/zsl1_description/`

## 快速开始

### 训练策略

在 ZsiBot (ZSL1) 上训练 HimLoco 策略：

```bash
python scripts/himloco_rsl_rl/train.py --task ZsiBot-Velocity --headless
```

### 推理和播放

运行已训练的策略进行批量推理：

```bash
python scripts/himloco_rsl_rl/play.py --task ZsiBot-Velocity-Play
```

### 交互式控制

使用键盘在 Isaac Lab 中实时控制机器人：

```bash
python scripts/himloco_rsl_rl/play_interactive.py --task ZsiBot-Velocity-Play

# 键盘控制说明:
#   Numpad 8 / ↑   : 前进
#   Numpad 2 / ↓   : 后退
#   Numpad 4 / ←   : 向左平移
#   Numpad 6 / →   : 向右平移
#   Numpad 7 / Z   : 逆时针旋转
#   Numpad 9 / X   : 顺时针旋转
```

### 模型导出

将训练好的模型导出用于部署：

```bash
python scripts/himloco_rsl_rl/play.py --task ZsiBot-Velocity-Play
```

导出的文件位于 `logs/` 目录下：
- `policy.pt` - TorchScript JIT 模型（单文件部署）
- `encoder.onnx` & `policy.onnx` - ONNX 格式（分离模型）

## Sim2Sim 验证

在 Mujoco 中验证训练好的策略。

### 安装Mujoco

```bash
pip install mujoco mujoco-viewer onnxruntime numpy pygame
```

### 运行 Sim2Sim

```bash
# 位置控制模式
cd zsibot_mujoco/mujoco_sim2sim
python sim2sim_zsibot.py

# 力矩控制模式
python sim2sim_zsibot_torque.py
```

### 配置说明

Mujoco 模型位于 `zsibot_mujoco/robot/zsl1_description/xml/`：
- `zsibot_scene.xml` - 基础场景
- `zsibot_scene_torque.xml` - 力矩控制场景
- `zsibot_floating_stairs_gen.xml` - 带楼梯的场景

## Sim2Real 部署

详情见真机部署的README文档

## 参考资源

- [HimLoco](https://github.com/RoboLoco/HimLoco)
- [HimLoco_lab](https://github.com/IsaacZH/himloco_lab)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [Mujoco](https://mujoco.org/)
