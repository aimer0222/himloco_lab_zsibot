# HimLoco Lab - ZsiBot

## Project Overview

This project implements a quadruped robot gait controller based on the **HimLoco** algorithm, using **Isaac Lab** for reinforcement learning training, with support for **Mujoco Sim2Sim** validation and **Sim2Real** deployment.

Adapted and optimized for the **ZsiBot (ZSL1)** robot model.

## Project Structure

```
himloco_lab/
├── scripts/                              # Scripts
│   ├── himloco_rsl_rl/
│   │   ├── train.py                      # Training script
│   │   ├── play.py                       # Batch inference and policy export
│   │   ├── play_interactive.py           # Interactive control script
│   │   └── cli_args.py
│   ├── list_envs.py                      # List available environments
│   ├── zero_agent.py
│   └── random_agent.py
│
├── source/himloco_lab/                   # Main source code
│   └── himloco_lab/
│       ├── tasks/
│       │   └── locomotion/
│       │       ├── mdp/                  # Observations, actions, rewards, terminations
│       │       └── robots/zsibot/        # ZsiBot training configuration
│       │
│       ├── rsl_rl/                       # HimLoco algorithm implementation
│       │   ├── config/                  # Algorithm configuration
│       │   ├── modules/                  # HIMActorCritic, HIMEstimator networks
│       │   ├── algorithms/               # HIMOnPolicyRunner training logic
│       │   ├── wrappers/                 # Environment adapters
│       │   └── env/                      # VecEnv interface
│       │
│       ├── assets/                       # Robot assets
│       │   └── zsibot.py                 # ZsiBot URDF configuration
│       │
│       ├── robot/zsl1_description/       # ZSL1 robot description
│       │   ├── urdf/                     # URDF files
│       │   └── meshes/                   # STL mesh files
│       │
│       └── utils/
│           └── export_policy.py           # JIT and ONNX export utilities
│
├── zsibot_mujoco/                        # Mujoco Sim2Sim
│   ├── robot/zsl1_description/           # Mujoco robot models
│   │   ├── xml/                          # Mujoco XML scene files
│   │   ├── urdf/                         # Mujoco URDF files
│   │   └── meshes/                       # Mesh files
│   ├── mujoco_sim2sim/                   # Sim2Sim inference scripts
│   │   ├── sim2sim_zsibot.py             # Position control
│   │   ├── sim2sim_zsibot_torque.py     # Torque control
│   │   └── torque_controller.py           # Torque controller
│   └── deploy/
│       └── lowlevel_state_machine_himloco.py  # Real robot deployment controller
│
└── deploy/                               # Robot deployment code
    └── robots/go2/                       # Go2 controller (to be adapted)
```

## Algorithm

[HimLoco](https://github.com/InternRobotics/HIMLoco/blob/main/projects/himloco/README.md)

![HimLoco](https://github.com/InternRobotics/HIMLoco/blob/main/assets/overview.jpeg)

HimLoco uses a dual-network architecture with actor-critic framework and state estimator, enabling stable gait control on quadruped robots.

## Installation

### 1. Install Isaac Lab

Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install Isaac Lab.
The pip installation method is recommended.

### 2. Clone This Repository

```bash
git clone https://github.com/your-username/himloco_lab.git
cd himloco_lab
```

### 3. Install HimLoco Lab

```bash
python -m pip install -e source/himloco_lab
```

### 4. Dependencies

This project uses built-in ZSL1 robot description files, no need to download `unitree_ros` or `unitree_sdk2`:
- URDF files are located at `source/himloco_lab/himloco_lab/robot/zsl1_description/`
- Configuration has been updated to use relative paths

## Quick Start

### Train Policy

Train HimLoco policy on ZsiBot (ZSL1):

```bash
python scripts/himloco_rsl_rl/train.py --task ZsiBot-Velocity --headless
```

### Inference and Playback

Run trained policy for batch inference:

```bash
python scripts/himloco_rsl_rl/play.py --task ZsiBot-Velocity-Play
```

### Interactive Control

Control the robot in real-time using keyboard in Isaac Lab:

```bash
python scripts/himloco_rsl_rl/play_interactive.py --task ZsiBot-Velocity-Play

# Keyboard controls:
#   Numpad 8 / ↑   : Move forward
#   Numpad 2 / ↓   : Move backward
#   Numpad 4 / ←   : Strafe left
#   Numpad 6 / →   : Strafe right
#   Numpad 7 / Z   : Rotate counterclockwise
#   Numpad 9 / X   : Rotate clockwise
```

### Export Model

Export trained model for deployment:

```bash
python scripts/himloco_rsl_rl/play.py --task ZsiBot-Velocity-Play
```

Exported files are in the `logs/` directory:
- `policy.pt` - TorchScript JIT model (single file deployment)
- `encoder.onnx` & `policy.onnx` - ONNX format (separated models)

## Sim2Sim Validation

Validate trained policies in Mujoco.

### Install Mujoco

```bash
pip install mujoco mujoco-viewer onnxruntime numpy pygame
```

### Run Sim2Sim

```bash
# Position control mode
cd zsibot_mujoco/mujoco_sim2sim
python sim2sim_zsibot.py

# Torque control mode
python sim2sim_zsibot_torque.py
```

### Configuration

Mujoco models are located at `zsibot_mujoco/robot/zsl1_description/xml/`:
- `zsibot_scene.xml` - Basic scene
- `zsibot_scene_torque.xml` - Torque control scene
- `zsibot_floating_stairs_gen.xml` - Scene with stairs

## Sim2Real Deployment

### Environment Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install

# Compile robot controller
cd zsibot_mujoco/deploy
mkdir build && cd build
cmake .. && make
```

### Start Real Robot Control

```bash
cd zsibot_mujoco/deploy/build
./zsibot_ctrl --network eth0
```

## TODO List

- [x] Train ZsiBot policy in Isaac Lab
- [x] Implement Mujoco Sim2Sim validation
- [x] Export torque control policy
- [ ] Adapt ZsiBot real robot controller
- [ ] Migrate to latest rsl_rl version

## References

- [HimLoco](https://github.com/RoboLoco/HimLoco)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
- [Mujoco](https://mujoco.org/)
