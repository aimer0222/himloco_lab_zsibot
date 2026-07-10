import time
import collections
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import pygame
import torch

# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path("/home/ws/nvidia/himloco_lab/zsibot_mujoco/robot/zsl1_description/xml/zsibot_floating_stairs_gen.xml")
# m = mujoco.MjModel.from_xml_path("/home/ws/nvidia/himloco_lab/zsibot_mujoco/robot/zsl1_description/xml/zsibot_scene.xml")
d = mujoco.MjData(m)

# ONNX 模型路径（由 play.py 从 checkpoint 导出）
# EXPORT_DIR = Path("/home/ws/nvidia/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-05-18_00-26-02/exported")
EXPORT_DIR = Path("/home/ws/nvidia/himloco_lab/logs/himloco_rsl_rl/zsibot_rough/2026-06-09_11-32-44/exported")
ENCODER_PATH = EXPORT_DIR / "encoder.onnx"
POLICY_PATH = EXPORT_DIR / "policy.onnx"

# deploy.yaml 对应配置
HISTORY_LENGTH = 6
NUM_ONE_STEP_OBS = 45
ENCODER_OUTPUT_DIM = 19
POLICY_INPUT_DIM = 64

# deploy.yaml 中的 joint_ids_map：将按腿排列的 MuJoCo actuator 顺序
# [FAR_ABAD, FAR_HIP, FAR_KNEE, FBL_ABAD, FBL_HIP, FBL_KNEE, RAR_ABAD, RAR_HIP, RAR_KNEE, RBL_ABAD, RBL_HIP, RBL_KNEE]
# 映射到策略使用的关节类型分组顺序
# [FAR_ABAD, FBL_ABAD, RAR_ABAD, RBL_ABAD, FAR_HIP, FBL_HIP, RAR_HIP, RBL_HIP, FAR_KNEE, FBL_KNEE, RAR_KNEE, RBL_KNEE]
JOINT_IDS_MAP = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int32)
CTRL_REORDER = np.argsort(JOINT_IDS_MAP)

# 观测中的关节顺序：与 deploy.yaml/default_joint_pos 保持一致
joint_names = [
    "FAR_ABAD_JOINT", "FBL_ABAD_JOINT", "RAR_ABAD_JOINT", "RBL_ABAD_JOINT",
    "FAR_HIP_JOINT", "FBL_HIP_JOINT", "RAR_HIP_JOINT", "RBL_HIP_JOINT",
    "FAR_KNEE_JOINT", "FBL_KNEE_JOINT", "RAR_KNEE_JOINT", "RBL_KNEE_JOINT",
]

DEFAULT_DOF_POS = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.8, 0.8, 0.8, 0.8,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)
ACTION_SCALE = np.array([0.25] * 12, dtype=np.float32)
COMMAND_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
BASE_ANG_VEL_SCALE = np.array([0.25, 0.25, 0.25], dtype=np.float32)
JOINT_VEL_SCALE = np.array([0.05] * 12, dtype=np.float32)

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]


def get_sensor_data(sensor_name):
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = m.sensor_adr[sensor_id]
    dim = m.sensor_dim[sensor_id]
    return d.sensordata[start_idx : start_idx + dim].astype(np.float32)


def world2self(quat, v):
    q_w = quat[0]
    q_vec = quat[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return (a - b + c).astype(np.float32)


def fmt(arr):
    arr = np.array(arr).reshape(-1)
    return "[" + ", ".join(f"{float(v):7.4f}" for v in arr) + "]"


def build_single_obs(last_action, commands, verbose=False):
    """构造单步 45 维 obs，顺序严格对齐 play.py：
    [velocity_commands, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action]
    """
    base_quat = get_sensor_data("imu_quat")
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = world2self(base_quat, gravity)
    imu_gyro = get_sensor_data("imu_gyro")

    dof_pos = np.array([get_sensor_data(name + "_POS")[0] for name in joint_names], dtype=np.float32)
    dof_vel = np.array([get_sensor_data(name + "_VEL")[0] for name in joint_names], dtype=np.float32)
    cmds = np.array(commands, dtype=np.float32)

    single_obs = np.clip(
        np.concatenate(
            [
                cmds * COMMAND_SCALE,                # 0:3 velocity_commands
                imu_gyro * BASE_ANG_VEL_SCALE,       # 3:6 base_ang_vel
                projected_gravity,                   # 6:9 projected_gravity
                (dof_pos - DEFAULT_DOF_POS),         # 9:21 joint_pos_rel
                dof_vel * JOINT_VEL_SCALE,           # 21:33 joint_vel_rel
                last_action,                         # 33:45 last_action
            ],
            axis=-1,
        ),
        -100.0,
        100.0,
    ).astype(np.float32)

    if verbose:
        print("commands:          ", fmt(cmds))
        print("imu_gyro:          ", fmt(imu_gyro))
        print("projected_gravity: ", fmt(projected_gravity))
        print("dof_pos:           ", fmt(dof_pos))
        print("dof_vel:           ", fmt(dof_vel))
        print("last_action:       ", fmt(last_action))
        print("single_obs[0:45]:  ", fmt(single_obs))
        print("=" * 60)

    return single_obs


def build_history_obs(obs_history, current_obs):
    """构造 270 维历史观测：[obs_t, obs_t-1, ..., obs_t-5]。"""
    obs_history.appendleft(current_obs)
    if len(obs_history) < HISTORY_LENGTH:
        padded = list(obs_history) + [np.zeros(NUM_ONE_STEP_OBS, dtype=np.float32) for _ in range(HISTORY_LENGTH - len(obs_history))]
    else:
        padded = list(obs_history)
    history_obs = np.concatenate(padded, axis=-1).astype(np.float32)
    return history_obs


def update_commands_pygame(commands, keys):
    step = 0.05
    if keys[pygame.K_KP8] or keys[pygame.K_8]:
        commands[0] += step
    if keys[pygame.K_KP2] or keys[pygame.K_2]:
        commands[0] -= step
    if keys[pygame.K_KP6] or keys[pygame.K_6]:
        commands[1] -= step
    if keys[pygame.K_KP4] or keys[pygame.K_4]:
        commands[1] += step
    if keys[pygame.K_KP7] or keys[pygame.K_7]:
        commands[2] += step
    if keys[pygame.K_KP9] or keys[pygame.K_9]:
        commands[2] -= step

    for i in range(3):
        if not any([
            (i == 0 and (keys[pygame.K_KP8] or keys[pygame.K_8] or keys[pygame.K_KP2] or keys[pygame.K_2])),
            (i == 1 and (keys[pygame.K_KP6] or keys[pygame.K_6] or keys[pygame.K_KP4] or keys[pygame.K_4])),
            (i == 2 and (keys[pygame.K_KP7] or keys[pygame.K_7] or keys[pygame.K_KP9] or keys[pygame.K_9])),
        ]):
            if abs(commands[i]) < step:
                commands[i] = 0.0
            else:
                commands[i] -= step * (1 if commands[i] > 0 else -1)

    commands[0] = max(min(commands[0], 0.8), -0.8)
    commands[1] = max(min(commands[1], 1.0), -1.0)
    commands[2] = max(min(commands[2], 1.0), -1.0)
    return commands


def create_sessions():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoder ONNX not found: {ENCODER_PATH}")
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Policy ONNX not found: {POLICY_PATH}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    encoder_session = ort.InferenceSession(str(ENCODER_PATH), sess_options, providers=providers)
    policy_session = ort.InferenceSession(str(POLICY_PATH), sess_options, providers=providers)

    enc_in = encoder_session.get_inputs()[0]
    enc_out = encoder_session.get_outputs()[0]
    pol_in = policy_session.get_inputs()[0]
    pol_out = policy_session.get_outputs()[0]

    print(f"[INFO] ONNX providers: {providers}")
    print(f"[INFO] Encoder input: {enc_in.name} {enc_in.shape}, output: {enc_out.name} {enc_out.shape}")
    print(f"[INFO] Policy input: {pol_in.name} {pol_in.shape}, output: {pol_out.name} {pol_out.shape}")

    return encoder_session, policy_session


def main():
    encoder_session, policy_session = create_sessions()

    actions = np.zeros(12, dtype=np.float32)
    commands = [0.0, 0.0, 0.0]
    obs_history = collections.deque(maxlen=HISTORY_LENGTH)

    pygame.init()
    pygame.display.set_mode((200, 200))
    pygame.display.set_caption("ZSIBot Keyboard Control (ONNX)")

    for _ in range(200):
        mujoco.mj_step(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()
            commands = update_commands_pygame(commands, keys)

            obs_single = build_single_obs(last_action=actions, commands=commands, verbose=False)
            obs_history_270 = build_history_obs(obs_history, obs_single).reshape(1, -1)

            encoder_output = encoder_session.run(
                [encoder_session.get_outputs()[0].name],
                {encoder_session.get_inputs()[0].name: obs_history_270},
            )[0]

            if encoder_output.shape[1] != ENCODER_OUTPUT_DIM:
                raise RuntimeError(f"Unexpected encoder output dim: {encoder_output.shape}")

            current_obs = obs_single.reshape(1, -1)
            policy_input = np.concatenate([current_obs, encoder_output], axis=1).astype(np.float32)

            if policy_input.shape[1] != POLICY_INPUT_DIM:
                raise RuntimeError(f"Unexpected policy input dim: {policy_input.shape}")

            actions = policy_session.run(
                [policy_session.get_outputs()[0].name],
                {policy_session.get_inputs()[0].name: policy_input},
            )[0][0].astype(np.float32)

            act_policy_order = np.clip(actions * ACTION_SCALE + DEFAULT_DOF_POS, -100.0, 100.0)
            act_mujoco_order = act_policy_order[CTRL_REORDER]
            for i in range(12):
                d.ctrl[i] = float(act_mujoco_order[i])

            step_start = time.time()
            for _ in range(4):
                mujoco.mj_step(m, d)
            viewer.sync()

            time_until_next_step = m.opt.timestep * 4 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()