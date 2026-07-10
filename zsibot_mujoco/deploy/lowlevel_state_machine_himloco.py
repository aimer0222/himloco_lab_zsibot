import os
import platform
import sys
import collections
import time
import threading
from enum import Enum

import numpy as np
import onnxruntime as ort

arch = platform.machine().replace('amd64', 'x86_64').replace('arm64', 'aarch64')
lib_path = os.path.abspath(f'{os.path.dirname(__file__)}/../lib/zsl-1/{arch}')
sys.path.insert(0, lib_path)
import mc_sdk_zsl_1_py


# 模型路径配置 - himloco ONNX 模型
script_dir = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(script_dir, "model/himloco/2026-06-09_11-32-44")
ENCODER_PATH = os.path.join(EXPORT_DIR, "encoder.onnx")
POLICY_PATH = os.path.join(EXPORT_DIR, "policy.onnx")

# himloco 模型配置
HISTORY_LENGTH = 6
NUM_ONE_STEP_OBS = 45
ENCODER_OUTPUT_DIM = 19
POLICY_INPUT_DIM = 64

# 观测中的关节顺序（与训练时保持一致，与电机 state 顺序一致）
# 顺序: FR, FL, RR, RL；每组: abad, hip, knee
JOINT_NAMES = [
    "FAR_ABAD_JOINT", "FBL_ABAD_JOINT", "RAR_ABAD_JOINT", "RBL_ABAD_JOINT",
    "FAR_HIP_JOINT", "FBL_HIP_JOINT", "RAR_HIP_JOINT", "RBL_HIP_JOINT",
    "FAR_KNEE_JOINT", "FBL_KNEE_JOINT", "RAR_KNEE_JOINT", "RBL_KNEE_JOINT",
]

# 默认关节位置
DEFAULT_DOF_POS = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.8, 0.8, 0.8, 0.8,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

ACTION_SCALE = np.array([0.25] * 12, dtype=np.float32)
COMMAND_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
BASE_ANG_VEL_SCALE = np.array([0.25, 0.25, 0.25], dtype=np.float32)
JOINT_VEL_SCALE = np.array([0.05] * 12, dtype=np.float32)

# ONNX providers
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]


class ControlState(Enum):
    """控制状态枚举"""
    PASSIVE = 0     # 被动/停止状态
    STAND_UP = 1   # 站立起身状态
    RL_CONTROL = 2 # RL策略控制状态


class MotorStateMachine:
    def __init__(self):
        # 状态机相关
        # 状态流程: PASSIVE -> STAND_UP -> RL_CONTROL -> PASSIVE
        self.current_state = ControlState.PASSIVE
        self.previous_state = ControlState.PASSIVE

        # STAND_UP 等待确认标志
        self.stand_up_waiting_for_confirm = False

        # 初始化电机接口
        self.motor_func = mc_sdk_zsl_1_py.LowLevel()
        self.motor_func.initRobot("192.168.123.18", 43988, "192.168.123.10")

        # 加载 ONNX 模型 - himloco encoder + policy
        print(f"[INFO] Loading ONNX models from {EXPORT_DIR}...")
        print(f"[INFO] ONNX providers: {ONNX_PROVIDERS}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        self.encoder_session = ort.InferenceSession(ENCODER_PATH, sess_options, providers=ONNX_PROVIDERS)
        self.policy_session = ort.InferenceSession(POLICY_PATH, sess_options, providers=ONNX_PROVIDERS)

        enc_in = self.encoder_session.get_inputs()[0]
        enc_out = self.encoder_session.get_outputs()[0]
        pol_in = self.policy_session.get_inputs()[0]
        pol_out = self.policy_session.get_outputs()[0]

        print(f"[INFO] Encoder input: {enc_in.name} {enc_in.shape}, output: {enc_out.name} {enc_out.shape}")
        print(f"[INFO] Policy input: {pol_in.name} {pol_in.shape}, output: {pol_out.name} {pol_out.shape}")

        # 观测历史缓冲区
        self.obs_history = collections.deque(maxlen=HISTORY_LENGTH)

        # 默认关节位置 (与训练时一致)
        self.num_dof = 12

        # 上一帧动作缓存
        self.last_actions = np.zeros(12, dtype=np.float32)

        # 命令 [vx, vy, wz]
        self.commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # PD控制参数
        self.kp = 20.0
        self.kd = 0.7

        # stand_up状态的子阶段
        self.stand_up_stage = 0  # 0: 空闲, 1: 阶段1 (高位准备), 2: 阶段2 (默认姿态)
        self.stand_up_progress = 0.0
        self.stand_up_duration = 2.0

        # 关节初始位置 (用于插值)
        self.init_q_abad = [0.0] * 4
        self.init_q_hip = [0.0] * 4
        self.init_q_knee = [0.0] * 4

        # 站立目标位置
        self.stand_up_targets = {
            'stage1': {'abad': 0.0, 'hip': 1.4, 'knee': -2.4},
            'stage2': {'abad': 0.0, 'hip': 0.8, 'knee': -1.5}
        }


        # 一次性触发事件队列（数字键 8/2/4/6/7/9 用于增量控制）
        self.key_events = []
        self.event_lock = threading.Lock()

        # 平滑归零标志：按下 0 后置 True，由 update_commands_keyboard 线性回归到 0 再清回 False
        self.zero_cmd_pending = False

        # 关节限位 (rad) - 来自 URDF
        self.JOINT_LIMITS = {
            'abad':  (-0.4887,  0.4887),
            'hip':   (-1.1519,  2.967),
            'knee':  (-2.723,  -0.602),
        }

        # 安全相关
        self.start_control = False

        # 打印频率控制 (5Hz = 每秒5次，500Hz / 5Hz = 100 帧/次)
        self.print_counter = 0
        self.print_interval = 100

        print(f"[StateMachine] Initialized with himloco ONNX model, current state: {self.current_state.name}")
    
    def calculate_projected_gravity(self, quat):
        """计算重力向量在机身坐标系下的投影"""
        q_w = quat[0]
        q_vec = np.array(quat[1:])
        v_vec = np.array([0.0, 0.0, -1.0])

        a = v_vec * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v_vec) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v_vec) * 2.0
        return (a - b + c).astype(np.float32)

    def build_single_obs(self, state):
        """构建单步 45 维 obs，与 himloco 训练时的顺序一致：
        [velocity_commands, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action]
        """
        # 获取 IMU 数据
        quat = self.motor_func.getQuaternion()
        gyro = self.motor_func.getBodyGyro()

        # 计算重力投影
        projected_gravity = self.calculate_projected_gravity(quat)

        # 构造关节位置和速度数组（顺序: FR, FL, RR, RL；每组: abad, hip, knee）
        # 与 state.q_abad[i], state.q_hip[i], state.q_knee[i] 顺序一致
        # 也与模型期望的关节顺序一致，直接拼接
        dof_pos = np.concatenate([
            np.array(state.q_abad, dtype=np.float32),
            np.array(state.q_hip, dtype=np.float32),
            np.array(state.q_knee, dtype=np.float32),
        ])
        dof_vel = np.concatenate([
            np.array(state.qd_abad, dtype=np.float32),
            np.array(state.qd_hip, dtype=np.float32),
            np.array(state.qd_knee, dtype=np.float32),
        ])

        single_obs = np.clip(
            np.concatenate([
                self.commands * COMMAND_SCALE,               # 0:3 velocity_commands
                np.array(gyro, dtype=np.float32) * BASE_ANG_VEL_SCALE,  # 3:6 base_ang_vel
                projected_gravity,                            # 6:9 projected_gravity
                dof_pos - DEFAULT_DOF_POS,                   # 9:21 joint_pos_rel
                dof_vel * JOINT_VEL_SCALE,                   # 21:33 joint_vel_rel
                self.last_actions,                            # 33:45 last_action
            ]),
            -100.0,
            100.0,
        ).astype(np.float32)

        return single_obs

    def build_history_obs(self, current_obs):
        """构造 270 维历史观测：[obs_t, obs_t-1, ..., obs_t-5]。"""
        self.obs_history.appendleft(current_obs)

        if len(self.obs_history) < HISTORY_LENGTH:
            padded = list(self.obs_history) + [
                np.zeros(NUM_ONE_STEP_OBS, dtype=np.float32)
                for _ in range(HISTORY_LENGTH - len(self.obs_history))
            ]
        else:
            padded = list(self.obs_history)

        history_obs = np.concatenate(padded, axis=-1).astype(np.float32)
        return history_obs.reshape(1, -1)
    
    def update_commands_keyboard(self):
        """键盘控制命令 (命令行输入) - 平滑归零处理"""
        step = 0.1

        # 平滑归零：按下 0 后，每帧 cmd 线性回归到 0
        if self.zero_cmd_pending:
            for i in range(3):
                v = self.commands[i]
                if abs(v) <= step:
                    self.commands[i] = 0.0
                else:
                    self.commands[i] -= step * (1 if v > 0 else -1)
            # 三轴都到 0 时清掉标志
            if float(np.max(np.abs(self.commands))) == 0.0:
                self.zero_cmd_pending = False
                print("[CMD] 归零完成")

        # 限制范围
        self.commands[0] = max(min(self.commands[0], 0.7), -0.7)
        self.commands[1] = max(min(self.commands[1], 1.0), -1.0)
        self.commands[2] = max(min(self.commands[2], 2.0), -2.0)

    def display_commands(self):
        """在命令行持续打印当前 cmd 与按键提示"""
        msg = (f"\r[CMD] vx={self.commands[0]:+.2f}  vy={self.commands[1]:+.2f}  wz={self.commands[2]:+.2f}  "
               f"| 8/2:前后  4/6:左右  7/9:旋转  0:归零  "
               f"S:站立 R:进RL  P:停止 Q:退出")
        sys.stdout.write(msg)
        sys.stdout.flush()
    
    def transition_to(self, new_state):
        """状态转换"""
        if new_state != self.current_state:
            print(f"[StateMachine] Transition: {self.current_state.name} -> {new_state.name}")
            self.previous_state = self.current_state
            self.current_state = new_state
            
            # 状态进入时的初始化
            if new_state == ControlState.STAND_UP:
                self._on_enter_stand_up()
            elif new_state == ControlState.RL_CONTROL:
                self._on_enter_rl_control()
            elif new_state == ControlState.PASSIVE:
                self._on_enter_passive()
    
    def _on_enter_stand_up(self):
        """进入STAND_UP状态"""
        self.stand_up_stage = 1
        self.stand_up_progress = 0.0
        self.stand_up_waiting_for_confirm = False
        
        # 获取当前关节状态作为插值起点
        state = self.motor_func.getMotorState()
        while not self.motor_func.haveMotorData():
            time.sleep(0.001)
            state = self.motor_func.getMotorState()
        
        for i in range(4):
            self.init_q_abad[i] = state.q_abad[i]
            self.init_q_hip[i] = state.q_hip[i]
            self.init_q_knee[i] = state.q_knee[i]
        
        print("[StandUp] Starting stand up sequence...")
    
    def _on_enter_rl_control(self):
        """进入RL_CONTROL状态"""
        self.start_control = True
        self.commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_history.clear()
        print("[RL_Control] RL policy control enabled")
    
    def _on_enter_passive(self):
        """进入PASSIVE状态"""
        self.start_control = False
        print("[Passive] Entering passive mode")
    
    def stand_up(self):
        """stand_up状态处理 - 站立起身"""
        if self.current_state != ControlState.STAND_UP:
            return
        
        state = self.motor_func.getMotorState()
        if not self.motor_func.haveMotorData():
            time.sleep(0.001)
            return
        
        self.stand_up_progress += 0.002
        ratio = min(self.stand_up_progress / self.stand_up_duration, 1.0)
        
        cmd = mc_sdk_zsl_1_py.MotorCommand()
        
        if self.stand_up_stage == 1:
            # 阶段1：插值到高位准备姿态
            targets = self.stand_up_targets['stage1']
            for i in range(4):
                cmd.q_des_abad[i] = ratio * targets['abad'] + (1.0 - ratio) * self.init_q_abad[i]
                cmd.q_des_hip[i] = ratio * targets['hip'] + (1.0 - ratio) * self.init_q_hip[i]
                cmd.q_des_knee[i] = ratio * targets['knee'] + (1.0 - ratio) * self.init_q_knee[i]
                cmd.kp_abad[i] = 60
                cmd.kp_hip[i] = 60
                cmd.kp_knee[i] = 60
                cmd.kd_abad[i] = 1.0
                cmd.kd_hip[i] = 1.0
                cmd.kd_knee[i] = 1.0
            
            # 阶段1完成，进入阶段2
            if ratio >= 1.0:
                self.stand_up_stage = 2
                self.stand_up_progress = 0.0
                
                # 获取当前关节状态作为下一阶段插值起点
                state = self.motor_func.getMotorState()
                while not self.motor_func.haveMotorData():
                    time.sleep(0.001)
                    state = self.motor_func.getMotorState()
                for i in range(4):
                    self.init_q_abad[i] = state.q_abad[i]
                    self.init_q_hip[i] = state.q_hip[i]
                    self.init_q_knee[i] = state.q_knee[i]
                
                print("[StandUp] Stage 1 complete, transitioning to Stage 2...")
        
        elif self.stand_up_stage == 2:
            # 阶段2：插值到policy默认姿态
            targets = self.stand_up_targets['stage2']
            for i in range(4):
                cmd.q_des_abad[i] = ratio * targets['abad'] + (1.0 - ratio) * self.init_q_abad[i]
                cmd.q_des_hip[i] = ratio * targets['hip'] + (1.0 - ratio) * self.init_q_hip[i]
                cmd.q_des_knee[i] = ratio * targets['knee'] + (1.0 - ratio) * self.init_q_knee[i]
                cmd.kp_abad[i] = 60
                cmd.kp_hip[i] = 60
                cmd.kp_knee[i] = 60
                cmd.kd_abad[i] = 1.0
                cmd.kd_hip[i] = 1.0
                cmd.kd_knee[i] = 1.0
            
            # 阶段2完成，等待用户按键确认切换到RL_CONTROL
            if ratio >= 1.0 and not self.stand_up_waiting_for_confirm:
                self.stand_up_waiting_for_confirm = True
                print("[StandUp] Stand up complete! Press [R] to enter RL_CONTROL or [P] to stop.")
        
        self.motor_func.sendMotorCmd(cmd)
        time.sleep(0.002)
    
    def RL_control(self):
        """RL_control状态处理 - RL策略控制 (使用 himloco ONNX 模型)"""
        if self.current_state != ControlState.RL_CONTROL:
            return

        state = self.motor_func.getMotorState()
        if not self.motor_func.haveMotorData():
            time.sleep(0.001)
            return

        # 更新键盘命令
        self.update_commands_keyboard()

        # 构建观测
        single_obs = self.build_single_obs(state)
        history_obs = self.build_history_obs(single_obs)

        # Encoder 推理
        encoder_output = self.encoder_session.run(
            [self.encoder_session.get_outputs()[0].name],
            {self.encoder_session.get_inputs()[0].name: history_obs},
        )[0]

        if encoder_output.shape[1] != ENCODER_OUTPUT_DIM:
            raise RuntimeError(f"Unexpected encoder output dim: {encoder_output.shape}, expected {ENCODER_OUTPUT_DIM}")

        # Policy 推理
        current_obs = single_obs.reshape(1, -1)
        policy_input = np.concatenate([current_obs, encoder_output], axis=1).astype(np.float32)

        if policy_input.shape[1] != POLICY_INPUT_DIM:
            raise RuntimeError(f"Unexpected policy input dim: {policy_input.shape}, expected (_, {POLICY_INPUT_DIM})")

        actions = self.policy_session.run(
            [self.policy_session.get_outputs()[0].name],
            {self.policy_session.get_inputs()[0].name: policy_input},
        )[0][0].astype(np.float32)

        # 更新上一帧动作缓存
        self.last_actions = actions.copy()

        # 计算目标关节位置（从 policy 顺序直接使用，无需重排）
        target_pos = np.clip(actions * ACTION_SCALE + DEFAULT_DOF_POS, -100.0, 100.0)

        # 打印频率控制 (5Hz)
        self.print_counter += 1
        if self.print_counter >= self.print_interval:
            self.print_counter = 0

            print("=" * 60)
            print("[RL] Observation at 5Hz:")
            print(f"  Commands (vx, vy, wz): {[f'{c:.3f}' for c in self.commands]}")
            print(f"  Raw actions:           {[f'{a:.4f}' for a in actions]}")
            print(f"  Target pos (motor order):  {[f'{p:.3f}' for p in target_pos]}")
            print("=" * 60)

        # 发送指令
        cmd = mc_sdk_zsl_1_py.MotorCommand()
        for i in range(4):
            # 每条腿的所有关节: FR_abad, FL_abad, RR_abad, RL_abad
            cmd.q_des_abad[i] = target_pos[i]
        for i in range(4):
            # 每条腿的所有关节: FR_hip, FL_hip, RR_hip, RL_hip
            cmd.q_des_hip[i] = target_pos[4 + i]
        for i in range(4):
            # 每条腿的所有关节: FR_knee, FL_knee, RR_knee, RL_knee
            cmd.q_des_knee[i] = target_pos[8 + i]

        for i in range(4):
            cmd.kp_abad[i] = self.kp
            cmd.kp_hip[i] = self.kp
            cmd.kp_knee[i] = self.kp
            cmd.kd_abad[i] = self.kd
            cmd.kd_hip[i] = self.kd
            cmd.kd_knee[i] = self.kd

            cmd.qd_des_abad[i] = 0.0
            cmd.qd_des_hip[i] = 0.0
            cmd.qd_des_knee[i] = 0.0
            cmd.tau_abad_ff[i] = 0.0
            cmd.tau_hip_ff[i] = 0.0
            cmd.tau_knee_ff[i] = 0.0

        # 仅供调试的时候注释，正式运行的时候取消注释
        self.motor_func.sendMotorCmd(cmd)
        time.sleep(0.005)
    
    def passive(self):
        """passive状态处理 - 被动/停止状态"""
        if self.current_state != ControlState.PASSIVE:
            return
        
        cmd = mc_sdk_zsl_1_py.MotorCommand()
        for i in range(4):
            cmd.q_des_abad[i] = 0.0
            cmd.q_des_hip[i] = 0.0
            cmd.q_des_knee[i] = -1.5
            cmd.kp_abad[i] = 0.0
            cmd.kp_hip[i] = 0.0
            cmd.kp_knee[i] = 0.0
            cmd.kd_abad[i] = 4.0
            cmd.kd_hip[i] = 4.0
            cmd.kd_knee[i] = 4.0
        
        ret = self.motor_func.sendMotorCmd(cmd)
        if ret < 0:
            print("[Passive] Send cmd error")
        
        time.sleep(0.002)
    
    def handle_keyboard_input(self):
        """处理键盘输入进行状态切换 (命令行输入)"""
        # 取出并清空一次性事件队列
        with self.event_lock:
            events = self.key_events
            self.key_events = []

        for key in events:
            # P键：紧急停止，切换到PASSIVE（最高优先级，立即返回）
            if key == 'p':
                if self.current_state != ControlState.PASSIVE:
                    self.transition_to(ControlState.PASSIVE)
                return True

            # Q键：退出程序
            if key == 'q':
                return False

            # 数字键：增量控制（仅在RL_CONTROL状态下生效）
            if self.current_state == ControlState.RL_CONTROL:
                step = 0.1
                if key == '8':  # 前进
                    self.commands[0] = min(self.commands[0] + step, 0.7)
                elif key == '2':  # 后退
                    self.commands[0] = max(self.commands[0] - step, -0.7)
                elif key == '6':  # 右移
                    self.commands[1] = min(self.commands[1] + step, 1.0)
                elif key == '4':  # 左移
                    self.commands[1] = max(self.commands[1] - step, -1.0)
                elif key == '7':  # 左旋
                    self.commands[2] = min(self.commands[2] + step, 2.0)
                elif key == '9':  # 右旋
                    self.commands[2] = max(self.commands[2] - step, -2.0)

                # 数字键按下时打印当前命令
                if key in ('8', '2', '4', '6', '7', '9'):
                    print(f"[CMD] vx={self.commands[0]:+.2f}  vy={self.commands[1]:+.2f}  wz={self.commands[2]:+.2f}")
                    continue

            # 0键：把 cmd 线性归零（打标志，由 update_commands_keyboard 实际执行）
            if key == '0':
                self.zero_cmd_pending = True
                print(f"[CMD] -> 平滑归零启动  vx={self.commands[0]:+.2f}  vy={self.commands[1]:+.2f}  wz={self.commands[2]:+.2f}")
                continue

            # S键：从PASSIVE切换到STAND_UP
            if key == 's':
                if self.current_state == ControlState.PASSIVE:
                    self.transition_to(ControlState.STAND_UP)
                return True

            # R键：从STAND_UP等待确认状态切换到RL_CONTROL
            if key == 'r':
                if self.current_state == ControlState.STAND_UP and self.stand_up_waiting_for_confirm:
                    self.stand_up_waiting_for_confirm = False
                    self.transition_to(ControlState.RL_CONTROL)
                return True

        return True

    def cli_input_loop(self):
        """命令行输入线程：持续读取用户按键，更新共享状态/事件队列"""
        print("=" * 50)
        print("命令行控制已就绪 (无 pygame)")
        print("=" * 50)
        print("状态切换: S=站立 R=进RL P=停止 Q=退出")
        print("增量控制: 8/2=前后  4/6=左右  7/9=旋转 (按一次 ±0.1)")
        print("归零:     0 (把 cmd 全部归零)")
        print("=" * 50)
        print("请输入按键后回车 (例如 8 然后回车 表示前进一次):")

        while True:
            try:
                raw = input()
            except EOFError:
                # stdin 关闭，结束线程
                return
            if not raw:
                continue
            key = raw.strip().lower()

            if key == 'quit' or key == 'exit':
                with self.event_lock:
                    self.key_events.append('q')
                return

            # 映射有效按键
            valid = {'8', '2', '4', '6', '7', '9', '0',
                     's', 'r', 'q', 'p'}
            if key not in valid:
                # 无效按键忽略
                print(f"[CLI] 无效按键: '{raw}' (忽略)")
                continue

            mapped = 'p' if key == 'p' else key

            # 所有有效按键都进入事件队列（包括数字键）
            with self.event_lock:
                self.key_events.append(mapped)
    
    def run(self):
        """主运行循环"""
        # 诊断：检查 DISPLAY 和 X11 socket（现在不依赖图形，但保留以便排查）
        import os as _os
        print(f"[Diag] DISPLAY={_os.environ.get('DISPLAY', '<NOT SET>')}")
        print(f"[Diag] 模式: 命令行输入 (无 pygame / 无 X server)")

        # 启动命令行输入线程
        self.cli_thread = threading.Thread(target=self.cli_input_loop, daemon=True)
        self.cli_thread.start()

        print("=" * 50)
        print("ZSIBot State Machine Controller (命令行模式)")
        print("=" * 50)
        print("State Flow: PASSIVE -> STAND_UP -> RL_CONTROL -> PASSIVE")
        print("Controls:")
        print("  [S] - Start STAND_UP (from PASSIVE)")
        print("  [R] - Enter RL_CONTROL (after stand up complete)")
        print("  [P] - Switch to PASSIVE (safe stop)")
        print("  [Q] / quit / exit - Quit")
        print("  [8/2] [4/6] [7/9] - vx / vy / wz 增量 ±0.1")
        print("  [0] - 把 cmd 全部归零")
        print("=" * 50)

        running = True
        while running:
            # 处理键盘输入（消费 key_events 队列）
            if not self.handle_keyboard_input():
                running = False
                break

            # 根据当前状态执行对应的处理函数
            if self.current_state == ControlState.STAND_UP:
                self.stand_up()
            elif self.current_state == ControlState.RL_CONTROL:
                self.RL_control()
            elif self.current_state == ControlState.PASSIVE:
                self.passive()

            # 控制循环频率 (原 pygame clock.tick(500)，即 2ms 一帧)
            time.sleep(0.002)

        # 退出前进入PASSIVE状态
        self.transition_to(ControlState.PASSIVE)
        print("\n[Main] Sending passive commands before exit...")
        for _ in range(100):
            self.passive()

        print("[Main] Exit complete")
    
    def stop(self):
        """停止并进入被动模式"""
        self.transition_to(ControlState.PASSIVE)
        
        flag = True
        send_msg_count = 0
        while flag:
            self.passive()
            send_msg_count += 1
            if send_msg_count > 1500:
                flag = False


if __name__ == "__main__":
    motor_control = MotorStateMachine()
    try:
        motor_control.run()
    except KeyboardInterrupt:
        motor_control.stop()
