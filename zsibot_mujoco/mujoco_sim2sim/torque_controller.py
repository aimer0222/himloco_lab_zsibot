"""力矩控制模块（PD + 力矩限幅）。

设计目标：
- 与 himloco deploy 端 `lowlevel_state_machine_himloco.py` 的 PD 逻辑保持一致：
  τ = kp * (q_target - q) - kd * qd  + τ_ff
- 与 Isaac Lab 中 `DCMotorCfg` 的饱和行为保持一致：
  τ_max(qd) = clip(τ_sat * (1 - qd/vel_limit), 0, τ_max)
  τ_min(qd) = clip(τ_sat * (-1 - qd/vel_limit), -τ_max, 0)
- 可被 sim2sim 与真实部署端共用一份接口。
"""
import numpy as np


class TorquePDController:
    """基于 PD + DC-motor 饱和模型的力矩控制器。

    Parameters
    ----------
    kp : float | np.ndarray
        关节刚度（与 himloco deploy 一致，默认 20.0）。
    kd : float | np.ndarray
        关节阻尼（与 himloco deploy 一致，默认 0.7）。
    effort_limit : float | np.ndarray
        关节最大扭矩，与 MuJoCo `<motor ctrlrange>` / 实际电机保持一致。
    saturation_effort : float | np.ndarray, optional
        直流电机饱和扭矩，默认等于 effort_limit。
    velocity_limit : float | np.ndarray, optional
        直流电机最大关节速度（rad/s），用于 DC-motor 速度相关饱和。
    """

    def __init__(
        self,
        kp: float = 20.0,
        kd: float = 0.7,
        effort_limit: float = 28.0,
        saturation_effort: float | None = None,
        velocity_limit: float = 28.0,
    ):
        self.kp = np.asarray(kp, dtype=np.float32)
        self.kd = np.asarray(kd, dtype=np.float32)
        self.effort_limit = np.asarray(effort_limit, dtype=np.float32)
        self.saturation_effort = np.asarray(
            saturation_effort if saturation_effort is not None else effort_limit,
            dtype=np.float32,
        )
        self.velocity_limit = np.asarray(velocity_limit, dtype=np.float32)

        if np.any(self.velocity_limit <= 0):
            raise ValueError("velocity_limit must be > 0")

    def compute(
        self,
        q_target: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        tau_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        """计算最终输出力矩。

        Parameters
        ----------
        q_target : (N,) array
            目标关节位置（rad），通常由策略 action 推导。
        q : (N,) array
            当前关节位置（rad），来自状态估计/传感器。
        qd : (N,) array
            当前关节速度（rad/s）。
        tau_ff : (N,) array, optional
            前馈力矩，None 时视为 0。

        Returns
        -------
        tau : (N,) array
            限幅后的输出力矩，可直接写入 MuJoCo `d.ctrl` 或电机命令。
        """
        q_target = np.asarray(q_target, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        qd = np.asarray(qd, dtype=np.float32)

        if tau_ff is None:
            tau_ff = np.zeros_like(q_target)
        else:
            tau_ff = np.asarray(tau_ff, dtype=np.float32)

        # PD 控制律 —— 与 himloco deploy 端 `cmd.q_des_* / cmd.kp_* / cmd.kd_*` 等价
        tau_unclamped = self.kp * (q_target - q) - self.kd * qd + tau_ff

        # DC-motor 速度相关饱和 —— 与 Isaac Lab `DCMotor._clip_effort` 等价
        tau_max = self.saturation_effort * (1.0 - qd / self.velocity_limit)
        tau_max = np.clip(tau_max, 0.0, self.effort_limit)

        tau_min = self.saturation_effort * (-1.0 - qd / self.velocity_limit)
        tau_min = np.clip(tau_min, -self.effort_limit, 0.0)

        return np.clip(tau_unclamped, tau_min, tau_max).astype(np.float32)

    # ---- 工具函数：与 himloco deploy / sim2sim 共用 ----
    @staticmethod
    def policy_action_to_target_pos(
        actions: np.ndarray,
        action_scale: np.ndarray,
        default_pos: np.ndarray,
    ) -> np.ndarray:
        """策略 action → 目标位置（policy 端通用公式）。"""
        actions = np.asarray(actions, dtype=np.float32)
        return np.clip(actions * action_scale + default_pos, -100.0, 100.0).astype(
            np.float32
        )