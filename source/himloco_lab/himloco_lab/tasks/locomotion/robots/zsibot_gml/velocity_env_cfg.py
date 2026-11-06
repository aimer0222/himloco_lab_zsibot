"""Velocity task for the ZSL1 GML inspection platform.

All task and control settings are inherited from the original ZSL1 task. The
only difference is the robot URDF used by the scene.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from himloco_lab.assets.zsibot import ZSIBOT_ZSL1_GML_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion.robots.zsibot.velocity_env_cfg import (
    RobotEnvCfg as ZsiBotRobotEnvCfg,
    RobotPlayEnvCfg as ZsiBotRobotPlayEnvCfg,
    RobotSceneCfg as ZsiBotRobotSceneCfg,
)


@configclass
class RobotSceneCfg(ZsiBotRobotSceneCfg):
    """Original ZSL1 scene with the GML inspection platform asset."""

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RobotEnvCfg(ZsiBotRobotEnvCfg):
    """Training configuration for the ZSL1 GML inspection platform."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class RobotPlayEnvCfg(ZsiBotRobotPlayEnvCfg):
    """Playback configuration for the ZSL1 GML inspection platform."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=64, env_spacing=2.5)
