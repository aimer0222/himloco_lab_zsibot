# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for ZsiBot robots."""

import os

# Directory of this file
DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    )


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = False
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `zsibot_mujoco`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        import os

        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


ZSIBOT_ZSL1_CFG = ArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{DIRECTORY}/robot/zsl1_description/urdf/zsl1_payload.urdf",
        # asset_path=f"{DIRECTORY}/robot/zsl1_description/urdf/zsl1.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*_ABAD_JOINT": 0.0,
            ".*_HIP_JOINT": 0.8,
            ".*_KNEE_JOINT": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            effort_limit=28,
            saturation_effort=28,
            velocity_limit=28,
            stiffness=20.0,
            damping=0.7,
            friction=0.0,
        ),
    },
)
"""Configuration of ZsiBot ZSL1 robot."""


ZSIBOT_ZSL1_GML_CFG = ZSIBOT_ZSL1_CFG.replace(
    spawn=ZSIBOT_ZSL1_CFG.spawn.replace(
        asset_path=f"{DIRECTORY}/robot/zsl1_gml/urdf/zsl1_inspection.urdf",
    ),
)
"""Configuration of the ZsiBot ZSL1 GML inspection platform.

The GML platform keeps the same leg actuators, initial joint state, and control
parameters as :data:`ZSIBOT_ZSL1_CFG`. Only the robot description is replaced.
"""
