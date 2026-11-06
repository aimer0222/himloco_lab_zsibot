import gymnasium as gym


gym.register(
    id="ZsiBot-GML-Velocity",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "himloco_rsl_rl_cfg": "himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
        "rsl_rl_cfg_entry_point": "himloco_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="ZsiBot-GML-Velocity-Play",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "himloco_rsl_rl_cfg": "himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)
