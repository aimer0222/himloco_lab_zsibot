# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent trained with HimLoco RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


class PolicyLogger:
    """Logs the raw history-stacked obs and actions from the policy inference loop.

    Writes one record per timestep (rate-limited to 5 Hz) with:
      - step index
      - wall-clock timestamp
      - full history-stacked obs tensor for env=0  [shape: (45*(H+1),)]
      - current-step obs split into semantic chunks (first 45 dims)
      - 12-dim action tensor for env=0
    """

    # Semantic chunk layout for the current-step slice (first 45 dims).
    # Concatenation in obs_history_buf:
    #   [current_step(45), hist_step_1(45), hist_step_2(45), ...]
    OBS_CHUNK_LABELS = [
        ("velocity_commands",  0,  3),
        ("base_ang_vel",      3,  6),
        ("projected_gravity", 6,  9),
        ("joint_pos_rel",     9, 21),
        ("joint_vel_rel",    21, 33),
        ("last_action",      33, 45),
    ]

    def __init__(self, log_dir: str, num_one_step_obs: int, history_length: int, rate_hz: float = 5.0):
        self.rate_hz = rate_hz
        self.min_interval = 1.0 / rate_hz
        self.num_one_step_obs = num_one_step_obs
        self.history_length = history_length
        self.total_obs_dim = num_one_step_obs * (history_length + 1)

        output_dir = Path(log_dir) / "policy_logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.fpath = output_dir / f"policy_log_{timestamp}.jsonl"
        self._file = open(self.fpath, "w", buffering=1)

        legend = {name: {"start": s, "end": e} for name, s, e in self.OBS_CHUNK_LABELS}
        self._file.write(f"# OBS_CHUNK_LEGEND: {json.dumps(legend)}\n")
        self._file.write(f"# META: num_one_step_obs={num_one_step_obs}, history_length={history_length}, total_obs_dim={self.total_obs_dim}\n")
        self._file.write("# COLUMNS: step | timestamp_s | obs_full_env0 (history-stacked) | obs_current_step_45d | obs_chunks_env0 | action_env0\n")

        self._last_write_time = 0.0
        print(f"[PolicyLogger] Writing to {self.fpath}")
        print(f"[PolicyLogger] Logging shape: obs=({self.total_obs_dim},), action=(12,)")

    def log(self, step: int, obs: torch.Tensor, actions: torch.Tensor):
        """Write a record if enough wall-clock time has elapsed since the last write."""
        now = time.time()
        if now - self._last_write_time < self.min_interval:
            return
        self._last_write_time = now

        # Env=0 only
        obs0 = obs[0].cpu().tolist()
        act0 = actions[0].cpu().tolist()

        # Current-step slice (first 45 dims)
        one_step = obs[0, : self.num_one_step_obs].cpu().tolist()
        chunks = {name: one_step[s:e] for name, s, e in self.OBS_CHUNK_LABELS}

        record = {
            "step": step,
            "timestamp_s": round(now, 4),
            "obs_full_env0": obs0,
            "obs_current_step_45d": one_step,
            "obs_chunks_env0": chunks,
            "action_env0": act0,
        }
        self._file.write(json.dumps(record) + "\n")

    def close(self):
        self._file.close()
        print(f"[PolicyLogger] Closed {self.fpath}")

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with HimLoco RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--env_cfg_entry_point", type=str, default="play_env_cfg_entry_point", help="Name of the play environment configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from himloco_lab.utils import export_himloco_policy_as_jit, export_himloco_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Play with HimLoco RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    # A bare checkpoint name together with --load_run is resolved inside that
    # run directory. Paths (absolute or relative) keep the existing direct-file
    # behavior supported by retrieve_file_path().
    checkpoint_is_bare_name = args_cli.checkpoint and os.path.basename(args_cli.checkpoint) == args_cli.checkpoint
    if checkpoint_is_bare_name and args_cli.load_run is not None:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length
    )

    print(f"[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
    print(f"[INFO] history_length: {env.history_length}")
    print(f"[INFO] num_obs (total): {env.num_obs}")
    if env.num_one_step_privileged_obs is not None:
        print(f"[INFO] num_one_step_privileged_obs: {env.num_one_step_privileged_obs}")
        print(f"[INFO] privileged_history_length: {env.privileged_history_length}")
        print(f"[INFO] num_privileged_obs (total): {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    # Instantiate the policy logger
    policy_logger = PolicyLogger(
        log_dir=log_dir,
        num_one_step_obs=env.num_one_step_obs,
        history_length=env.history_length,
        rate_hz=5.0,
    )

    # Print joint order for each observation/action term that uses joints
    robot = env.unwrapped.scene["robot"]
    all_joint_names = robot.joint_names
    print("[INFO] Joint order in policy observations (URDF declaration order):")
    print(f"  All DOFs ({len(all_joint_names)}): {all_joint_names}")
    print()

    # Find joint_ids for the two key observation terms and for the action term
    obs_term_cfgs = env.unwrapped.observation_manager._group_obs_term_cfgs["policy"]
    obs_names = env.unwrapped.observation_manager.active_terms["policy"]

    joint_obs_terms = {
        "joint_pos_rel": "joint_pos_rel",
        "joint_vel_rel": "joint_vel_rel",
        "last_action": "last_action",
    }

    print(f"  {'Term':<20} {'Dim':<6} {'Joint Names'}")
    print(f"  {'-'*20} {'-'*6} {'-'*60}")
    for obs_name, label in joint_obs_terms.items():
        if obs_name not in obs_names:
            continue
        obs_idx = obs_names.index(obs_name)
        obs_cfg = obs_term_cfgs[obs_idx]
        obs_tensor = obs_cfg.func(env.unwrapped, **obs_cfg.params)
        dim = obs_tensor.shape[1]
        if dim == 12:
            print(f"  {obs_name:<20} {dim:<6} {all_joint_names}")

    # Also show action term joint order
    act_names = env.unwrapped.action_manager.active_terms
    print()
    print(f"  {'Term':<20} {'Dim':<6} {'Joint Names'}")
    print(f"  {'-'*20} {'-'*6} {'-'*60}")
    for act_name in act_names:
        act_term = next(iter(env.unwrapped.action_manager._terms.values()))
        act_jids = act_term._joint_ids
        if act_jids == slice(None):
            act_joint_names = all_joint_names
        else:
            act_joint_names = [all_joint_names[j] for j in act_jids]
        print(f"  {act_name:<20} {act_term.action_dim:<6} {act_joint_names}")
    print()

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # create runner from HimLoco RSL-RL
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # load the checkpoint
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.device)

    # export HimLoco dual network (encoder + policy) as TorchScript JIT and ONNX
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    print(f"[INFO] Exporting HimLoco dual network to: {export_model_dir}")
    
    # Export as TorchScript JIT (combined encoder + policy)
    export_himloco_policy_as_jit(
        runner.alg.actor_critic,
        path=export_model_dir,
        policy_filename="policy.pt"
    )
    
    # Export as ONNX (separate encoder and policy)
    export_himloco_policy_as_onnx(
        runner.alg.actor_critic,
        path=export_model_dir,
        encoder_filename="encoder.onnx",
        policy_filename="policy.onnx",
        verbose=False
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # simulate environment
    print("[INFO] Starting playback...")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(actions)

        # Log policy obs and actions at 5 Hz
        policy_logger.log(timestep, obs, actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the policy logger
    policy_logger.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
