import sys
from rl_zoo3.train import train
from gymnasium.envs.registration import register

register(
    id='GPUHumanoid',
    entry_point='simulation.gpu_vec_env:GPUVecEnv'
)
register(
    id='CPUHumanoid',
    entry_point='simulation.cpu_env:CPUEnv'
)

reward_fn = "standingRewardFn"

platform_choice = int(input("CPU or GPU? (0 or 1)\n"))

if platform_choice == 0:
    sys.argv = ["python", "-optimize",
                "--algo", "ppo",
                "--env", "CPUHumanoid",
                "--log-folder", "data/tuning_logs",
                "-n", "50000",
                "--n-trials", "1000",
                "--n-jobs", "2",
                "--sampler", "tpe",
                "--pruner", "median",
                "--env-kwargs", "reward_fn:\"{}\"".format(reward_fn), "render_mode:",
                "--conf-file", "simulation.hyperparam_config"]
else:
    sys.argv = ["python", "-optimize",
                "--algo", "ppo",
                "--env", "GPUHumanoid",
                "--log-folder", "data/tuning_logs",
                "-n", "50000",
                "--n-trials", "1000",
                "--n-jobs", "2",
                "--sampler", "tpe",
                "--pruner", "median",
                "--env-kwargs", "num_envs:256", "reward_fn:\"{}\"".format(reward_fn),
                "--conf-file", "simulation.hyperparam_config"]
train()