import sys
import os
from rl_zoo3.train import train
import simulation

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

reward_fn = "standingRewardFn"

platform_choice = int(input("CPU or GPU? (0 or 1)\n"))

if platform_choice == 0:
    sys.argv = ["python", "-optimize",
                "--algo", "ppo",
                "--env", "CPUHumanoid",
                "--log-folder", "data/tuning_logs",
                "-n", "50000",
                "--n-trials", "1000",
                "--n-jobs", "16",
                "--sampler", "tpe",
                "--pruner", "median",
                "-P",
                "--verbose", "0",
                "--env-kwargs", "reward_fn:\"{}\"".format(reward_fn), "randomization_factor:1",
                "--conf-file", "simulation.hyperparam_config"]
else:
    sys.argv = ["python", "-optimize",
                "--algo", "ppo",
                "--env", "GPUHumanoid",
                "--log-folder", "data/tuning_logs",
                "-n", "50000",
                "--n-trials", "1000",
                "--n-jobs", "1",
                "--sampler", "tpe",
                "--pruner", "median",
                "-P",
                "--env-is-vectorized",
                "--verbose", "0",
                "--env-kwargs", "num_envs:256", "reward_fn:\"{}\"".format(reward_fn), "randomization_factor:1",
                "--conf-file", "simulation.hyperparam_config"]
train()