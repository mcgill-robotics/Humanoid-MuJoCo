import sys
import os
from rl_zoo3.train import train
import simulation

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

reward_fn = "standingRewardFn"

sys.argv = ["python", "-optimize",
            "--algo", "ppo",
            "--env", "CPUHumanoid",
            "--log-folder", "data/tuning_logs",
            "-n", "10000",
            "--eval-freq", "5000",
            "--n-trials", "500",
            "--n-jobs", "24",
            "--sampler", "tpe",
            "--pruner", "median",
            "-P",
            "--verbose", "0",
            "--save-freq", "5000",
            "--eval-episodes", "10",
            "--log-interval", "1",
            "--env-kwargs", "reward_fn:\"{}\"".format(reward_fn), "randomization_factor:1",
            "--eval-env-kwargs", "reward_fn:\"{}\"".format(reward_fn), "randomization_factor:1",
            "--conf-file", "simulation.hyperparam_config"]

train()