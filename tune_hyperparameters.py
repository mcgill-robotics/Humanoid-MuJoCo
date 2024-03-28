import sys
import os
from rl_zoo3.train import train
import simulation

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

reward_fn = "controlInputRewardFn"

sys.argv = ["python", "-optimize",
            "--algo", "ppo",
            "--env", "CPUHumanoid",
            "--log-folder", "data/tuning_logs",
            "-n", "150000",
            "--n-evaluations", "5",
            "--n-trials", "250",
            "--n-jobs", "24",
            "--sampler", "tpe",
            "--pruner", "median",
            "--verbose", "0",
            "--eval-episodes", "10",
            "--env-kwargs", "reward_fn:\"{}\"".format(reward_fn), "randomization_factor:1",
            "--eval-env-kwargs", "reward_fn:\"{}\"".format(reward_fn), "randomization_factor:1",
            "--conf-file", "simulation.hyperparam_config"]

train()