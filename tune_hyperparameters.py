import sys
import os
from rl_zoo3.train import train
import simulation

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

reward_fn = "SELECTED_REWARD_FUNCTION"

sys.argv = [
    "python",
    "-optimize",
    "--algo",
    "ppo",
    "--env",
    "CPUHumanoid",
    "--log-folder",
    "data/tuning_logs",
    "-n",
    "150000",
    "--n-evaluations",
    "5",
    "--n-trials",
    "250",
    "--n-jobs",
    "1",
    "--sampler",
    "tpe",
    "--pruner",
    "median",
    "--verbose",
    "5",
    "--eval-episodes",
    "10",
    "--env-kwargs",
    'reward_fn:"{}"'.format(reward_fn),
    "randomization_factor:1",
    "--eval-env-kwargs",
    'reward_fn:"{}"'.format(reward_fn),
    "randomization_factor:1",
    "use_potential_rewards:0",
    "--conf-file",
    "simulation.hyperparam_config",
]

train()
