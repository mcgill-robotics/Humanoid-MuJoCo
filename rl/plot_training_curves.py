import matplotlib.pyplot as plt
import numpy as np
from simulation.simulation_parameters import *

eval_file_dirs = ["data/SAC/0_steps/"]

#####################

timesteps = []
rewards = []
ep_lengths = []
randomization_factors = []

for i in range(len(eval_file_dirs)):
    evaluations = np.load(eval_file_dirs[i] + "evaluations.npz")

    timesteps.extend(evaluations["timesteps"])
    rewards.extend(np.mean(evaluations["results"], axis=1))
    ep_lengths.extend(np.mean(evaluations["ep_lengths"], axis=1))
    num_episodes_averaged = evaluations["results"].shape[1]

    with open(eval_file_dirs[i] + "randomization_factors.csv", "r") as f:
        randomization_factors.extend([float(line) for line in f.readlines()])

#### PLOT REWARDS

if len(rewards) > 1:
    fig, ax1 = plt.subplots()

    ax1.plot(timesteps, rewards, color="blue")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(timesteps, randomization_factors, color="red")
    ax2.set_ylabel("Randomization Factor", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_ylim(
        [min(rewards) - 0.05 * (max(rewards) - min(rewards)), max(rewards) * 1.05]
    )
    ax2.set_ylim([0, 1])

    # Add title and legend
    plt.title(
        "Evaluation Reward for Standing Behavior (avg. over {} episodes)".format(
            num_episodes_averaged
        )
    )
    fig.tight_layout()  # To ensure no label overlap

    # Show grid and plot
    plt.grid(True)
    plt.show()
print("Rewards: {}".format(rewards))
#### PLOT EPISODE LENGTHS

if len(ep_lengths) > 1:
    fig, ax1 = plt.subplots()

    ax1.plot(timesteps, np.array(ep_lengths) / CONTROL_FREQUENCY, color="blue")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Episode Length (seconds)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(timesteps, randomization_factors, color="red")
    ax2.set_ylabel("Randomization Factor", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_ylim([0, 10])
    ax2.set_ylim([0, 1])

    # Add title and legend
    plt.title(
        "Evaluation Episode Lengths for Standing Behavior (avg. over {} episodes per evaluation)".format(
            num_episodes_averaged
        )
    )
    fig.tight_layout()  # To ensure no label overlap

    # Show grid and plot
    plt.grid(True)
    plt.show()

print("Episode Lengths: {}".format(ep_lengths))
