import matplotlib.pyplot as plt
import numpy as np

LOG_NAMES = ["SAC_CPU_Vincent"]
RANDOMIZATION_FACTORS = [0.1]
eval_files = ["evaluations.npz"]

#####################

timesteps = []
rewards = []
ep_lengths = []

for i in range(len(LOG_NAMES)):
    evaluations = np.load(
        "data/{}/training_results_r{}/{}".format(
            LOG_NAMES[i % len(LOG_NAMES)],
            RANDOMIZATION_FACTORS[i % len(RANDOMIZATION_FACTORS)],
            eval_files[i % len(eval_files)],
        )
    )

    timesteps.extend(evaluations["timesteps"])
    rewards.extend(np.mean(evaluations["results"], axis=1))
    ep_lengths.extend(np.mean(evaluations["ep_lengths"], axis=1))
    num_episodes_averaged = evaluations["results"].shape[1]


#### PLOT REWARDS

window_size = 100
running_avgs = []
for i in range(len(rewards)):
    running_avg = 0
    n = 0
    for i in range(i, max(-1, i - window_size), -1):
        n += 1
        running_avg += rewards[i]
    running_avg = running_avg / n
    running_avgs.append(running_avg)

if len(rewards) > 1:
    # Plot the curve
    plt.plot(timesteps, rewards, label="Reward", color="blue")
    # plt.plot(timesteps, running_avgs, label="Running Average", color="red")

    # Add labels and title
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title(
        "Evaluation Reward for Standing Behavior (avg. over {} episodes)".format(
            num_episodes_averaged
        )
    )

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.legend()
    plt.show()
print("Rewards: {}".format(rewards))
#### PLOT EPISODE LENGTHS

window_size = 100
running_avgs = []
for i in range(len(ep_lengths)):
    running_avg = 0
    n = 0
    for i in range(i, max(-1, i - window_size), -1):
        n += 1
        running_avg += ep_lengths[i]
    running_avg = running_avg / n
    running_avgs.append(running_avg)

if len(ep_lengths) > 1:
    # Plot the curve
    plt.plot(timesteps, ep_lengths, label="Episode Length", color="blue")
    # plt.plot(timesteps, running_avgs, label="Running Average", color="red")

    # Add labels and title
    plt.xlabel("Steps")
    plt.ylabel("Steps before termination")
    plt.title(
        "Evaluation Episode Lengths for Standing Behavior (avg. over {} episodes per evaluation)".format(
            num_episodes_averaged
        )
    )

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.legend()
    plt.show()

print("Episode Lengths: {}".format(ep_lengths))
