import matplotlib.pyplot as plt
import numpy as np

LOG_NAME = "SAC"  # "SAC" # "PPO"
RANDOMIZATION_FACTOR = 1.0
evaluations = np.load(
    "data/{}/training_results_r{}/evaluations.npz".format(
        LOG_NAME, RANDOMIZATION_FACTOR
    )
)

timesteps = evaluations["timesteps"]
rewards = np.mean(evaluations["results"], axis=1)
ep_lengths = np.mean(evaluations["ep_lengths"], axis=1)
num_episodes_averaged = evaluations["results"].shape[1]


def smoothMAconv(depth, data, scale=1):  # Moving average by numpy convolution
    dz = np.diff(depth)
    N = int(scale / dz[0])
    smoothed = 0
    return smoothed


#### PLOT REWARDS

window_size = 10
running_avgs = []
for i in range(len(rewards)):
    running_avg = 0
    n = 0
    for i in range(i, max(-1, i - window_size), -1):
        n += 1
        running_avg += rewards[i]
    running_avg = running_avg / n
    running_avgs.append(running_avg)

# Plot the curve
plt.plot(timesteps, rewards, label="Reward", color="blue")
# plt.plot(timesteps, running_avgs, label="Running Average", color="red")

# Add labels and title
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title(
    "Evaluation Reward for Standing Behavior (avg. over {} episodes per evaluation)".format(
        num_episodes_averaged
    )
)

# Show grid
plt.grid(True)

# Show the plot
plt.legend()
plt.show()

#### PLOT EPISODE LENGTHS

window_size = 10
running_avgs = []
for i in range(len(ep_lengths)):
    running_avg = 0
    n = 0
    for i in range(i, max(-1, i - window_size), -1):
        n += 1
        running_avg += ep_lengths[i]
    running_avg = running_avg / n
    running_avgs.append(running_avg)

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
