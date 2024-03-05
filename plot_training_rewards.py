import matplotlib.pyplot as plt
import numpy as np

timesteps = []
rewards = []

train_id = input("Enter the integer ID of the training CSV:    ")

# Read the CSV file
with open("data/training_logs/Standing/PPO_Standing_log_{}.csv".format(train_id), 'r') as file:
    # Skip header row
    next(file)
    
    for line in file:
        # Split the line by comma
        values = line.strip().split(',')
        
        # Extract X and Y values
        timesteps.append(float(values[1]))
        rewards.append(float(values[2]))

def smoothMAconv(depth, data, scale=1): # Moving average by numpy convolution
    dz = np.diff(depth) 
    N = int(scale/dz[0])
    smoothed=0
    return smoothed

window_size = 100
running_avgs = []
for i in range(len(rewards)):
    running_avg = 0
    n = 0
    for i in range(i, max(-1, i-window_size), -1):
        n += 1
        running_avg += rewards[i]
    running_avg = running_avg / n
    running_avgs.append(running_avg)

# Plot the curve
plt.plot(timesteps, rewards, label="Reward", color="blue")
plt.plot(timesteps, running_avgs, label="Running Average", color="red")

# Add labels and title
plt.xlabel('Timesteps')
plt.ylabel('Average Reward')
plt.title('Training Performance for Standing Behavior')

# Show grid
plt.grid(True)

# Show the plot
plt.legend()
plt.show()