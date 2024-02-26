import matplotlib.pyplot as plt
import numpy as np

timesteps = []
rewards = []

# Read the CSV file
with open("data\PPO_logs\GPUStanding\PPO_GPUStanding_log_6.csv", 'r') as file:
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

window_size = 5
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
plt.plot(timesteps, rewards)
plt.plot(timesteps, running_avgs)

# Add labels and title
plt.xlabel('Timesteps')
plt.ylabel('Average Reward')
plt.title('Training Performance for Standing Behavior')

# Show grid
plt.grid(True)

# Show the plot
plt.show()