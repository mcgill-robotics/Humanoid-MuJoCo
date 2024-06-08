import matplotlib.pyplot as plt

# Read the text file
with open("cpu_obs", "r") as file:
    data_cpu = file.read()
with open("gpu_obs", "r") as file:
    data_gpu = file.read()


gpu_obs = [v.split() for v in data_gpu.split("[") if len(v.split()) != 0]
cpu_obs = [v.split() for v in data_cpu.split("[") if len(v.split()) != 0]

# Process the data
values = []
for i in range(len(gpu_obs)):
    obs_g = [float(v.replace("]", "")) for v in gpu_obs[i]]
    obs_c = [float(v.replace("]", "")) for v in cpu_obs[i]]
    values.append([abs(obs_g[j] - obs_c[j]) for j in range(len(obs_g))])

max_diffs = []

for i in range(len(values[0])):
    y = [values[j][i] for j in range(len(values))]
    max_diffs.append(max([abs(v) for v in y]))

# Plot the values
# for i in range(len(values[0])):
#     x = [j for j in range(len(values))]
#     y = [values[j][i] for j in range(len(values))]
#     max_diffs.append(max([abs(v) for v in y]))
#     plt.plot(x, y, label="Index {}".format(i))

# plt.xlabel("Time")
# plt.ylabel("Error")
# plt.title("Obs. Difference between CPU and GPU for index {}".format(i))
# plt.legend()
# plt.show()

for i in range(len(max_diffs)):
    print(f"{i} : {max_diffs[i]}")
