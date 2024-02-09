from gpu_batch_simulation import GPUBatchSimulation
import time
from reward_functions import *
import matplotlib.pyplot as plt

timestep = 0.005
simulation_time = 500 #seconds
upper_bound_training_time = 50112000
physics_steps_per_control_step = 5
sim_batch_sizes = [128, 256, 512, 1024]
# for reference, DeepMind's longest training took approximately 50,112,000 Seconds (580 days) simulation time

plot_x = [7,8,9,10]
gpu_plot_y = []

for sim_batch_size in sim_batch_sizes:

    sim_batch = GPUBatchSimulation(count=sim_batch_size, xml_path="assets/world.xml", reward_fn=standingRewardFn, physics_steps_per_control_step=physics_steps_per_control_step, timestep=timestep, randomization_factor=1)

    # step once because the first one takes significantly longer than the rest
    rewards = sim_batch.step()
            
    sim_time_executed = 0
    start_time = time.time()

    while sim_time_executed < simulation_time:
        while all(sim_batch.data_batch.time < 2):
            observations = sim_batch.getObs()
            rewards = sim_batch.step()
            sim_time_executed += timestep * sim_batch_size * physics_steps_per_control_step
            print("{}%".format(100 * sim_time_executed / simulation_time))
        sim_batch.reset()
            
    end_time = time.time()

    sim_per_wall_clock = sim_time_executed / (end_time - start_time)
    gpu_plot_y.append(sim_per_wall_clock)

    print("{} sim seconds per wall clock seconds".format(sim_per_wall_clock))

    print("Would take (upper bound) {} seconds ({} days) to finish training RL agent. (not including policy updates)".format(upper_bound_training_time / sim_per_wall_clock, (upper_bound_training_time / sim_per_wall_clock) / 86400))
    
    
sim_per_wall_clock = 13.193907058713098 # gotten by running cpu_benchmark.py
cpu_plot_y = [sim_per_wall_clock] * len(gpu_plot_y)

plt.plot(plot_x, gpu_plot_y, label='GPU')
plt.plot(plot_x, cpu_plot_y, label='CPU')
plt.xlabel('Batch Size (2^N)')
plt.ylabel('Seconds simulated per. wall clock seconds')
plt.legend()
plt.savefig("data/sim_benchmark.png")