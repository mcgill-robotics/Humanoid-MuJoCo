from gpu_batch_simulation import GPUBatchSimulation
from cpu_benchmark import estimateCPUSimSpeed
import time
from reward_functions import *
import matplotlib.pyplot as plt

timestep = 0.005
simulation_time = 500 #seconds
upper_bound_training_time = 50112000
physics_steps_per_control_step = 5
sim_batch_sizes = [100, 200, 300, 400, 500]
sim_batch_sizes = [512]
# for reference, DeepMind's longest training took approximately 50,112,000 Seconds (580 days) simulation time

gpu_plot_y = []

for sim_batch_size in sim_batch_sizes:

    sim_batch = GPUBatchSimulation(count=sim_batch_size, xml_path="assets/world.xml", reward_fn=standingRewardFn, physics_steps_per_control_step=physics_steps_per_control_step, timestep=timestep, randomization_factor=1)
            
    sim_time_executed = 0
    start_time = time.time()

    while sim_time_executed < simulation_time:
        while all(sim_batch.data_batch.time < 2):
            observations = sim_batch.getObs()
            sim_batch.step()
            rewards = sim_batch.computeReward()
            sim_time_executed += timestep * sim_batch_size * physics_steps_per_control_step
            print("{}%".format(100 * sim_time_executed / simulation_time))
        sim_batch.reset()
            
    end_time = time.time()

    sim_per_wall_clock = sim_time_executed / (end_time - start_time)
    gpu_plot_y.append(sim_per_wall_clock)

    print("{} sim seconds per wall clock seconds".format(sim_per_wall_clock))

    print("Would take (upper bound) {} seconds ({} days) to finish training RL agent. (not including policy updates)".format(upper_bound_training_time / sim_per_wall_clock, (upper_bound_training_time / sim_per_wall_clock) / 86400))

cpu_plot_y = [estimateCPUSimSpeed()] * len(gpu_plot_y)

plt.plot(sim_batch_sizes, gpu_plot_y, label='GPU')
plt.plot(sim_batch_sizes, cpu_plot_y, label='CPU')
plt.xlabel('Batch Size')
plt.ylabel('Seconds simulated per. wall clock seconds')
plt.legend()
# plt.savefig("data/sim_benchmark.png")