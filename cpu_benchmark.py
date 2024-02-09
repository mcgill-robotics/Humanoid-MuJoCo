from cpu_simulation import CPUSimulation
import time
from reward_functions import *

timestep = 0.005
simulation_time = 500 #seconds
upper_bound_training_time = 50112000
physics_steps_per_control_step = 5
# for reference, DeepMind's longest training took approximately 50,112,000 Seconds (580 days) simulation time

sim = CPUSimulation("assets/world.xml", reward_fn=standingRewardFn, timestep=timestep, physics_steps_per_control_step=physics_steps_per_control_step, randomization_factor=1)
sim_time_executed = 0

start_time = time.time()

while sim_time_executed < simulation_time:
    while sim.data.time < 2:
        sim.getObs()
        sim.step()
        sim.computeReward()
        sim_time_executed += timestep * physics_steps_per_control_step
        print("{}%".format(100 * sim_time_executed / simulation_time))
    sim.reset()
        
end_time = time.time()

sim_per_wall_clock = sim_time_executed / (end_time - start_time)

print("{} sim seconds per wall clock seconds".format(sim_per_wall_clock))

print("Would take (upper bound) {} seconds ({} days) to finish training RL agent. (not including policy updates)".format(upper_bound_training_time / sim_per_wall_clock, (upper_bound_training_time / sim_per_wall_clock) / 86400))