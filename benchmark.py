from simulation import Simulation
import time

timestep = 0.01
simulation_time = 500 #second
upper_bound_training_time = 50112000
# for reference, DeepMind's longest training took approximately 50,112,000 Seconds (580 days) simulation time

sim = Simulation("assets/flat.xml", timestep=timestep, randomization_factor=1)
sim_time_executed = 0

start_time = time.time()

while sim_time_executed < simulation_time:
    sim.reset()
    action = [0]*4
    while sim.data.time < 2:
        sim.getState()
        reward = sim.step(action)
        sim_time_executed += timestep
        print("{}%".format(100 * sim_time_executed / simulation_time))
        # sim.render()
        
end_time = time.time()

sim_per_wall_clock = simulation_time / (end_time - start_time)

print("{} sim seconds per wall clock seconds".format(sim_per_wall_clock))

print("Would take (upper bound) {} seconds to finish training RL agent. (not including policy updates)".format(upper_bound_training_time / sim_per_wall_clock))