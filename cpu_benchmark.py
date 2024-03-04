from humanoid.simulation.cpu_simulation import CPUSimulation
from humanoid.simulation.reward_functions import *
import time
from humanoid import SIM_XML_PATH


def estimateCPUSimSpeed():
    timestep = 0.001
    simulation_time = 100 #seconds
    physics_steps_per_control_step = 5

    sim = CPUSimulation(SIM_XML_PATH,
                        reward_fn=standingRewardFn,
                        timestep=timestep,
                        physics_steps_per_control_step=physics_steps_per_control_step,
                        randomization_factor=1)
    
    sim_time_executed = 0
    start_time = time.time()

    while sim_time_executed < simulation_time:
        isTerminal = False
        while not isTerminal:
            sim.getObs()
            sim.step()
            _, isTerminal = sim.computeReward()
            sim_time_executed += timestep * physics_steps_per_control_step
            print("{}%                   ".format(100 * sim_time_executed / simulation_time), end='\r')
        sim.reset()
            
    end_time = time.time()
    
    return sim_time_executed / (end_time - start_time)

if __name__ == "__main__":
    sim_per_wall_clock = estimateCPUSimSpeed()
    
    upper_bound_training_time = 50112000
    # for reference, DeepMind's longest training took approximately 50,112,000 Seconds (580 days) simulation time

    print("{} sim seconds per wall clock seconds                                  ".format(sim_per_wall_clock))

    print("Would take (upper bound) {} seconds ({} days) to finish training RL agent. (not including policy updates)".format(upper_bound_training_time / sim_per_wall_clock, (upper_bound_training_time / sim_per_wall_clock) / 86400))