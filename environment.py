from simulation import Simulation
import jax
from jax import numpy as jp
import numpy as np

# class for running parallel instances of the simulation at once

class Environment:
    def __init__(self, num_threads, xml_path, timestep=0.001, randomization_factor=0):
        if jax.default_backend() != 'gpu':
            print("ERROR: failed to find GPU device.")
            exit()
        
        self.num_threads = num_threads
        self.timestep = timestep
        self.randomization_factor = randomization_factor
            
        # Create num_threads simulation instances using vmap
        self.simulations = jax.jit(jax.vmap(lambda _ : Simulation(xml_path=xml_path, timestep=0.005, randomization_factor=self.timestep, run_on_gpu=True)))(jp.arange(num_threads))
        print(len(self.simulations))
        
    def step(self): pass
        
    def reset(self): pass
        
    def getObs(self): pass
        
    def getReward(self): pass


if __name__ == "__main__":
    env = Environment(num_threads=50, xml_path="assets/world.xml", timestep=0.005, randomization_factor=1)