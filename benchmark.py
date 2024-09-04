from simulation.simulation_parameters import *
from simulation.cpu_env import CPUEnv
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation import SIM_XML_PATH
import numpy as np
import time

sim = CPUEnv(
    xml_path=SIM_XML_PATH,
    reward_fn=SELECTED_REWARD_FUNCTION,
    randomization_factor=0,
    enable_rendering=False,
)
obs = sim.reset()

total_step_time = 0
total_step_calls = 0
total_reset_time = 0
total_reset_calls = 0

for i in range(1000):
    print("{:.3f}%".format(100 * i / 1000), end="\r")
    action = np.random.uniform(-1, 1, len(JOINT_NAMES))

    start_time = time.time()
    obs, reward, isTerminal, isTruncated, _ = sim.step(action)
    end_time = time.time()
    total_step_time += end_time - start_time
    total_step_calls += 1

    if isTerminal or isTruncated:
        start_time = time.time()
        sim.reset()
        end_time = time.time()
        total_reset_time += end_time - start_time
        total_reset_calls += 1

print(f"Avg. Step Time: {total_step_time / total_step_calls}")
print(f"Avg. Reset Time: {total_reset_time / total_reset_calls}")

# OG:
# Avg. Step Time: 0.0161263108253479
# Avg. Reset Time: 0.1434691371144475
