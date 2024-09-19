from simulation.simulation_parameters import *
from simulation.cpu_env import CPUEnv
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation import SIM_XML_PATH
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of environments to run in parallel",
    )
    argparser.add_argument(
        "--n-iters",
        type=int,
        default=1000,
        help="Number of steps to run",
    )
    args = argparser.parse_args()
    print(args)

    NUM_ENVS = args.n_envs
    cpu_envs = [
        CPUEnv(
            xml_path=SIM_XML_PATH,
            reward_fn=SELECTED_REWARD_FUNCTION,
            randomization_factor=1,
            enable_rendering=False,
        )
        for _ in range(NUM_ENVS)
    ]
    sim = DummyVecEnv(
        [cpu_envs.pop] * NUM_ENVS,
    )
    obs = sim.reset()

    total_step_time = 0
    total_step_calls = 0

    for i in range(args.n_iters // NUM_ENVS):
        print("{:.3f}%".format(100 * i / (args.n_iters // NUM_ENVS)), end="\r")
        action = np.random.uniform(-1, 1, (NUM_ENVS, len(JOINT_NAMES)))

        start_time = time.time()
        obs, _, _, _ = sim.step(action)
        end_time = time.time()
        total_step_time += end_time - start_time
        total_step_calls += NUM_ENVS

    print(f"Avg. Step Time: {total_step_time / total_step_calls}")
