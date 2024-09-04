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
    sim = DummyVecEnv(
        [
            lambda: CPUEnv(
                xml_path=SIM_XML_PATH,
                reward_fn=SELECTED_REWARD_FUNCTION,
                randomization_factor=1,
                enable_rendering=False,
            )
        ]
        * NUM_ENVS,
    )
    obs = sim.reset()

    total_step_time = 0
    total_step_calls = 0

    for i in range(args.n_iters // NUM_ENVS):
        print("{:.3f}%".format(100 * i / 1000), end="\r")
        action = np.random.uniform(-1, 1, (NUM_ENVS, len(JOINT_NAMES)))

        start_time = time.time()
        obs, _, _, _ = sim.step(action)
        end_time = time.time()
        total_step_time += end_time - start_time
        total_step_calls += NUM_ENVS

    print(f"Avg. Step Time: {total_step_time / total_step_calls}")

    # OG (25 CTRL HZ):
    # Avg. Step Time: 0.0161263108253479
    # Avg. Reset Time: 0.1434691371144475

    # 10 CTRL HZ
    # Avg. Step Time: 0.017626473426818848
    # Avg. Reset Time: 0.14543695523948866

    # 100 CTRL HZ
    # Avg. Step Time: 0.015401869773864747
    # Avg. Reset Time: 0.14616745048099095

    # >> going with 50

    # >> remove unused observations + pressure sensor geoms
    # Avg. Step Time: 0.013555242776870728
    # Avg. Reset Time: 0.14493062557318273

    # >> removed  <!-- <option jacobian="dense"> <flag eulerdamp="disable"/> </option> -->
    # Avg. Step Time: 0.012965701580047607
    # Avg. Reset Time: 0.1385032641582

    # >> reduced size of hfield
    # Avg. Step Time: 0.012962458610534668
    # Avg. Reset Time: 0.1354348904047257

    # >> avoid re-loading XML on every reset
    # Avg. Step Time: 0.013325360298156739
    # Avg. Reset Time: 0.03650708687611115
