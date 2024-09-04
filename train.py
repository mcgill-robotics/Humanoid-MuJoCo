import os
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
import numpy as np
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from simulation.simulation_parameters import *
from randomization_adaptation_callback import RewardAdaptationCallback
from stdout_log_callback import STDOutLogCallback
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, VecCheckNan
import argparse

###########################
##   ARGUMENT  PARSING   ##
###########################

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--n-envs", type=int, default=32, help="Number of environments to run in parallel"
)
argparser.add_argument(
    "--silent",
    action="store_true",
    help="Pass this flag to prevent printing logs to stdout",
)
argparser.add_argument(
    "--n-eval-episodes",
    type=int,
    default=10,
    help="Number of episodes to evaluate the model on",
)
argparser.add_argument(
    "--eval-freq",
    type=int,
    default=int(25_000 * (CONTROL_FREQUENCY / 100)),
    help="Frequency of evaluations in timesteps",
)
argparser.add_argument(
    "--checkpoint-freq",
    type=int,
    default=int(1_000_000 * (CONTROL_FREQUENCY / 100)),
    help="Frequency of checkpoint saving, in timesteps",
)
argparser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help="Path to checkpoint to continue training from (must point to .zip file, without the .zip extension in the path)",
)
argparser.add_argument(
    "--name",
    type=str,
    default="SAC",
    help="Subfolder path to save training results in",
)
argparser.add_argument(
    "--n-steps",
    type=int,
    default=int(50_000_000 * (CONTROL_FREQUENCY / 100)),
    help="Total timesteps to train policy for, per randomization factor (can do less if reward threshold is reached early)",
)
argparser.add_argument(
    "--rand-init", type=float, default=0, help="Initial randomization factor value"
)
argparser.add_argument(
    "--target-success-rate",
    type=int,
    default=0.8,
    help="Target portion of successful evaluations (if we are above this, decrease randomization, if we are below, increase randomization factor)",
)
argparser.add_argument(
    "--max-eval-time",
    type=int,
    default=10,
    help="Maximum seconds to run evaluation episode for before considering it a success.",
)

args = argparser.parse_args()
print(args)

##########################
##  SETUP TRAIN PARAMS  ##
##########################

NUM_ENVS = args.n_envs
N_EVAL_EPISODES = args.n_eval_episodes
TOTAL_TIMESTEPS = args.n_steps
PROGRESS_LOG_FREQ = 5
RANDOMIZATION_FACTOR_INIT = args.rand_init
MAX_EVAL_SIM_TIME = args.max_eval_time
TARGET_SUCCESS_RATE = args.target_success_rate
RANDOMIZATION_ADAPTATION_INCREMENT = 0.01
if args.ckpt is not None:
    CHECKPOINT = args.ckpt.lstrip().rstrip()
    LOG_DIR_SUFFIX = "/" + os.path.splitext(os.path.basename(CHECKPOINT))[0]
else:
    CHECKPOINT = None
    LOG_DIR_SUFFIX = "/0_steps"
MAX_EVALS_AT_MAX_REWARD = (
    10  # after 10 successful evaluations at max randomization, end training
)

log_dir = "data/{}/".format(args.name.strip()) + LOG_DIR_SUFFIX
EVAL_FREQ = args.eval_freq // NUM_ENVS
CHECKPOINT_FREQ = args.checkpoint_freq // NUM_ENVS

##########################
##  ENVIRONMENT  SETUP  ##
##########################

env = VecMonitor(
    DummyVecEnv(
        [
            lambda: CPUEnv(
                xml_path=SIM_XML_PATH,
                reward_fn=SELECTED_REWARD_FUNCTION,
                randomization_factor=RANDOMIZATION_FACTOR_INIT,
                enable_rendering=False,
            )
        ]
        * NUM_ENVS
    )
)
eval_env = VecMonitor(
    DummyVecEnv(
        [
            lambda: CPUEnv(
                xml_path=SIM_XML_PATH,
                reward_fn=SELECTED_REWARD_FUNCTION,
                randomization_factor=RANDOMIZATION_FACTOR_INIT,
                use_potential_rewards=False,
                max_simulation_time_override=MAX_EVAL_SIM_TIME,
                reward_override=1.0 / CONTROL_FREQUENCY,
                enable_rendering=False,
            )
        ]
        * N_EVAL_EPISODES
    )
)


env = VecCheckNan(env, raise_exception=True)
eval_env = VecCheckNan(eval_env, raise_exception=True)

##########################
## MODEL INITIALIZATION ##
##########################

print("\nBeginning training.\n")

if CHECKPOINT is None:
    additional_kwargs = {
        "batch_size": NUM_ENVS * 4,
    }
    policy_args = {
        "net_arch": dict(pi=[128, 128, 128], qf=[128, 128, 128]),
        "activation_fn": nn.Tanh,
        "log_std_init": -1,
    }

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        policy_kwargs=policy_args,
        **additional_kwargs,
    )
else:
    model = SAC.load(
        path=CHECKPOINT,
        env=env,
    )

##########################
##    TRAINING  LOOP    ##
##########################

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=log_dir,
    name_prefix="",
    verbose=0,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0,
)

reward_adaptation_callback = RewardAdaptationCallback(
    envs=[env, eval_env],
    eval_cb=eval_callback,
    success_reward_threshold=TARGET_SUCCESS_RATE * MAX_EVAL_SIM_TIME,
    max_evals_at_max_reward=MAX_EVALS_AT_MAX_REWARD,
    initial_randomization_factor=RANDOMIZATION_FACTOR_INIT,
    randomization_increment=RANDOMIZATION_ADAPTATION_INCREMENT,
    log_dir=log_dir,
)


progress_callback = STDOutLogCallback(
    log_dir=log_dir,
    num_envs=NUM_ENVS,
    total_timesteps=TOTAL_TIMESTEPS,
    log_freq=PROGRESS_LOG_FREQ,
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[
        progress_callback,
        eval_callback,
        reward_adaptation_callback,
        checkpoint_callback,
    ],
    log_interval=None if args.silent else 1,
    reset_num_timesteps=False,
    progress_bar=False if args.silent else True,
)
