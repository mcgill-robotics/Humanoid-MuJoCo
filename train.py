import os
from simulation.reward_functions import *
from simulation.gpu_vec_env import GPUVecEnv
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
import numpy as np
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from reward_adaptation_callback import RewardAdaptationCallback
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
    "--cpu", action="store_true", help="Pass this flag to run on CPU"
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
    default=10_000,
    help="Frequency of evaluations in timesteps",
)
argparser.add_argument(
    "--checkpoint-freq",
    type=int,
    default=100_000,
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
    default=None,
    help="Subfolder path to save training results in",
)
argparser.add_argument(
    "--n-steps",
    type=int,
    default=50_000_000,
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
SIMULATE_ON_GPU = not args.cpu
N_EVAL_EPISODES = args.n_eval_episodes
TOTAL_TIMESTEPS = args.n_steps
RANDOMIZATION_FACTOR_INIT = args.rand_init
MAX_EVAL_SIM_TIME = args.max_eval_time
TARGET_SUCCESS_RATE = args.target_success_rate
RANDOMIZATION_ADAPTATION_INCREMENT = 0.01
if args.ckpt is not None:
    CHECKPOINT = args.ckpt.lstrip().rstrip()
else:
    CHECKPOINT = None
MAX_EVALS_AT_MAX_REWARD = (
    10  # after 10 successful evaluations at max randomization, end training
)

log_dir = "data/{}/".format(args.name.strip())
EVAL_FREQ = args.eval_freq // NUM_ENVS
CHECKPOINT_FREQ = args.checkpoint_freq // NUM_ENVS

##########################
##  ENVIRONMENT  SETUP  ##
##########################

if SIMULATE_ON_GPU:
    env = VecMonitor(
        GPUVecEnv(
            num_envs=NUM_ENVS,
            xml_path=SIM_XML_PATH,
            reward_fn=standupReward,
            randomization_factor=RANDOMIZATION_FACTOR_INIT,
            enable_rendering=False,
        )
    )
    eval_env = VecMonitor(
        GPUVecEnv(
            num_envs=N_EVAL_EPISODES,
            xml_path=SIM_XML_PATH,
            reward_fn=standupReward,
            randomization_factor=RANDOMIZATION_FACTOR_INIT,
            use_potential_rewards=False,
            max_simulation_time_override=MAX_EVAL_SIM_TIME,
            enable_rendering=False,
        )
    )

    print("\nInitializing environments...      ", end="")
    env.reset()
    env.step(None)
    eval_env.reset()
    eval_env.step(None)
    print("Done")
else:
    env = VecMonitor(
        DummyVecEnv(
            [
                lambda: CPUEnv(
                    xml_path=SIM_XML_PATH,
                    reward_fn=standupReward,
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
                    reward_fn=standupReward,
                    randomization_factor=RANDOMIZATION_FACTOR_INIT,
                    use_potential_rewards=False,
                    max_simulation_time_override=MAX_EVAL_SIM_TIME,
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
        "batch_size": 128,
    }
    policy_args = {
        "net_arch": dict(pi=[64, 64], qf=[64, 64]),
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
    name_prefix="ckpt",
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
    eval_freq=EVAL_FREQ,
    eval_cb=eval_callback,
    success_threshold=TARGET_SUCCESS_RATE,
    max_evals_at_max_reward=MAX_EVALS_AT_MAX_REWARD,
    initial_randomization_factor=RANDOMIZATION_FACTOR_INIT,
    randomization_increment=RANDOMIZATION_ADAPTATION_INCREMENT,
    verbose=0,
    log_dir=log_dir,
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback, reward_adaptation_callback],
    log_interval=1,
    reset_num_timesteps=False,
    progress_bar=True,
)
