import os
from simulation.reward_functions import *
from simulation.gpu_vec_env import GPUVecEnv
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
import torch
from torch import nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, VecCheckNan
import argparse

###########################
##   ARGUMENT  PARSING   ##
###########################

argparser = argparse.ArgumentParser()
argparser.add_argument("--algo", type=str, default="td3", help="Algorithm to use")
argparser.add_argument(
    "--n-envs", type=int, default=256, help="Number of environments to run in parallel"
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
    "--n-evals",
    type=int,
    default=100,
    help="Number of evaluations to run, per randomization factor (can do less if reward threshold is reached early)",
)
argparser.add_argument(
    "--n-checkpoints",
    type=int,
    default=10,
    help="Number of checkpoints to save, per randomization factor (can do less if reward threshold is reached early)",
)
argparser.add_argument(
    "--iterations",
    type=int,
    default=1000,
    help="Total iterations to train policy for, per randomization factor (can do less if reward threshold is reached early)",
)
argparser.add_argument(
    "--rand-init", type=float, default=0, help="Initial randomization factor value"
)
argparser.add_argument(
    "--rand-increment",
    type=float,
    default=0.1,
    help="How much to increment the randomization factor once reward threshold is reached",
)
argparser.add_argument(
    "--reward-goal",
    type=int,
    default=950,  # best possible reward is 1150 (by definition of reward function)
    help="Reward goal to reach. Ends training or increments randomization factor once reached in evaluation.",
)
argparser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help="Path to checkpoint to continue training from (must point to .zip file, without the .zip extension in the path)",
)
argparser.add_argument(
    "--log-name",
    type=str,
    default=None,
    help="Subfolder path to save training results in",
)

args = argparser.parse_args()
print(args)

##########################
##  SETUP TRAIN PARAMS  ##
##########################

MODEL_TYPE = {"td3": TD3, "sac": SAC, "ppo": PPO}[args.algo.lower()]
NUM_ENVS = args.n_envs
SIMULATE_ON_GPU = not args.cpu
N_EVAL_EPISODES = args.n_eval_episodes
NUM_EVALS = args.n_evals
NUM_CHECKPOINTS = args.n_checkpoints
TOTAL_TIMESTEPS = (
    args.iterations * 2048 * NUM_ENVS
    if MODEL_TYPE == PPO
    else args.iterations * NUM_ENVS
)  # because PPO uses 2048 timesteps per iteration and SAC/TD3 use 1 timestep per iteration
RANDOMIZATION_FACTOR = args.rand_init
RANDOMIZATION_INCREMENT = args.rand_increment
SUCCESSFUL_TRAINING_REWARD_THRESHOLD = args.reward_goal
CHECKPOINT = args.ckpt

if args.log_name is not None:
    log_dir = "data/{}/training_results".format(args.log_name.strip())
else:
    log_dir = "data/{}/training_results".format(args.algo.upper().strip())
EVAL_FREQ = TOTAL_TIMESTEPS // (NUM_EVALS * NUM_ENVS)
CHECKPOINT_FREQ = TOTAL_TIMESTEPS // (NUM_CHECKPOINTS * NUM_ENVS)

##########################
##  ENVIRONMENT  SETUP  ##
##########################

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

if SIMULATE_ON_GPU:
    env = VecMonitor(
        GPUVecEnv(
            num_envs=NUM_ENVS,
            xml_path=SIM_XML_PATH,
            reward_fn=controlInputRewardFn,
            randomization_factor=RANDOMIZATION_FACTOR,
        )
    )

    print("\nInitializing environment...      ", end="")
    env.reset()
    env.step(None)
    print("Done")
else:
    env = VecMonitor(
        DummyVecEnv(
            [
                lambda: CPUEnv(
                    xml_path=SIM_XML_PATH,
                    reward_fn=controlInputRewardFn,
                    randomization_factor=RANDOMIZATION_FACTOR,
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
                reward_fn=controlInputRewardFn,
                randomization_factor=RANDOMIZATION_FACTOR,
                use_potential_rewards=False,
                max_simulation_time_override=10.0,
            )
        ]
    )
)


env = VecCheckNan(env, raise_exception=True)
eval_env = VecCheckNan(eval_env, raise_exception=True)

##########################
## MODEL INITIALIZATION ##
##########################

print("\nBeginning training.\n")


if CHECKPOINT is None:
    policy_args = {
        "net_arch": dict(pi=[64, 64, 64], vf=[64, 64, 64], qf=[64, 64, 64]),
        "activation_fn": nn.Tanh,
    }

    additional_kwargs = {}
    if MODEL_TYPE == SAC:
        additional_kwargs["log_std_init"] = -1
    elif MODEL_TYPE == TD3:
        pass
    elif MODEL_TYPE == PPO:
        pass

    model = MODEL_TYPE(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        policy_kwargs=policy_args,
        **additional_kwargs
    )
else:
    model = MODEL_TYPE.load(
        path=CHECKPOINT,
        env=env,
    )

##########################
##    TRAINING  LOOP    ##
##########################

while True:
    print(" >> TRAINING WITH RANDOMIZATION FACTOR {}".format(RANDOMIZATION_FACTOR))
    env.set_attr("randomization_factor", RANDOMIZATION_FACTOR)
    env.reset()
    eval_env.set_attr("randomization_factor", RANDOMIZATION_FACTOR)
    eval_env.reset()

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=log_dir + "_r{}".format(RANDOMIZATION_FACTOR),
        name_prefix="checkpoint",
        verbose=0,
    )

    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=SUCCESSFUL_TRAINING_REWARD_THRESHOLD, verbose=0
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "_r{}".format(RANDOMIZATION_FACTOR),
        log_path=log_dir + "_r{}".format(RANDOMIZATION_FACTOR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_training_callback,
        verbose=0,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        log_interval=1,
        tb_log_name="Standing_r{}".format(RANDOMIZATION_FACTOR),
        reset_num_timesteps=False,
        progress_bar=True,
    )

    print(
        " >> COMPLETED TRAINING WITH RANDOMIZATION FACTOR {}".format(
            RANDOMIZATION_FACTOR
        )
    )

    if RANDOMIZATION_FACTOR == 1:
        break

    RANDOMIZATION_FACTOR += RANDOMIZATION_INCREMENT
    RANDOMIZATION_FACTOR = 1 if RANDOMIZATION_FACTOR > 1 else RANDOMIZATION_FACTOR
