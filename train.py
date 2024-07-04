import os
from simulation.reward_functions import *
from simulation.gpu_vec_env import GPUVecEnv
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
import torch
import numpy as np
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
    "--eval-freq",
    type=int,
    default=100_000,
    help="Frequency of evaluations in timesteps",
)
argparser.add_argument(
    "--checkpoint-freq",
    type=int,
    default=1_000_000,
    help="Frequency of checkpoint saving, in timesteps",
)
argparser.add_argument(
    "--rand-init", type=float, default=0, help="Initial randomization factor value"
)
argparser.add_argument(
    "--reward-goal",
    type=int,
    default=950,  # best possible reward is 1 / timestep == 1_000
    help="Reward goal to reach (per second of sim time). Ends training or increments randomization factor once reached in evaluation.",
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
RANDOMIZATION_FACTOR = args.rand_init
RAND_FACTOR_INCREMENTS = [0.05] * 18
TRAINING_STEPS = [2_000_000] * 18
if abs(RANDOMIZATION_FACTOR + sum(RAND_FACTOR_INCREMENTS) - 1.0) > 0.001:
    print(
        "ERR: Randomization factor increments do not sum to 1.0 ({} + {}).".format(
            RANDOMIZATION_FACTOR, sum(RAND_FACTOR_INCREMENTS)
        )
    )
    exit()
MAX_EVAL_SIM_TIME = 10.0
SUCCESSFUL_TRAINING_REWARD_THRESHOLD = (
    np.inf if args.reward_goal <= 0 else args.reward_goal * MAX_EVAL_SIM_TIME
)
if args.ckpt is not None:
    CHECKPOINT = args.ckpt.lstrip().rstrip()
else:
    CHECKPOINT = None

if args.log_name is not None:
    log_dir = "data/{}/training_results".format(args.log_name.strip())
else:
    log_dir = "data/{}/training_results".format(args.algo.upper().strip())
EVAL_FREQ = args.eval_freq // NUM_ENVS
CHECKPOINT_FREQ = args.checkpoint_freq // NUM_ENVS

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
            enable_rendering=False,
        )
    )
    eval_env = VecMonitor(
        GPUVecEnv(
            num_envs=N_EVAL_EPISODES,
            xml_path=SIM_XML_PATH,
            reward_fn=controlInputRewardFn,
            randomization_factor=RANDOMIZATION_FACTOR,
            use_potential_rewards=False,
            max_simulation_time_override=10.0,
            enable_rendering=False,
            reward_override=1.0,
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
                    reward_fn=controlInputRewardFn,
                    randomization_factor=RANDOMIZATION_FACTOR,
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
                    reward_fn=controlInputRewardFn,
                    randomization_factor=RANDOMIZATION_FACTOR,
                    use_potential_rewards=False,
                    max_simulation_time_override=MAX_EVAL_SIM_TIME,
                    enable_rendering=False,
                    reward_override=1.0,
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
    additional_kwargs = {}
    policy_args = {
        "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256], qf=[256, 256, 256]),
        "activation_fn": nn.Tanh,
    }
    if MODEL_TYPE == SAC:
        policy_args["log_std_init"] = -1
        if False:  # CHANGE TO TRUE TO USE RL-ZOO3 PARAMS
            # from rl-zoo3 tuned params for Humanoid-v4:
            # n_timesteps: !!float 2e6
            # policy: 'MlpPolicy'
            # learning_starts: 10000

            # additional_kwargs["n_timesteps"] = 2e6
            additional_kwargs["learning_starts"] = 10000
    elif MODEL_TYPE == TD3:
        if False:  # CHANGE TO TRUE TO USE RL-ZOO3 PARAMS
            # from rl-zoo3 tuned params for Humanoid-v4:
            # n_timesteps: !!float 2e6
            # policy: 'MlpPolicy'
            # learning_starts: 10000
            # noise_type: 'normal'
            # noise_std: 0.1
            # train_freq: 1
            # gradient_steps: 1
            # learning_rate: !!float 1e-3
            # batch_size: 256
            # policy_kwargs: "dict(net_arch=[400, 300])"

            # additional_kwargs["n_timesteps"] = 2e6
            additional_kwargs["learning_starts"] = 10000
            additional_kwargs["noise_type"] = "normal"
            additional_kwargs["noise_std"] = 0.1
            additional_kwargs["train_freq"] = 1
            additional_kwargs["gradient_steps"] = 1
            additional_kwargs["learning_rate"] = 1e-3
            additional_kwargs["batch_size"] = 256
            policy_args["net_arch"] = [400, 300]
    elif MODEL_TYPE == PPO:
        if False:  # CHANGE TO TRUE TO USE RL-ZOO3 PARAMS
            # from rl-zoo3 tuned params for Humanoid-v4:
            # normalize: true
            # n_envs: 1
            # policy: 'MlpPolicy'
            # n_timesteps: !!float 1e7
            # batch_size: 256
            # n_steps: 512
            # gamma: 0.95
            # learning_rate: 3.56987e-05
            # ent_coef: 0.00238306
            # clip_range: 0.3
            # n_epochs: 5
            # gae_lambda: 0.9
            # max_grad_norm: 2
            # vf_coef: 0.431892
            # policy_kwargs: "dict(
            #             log_std_init=-2,
            #             ortho_init=False,
            #             activation_fn=nn.ReLU,
            #             net_arch=dict(pi=[256, 256], vf=[256, 256])
            #           )"

            # additional_kwargs["normalize"] = True
            # additional_kwargs["n_envs"] = 1
            # additional_kwargs["n_timesteps"] = 1e7
            additional_kwargs["batch_size"] = 256
            additional_kwargs["n_steps"] = 512
            additional_kwargs["gamma"] = 0.95
            additional_kwargs["learning_rate"] = 3.56987e-05
            additional_kwargs["ent_coef"] = 0.00238306
            additional_kwargs["clip_range"] = 0.3
            additional_kwargs["n_epochs"] = 5
            additional_kwargs["gae_lambda"] = 0.9
            additional_kwargs["max_grad_norm"] = 2
            additional_kwargs["vf_coef"] = 0.431892
            policy_args["log_std_init"] = -2
            policy_args["ortho_init"] = False
            # policy_args["activation_fn"] = nn.ReLU
            policy_args["net_arch"] = dict(pi=[256, 256], vf=[256, 256])

    model = MODEL_TYPE(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        policy_kwargs=policy_args,
        **additional_kwargs,
    )
else:
    model = MODEL_TYPE.load(
        path=CHECKPOINT,
        env=env,
    )

##########################
##    TRAINING  LOOP    ##
##########################

for n_steps, randomization_increment in zip(TRAINING_STEPS, RAND_FACTOR_INCREMENTS):
    print(" >> TRAINING WITH RANDOMIZATION FACTOR {:.1f}".format(RANDOMIZATION_FACTOR))
    env.set_attr("randomization_factor", RANDOMIZATION_FACTOR)
    eval_env.set_attr("randomization_factor", RANDOMIZATION_FACTOR)
    env.reset()
    eval_env.reset()

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=log_dir + "_r{:.1f}".format(RANDOMIZATION_FACTOR),
        name_prefix="checkpoint",
        verbose=0,
    )

    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=SUCCESSFUL_TRAINING_REWARD_THRESHOLD, verbose=0
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "_r{:.1f}".format(RANDOMIZATION_FACTOR),
        log_path=log_dir + "_r{:.1f}".format(RANDOMIZATION_FACTOR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_training_callback,
        verbose=0,
    )

    model.learn(
        total_timesteps=n_steps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=1,
        tb_log_name="Standing_r{:.1f}".format(RANDOMIZATION_FACTOR),
        reset_num_timesteps=False,
        progress_bar=True,
    )

    print(
        " >> COMPLETED TRAINING WITH RANDOMIZATION FACTOR {}".format(
            RANDOMIZATION_FACTOR
        )
    )

    RANDOMIZATION_FACTOR += randomization_increment
    RANDOMIZATION_FACTOR = 1 if RANDOMIZATION_FACTOR > 1 else RANDOMIZATION_FACTOR
