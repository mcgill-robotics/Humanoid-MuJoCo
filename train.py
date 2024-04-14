import os
from simulation.reward_functions import *
from simulation.gpu_vec_env import GPUVecEnv
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, VecNormalize

###########################
##  TRAINING PARAMETERS  ##
###########################

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

log_dir = "data/training_results"

##########################
##    HYPERPARAMETERS   ##
##########################

# FROM RL ZOO 3 HYPERPARAMETER TUNING
# Trial 19 finished with value: 262.5614284 and parameters: {'batch_size': 256, 'n_steps': 1024, 'gamma': 0.98, 'learning_rate': 0.0006905843913061805, 'ent_coef': 0.022694858251377927, 'clip_range': 0.1, 'n_epochs': 1, 'gae_lambda': 0.8, 'max_grad_norm': 0.7, 'vf_coef': 0.7445807875710113, 'net_arch': 'large', 'log_std_init': -0.5482045338158068, 'sde_sample_freq': 128, 'ortho_init': True, 'activation_fn': 'tanh'}. Best is trial 19 with value: 262.5614284.


NUM_ENVS = 64
SIMULATE_ON_GPU = False
N_EVAL_EPISODES = 10
POLICY_ITERATIONS = 1000
POLICY_UPDATE_TIMESTEPS = 2048
TOTAL_TIMESTEPS = POLICY_ITERATIONS * NUM_ENVS * POLICY_UPDATE_TIMESTEPS
CHECKPOINT = None
EVAL_FREQ = POLICY_UPDATE_TIMESTEPS
CHECKPOINT_FREQ = POLICY_UPDATE_TIMESTEPS * 5
RANDOMIZATION_INCREMENT = 0.1
RANDOMIZATION_FACTOR = 0  # starts at this, increments whenever training is successful
SUCCESSFUL_TRAINING_REWARD_THRESHOLD = 750
NORMALIZE = False  # whether or not to wrap env in a VecNormalize wrapper

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
            )
        ]
    )
)

if NORMALIZE:
    env = VecNormalize(env)
    eval_env = VecNormalize(eval_env)

print("\nBeginning training.\n")


if CHECKPOINT is None:
    policy_args = {
        "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
        "activation_fn": nn.Tanh,
        "ortho_init": True,
        "log_std_init": 0.0,
        "full_std": True,
        "use_expln": False,
        "squash_output": False,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": None,
    }

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=POLICY_UPDATE_TIMESTEPS,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        normalize_advantage=True,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        policy_kwargs=policy_args,
        verbose=1,
    )
else:
    model = PPO.load(
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
        verbose=1,
    )

    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=SUCCESSFUL_TRAINING_REWARD_THRESHOLD, verbose=1
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
