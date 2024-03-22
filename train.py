import os
from simulation.reward_functions import *
from simulation.gpu_vec_env import GPUVecEnv
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.torch_layers import FlattenExtractor
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
# Trial 44 finished with value: 452.2629128 and parameters: {'batch_size': 32, 'n_steps': 2048, 'gamma': 0.95, 'learning_rate': 0.00021243303991845677, 'ent_coef': 0.0001223532933316859, 'clip_range': 0.1, 'n_epochs': 5, 'gae_lambda': 0.99, 'max_grad_norm': 0.7, 'vf_coef': 0.6215142166966646, 'net_arch': 'medium', 'log_std_init': -2.9863975746713614, 'sde_sample_freq': -1, 'ortho_init': True, 'activation_fn': 'elu', 'lr_schedule': 'linear'}. Best is trial 44 with value: 452.2629128.

NUM_ENVS = 32
N_EVAL_EPISODES = 10
POLICY_ITERATIONS = 1000
POLICY_UPDATE_TIMESTEPS = 2048
TOTAL_TIMESTEPS = POLICY_ITERATIONS * NUM_ENVS * POLICY_UPDATE_TIMESTEPS
CHECKPOINT = None
EVAL_FREQ = POLICY_UPDATE_TIMESTEPS
CHECKPOINT_FREQ = POLICY_UPDATE_TIMESTEPS * 5
RANDOMIZATION_INCREMENT = 0.25
RANDOMIZATION_FACTOR = 1 # starts at this, increments whenever training is successful
SUCCESSFUL_TRAINING_REWARD_THRESHOLD = 950

# env = VecMonitor(GPUVecEnv(
#     num_envs=NUM_ENVS,
#     xml_path=SIM_XML_PATH,
#     reward_fn=standingRewardFn,
#     randomization_factor=RANDOMIZATION_FACTOR
# ))

# print("\nInitializing environment...      ", end='')
# env.reset()
# env.step(None)
# print("Done")

env = VecMonitor(DummyVecEnv([ lambda : CPUEnv(
                                    xml_path=SIM_XML_PATH,
                                    reward_fn=standingRewardFn,
                                    randomization_factor=RANDOMIZATION_FACTOR
                                )] * NUM_ENVS))

eval_env = VecMonitor(DummyVecEnv([ lambda : CPUEnv(
                                    xml_path=SIM_XML_PATH,
                                    reward_fn=standingRewardFn,
                                    randomization_factor=RANDOMIZATION_FACTOR
                                )]))

print("\nBeginning training.\n")


if CHECKPOINT is None:
    policy_args = {
        # "lr_schedule": lambda progress : progress * 0.0002,
        "net_arch": dict(pi=[256,256], vf=[256,256]),
        "activation_fn": nn.Tanh,
        "ortho_init": True,
        "log_std_init": -3,
        "full_std": False,
        "use_expln": False,
        "squash_output": False,
        "features_extractor_class": FlattenExtractor,
        "features_extractor_kwargs": None,
        "share_features_extractor": True,
        "normalize_images": False,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": None
    }

    model = PPO(
        policy = "MlpPolicy",
        env = env,
        learning_rate = 0.0002,
        n_steps = POLICY_UPDATE_TIMESTEPS,
        batch_size = 32,
        n_epochs = 5,
        gamma = 0.95,
        gae_lambda = 0.99,
        clip_range = 0.1,
        clip_range_vf = None,
        normalize_advantage = True,
        ent_coef = 0.0001,
        vf_coef = 0.625,
        max_grad_norm = 0.7,
        use_sde = True,
        sde_sample_freq = -1,
        rollout_buffer_class = None,
        rollout_buffer_kwargs = None,
        target_kl = None,
        stats_window_size = 100,
        tensorboard_log = None,
        policy_kwargs = policy_args,
        verbose = 1,
        seed = None,
        device = "auto",
        _init_setup_model = True,
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

    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ,
                                            save_path=log_dir + "_r{}".format(RANDOMIZATION_FACTOR),
                                            name_prefix="checkpoint",
                                            verbose=1)
    
    stop_training_callback = StopTrainingOnRewardThreshold(reward_threshold=SUCCESSFUL_TRAINING_REWARD_THRESHOLD, verbose=1)

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir + "_r{}".format(RANDOMIZATION_FACTOR),
                                log_path=log_dir + "_r{}".format(RANDOMIZATION_FACTOR), eval_freq=EVAL_FREQ,
                                n_eval_episodes=N_EVAL_EPISODES, deterministic=True,
                                render=False, callback_on_new_best=stop_training_callback, verbose=0)

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=[checkpoint_callback, eval_callback],
                log_interval = 1,
                tb_log_name = "Standing_r{}".format(RANDOMIZATION_FACTOR),
                reset_num_timesteps = False,
                progress_bar = True)
    
    print(" >> COMPLETED TRAINING WITH RANDOMIZATION FACTOR {}".format(RANDOMIZATION_FACTOR))
    
    if RANDOMIZATION_FACTOR == 1: break
    
    RANDOMIZATION_FACTOR += RANDOMIZATION_INCREMENT
    RANDOMIZATION_FACTOR = 1 if RANDOMIZATION_FACTOR > 1 else RANDOMIZATION_FACTOR