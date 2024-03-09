import os
from reward_functions import *
from simulation.gpu_vec_env import GPUVecEnv
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

###########################
##  TRAINING PARAMETERS  ##
###########################

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

eval_log_dir = "data/training_weights"
log_dir = "data/training_logs"
os.makedirs(eval_log_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

##########################
##    HYPERPARAMETERS   ##
##########################

NUM_ENVS = 512
N_EVAL_EPISODES = 20
POLICY_ITERATIONS = 1000

env = VecMonitor(GPUVecEnv(
    num_envs=NUM_ENVS,
    xml_path=SIM_XML_PATH,
    reward_fn=standingRewardFn,
    randomization_factor=0
))

print("Initializing environment...      ", end='')
env.reset()
print("Done")
print("Stepping environment...          ", end='')
env.step(None)
print("Done")

print("Beginning training.\n")

# env = VecMonitor(DummyVecEnv([ lambda : CPUEnv(
#                                 xml_path=SIM_XML_PATH,
#                                 reward_fn=standingRewardFn,
#                                 randomization_factor=0
#                             )]*NUM_ENVS))

eval_env = VecMonitor(DummyVecEnv([ lambda : CPUEnv(
                                    xml_path=SIM_XML_PATH,
                                    reward_fn=standingRewardFn,
                                    randomization_factor=0
                                )] * N_EVAL_EPISODES))

policy = lambda : ActorCriticPolicy(
    observation_space = env.observation_space,
    action_space = env.action_space,
    lr_schedule = lambda lr : lr,
    net_arch = [256,256,256],
    activation_fn = nn.ELU,
    ortho_init = True,
    use_sde = False,
    log_std_init = 0.0,
    full_std = False,
    use_expln = False,
    squash_output = False,
    features_extractor_class = FlattenExtractor,
    features_extractor_kwargs = None,
    share_features_extractor = True,
    normalize_images = False,
    optimizer_class = torch.optim.Adam,
    optimizer_kwargs = None
)

model = PPO(
    policy = "MlpPolicy",
    env = env,
    learning_rate = 0.00001,
    n_steps = 12,
    batch_size = 64,
    n_epochs = 10,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    clip_range_vf = None,
    normalize_advantage = True,
    ent_coef = 0.1,
    vf_coef = 1.0,
    max_grad_norm = 0.5,
    use_sde = False,
    sde_sample_freq = -1,
    rollout_buffer_class = None,
    rollout_buffer_kwargs = None,
    target_kl = None,
    stats_window_size = 100,
    tensorboard_log = None,
    policy_kwargs = None,
    verbose = 1,
    seed = None,
    device = "auto",
    _init_setup_model = True,
)

##########################
##    TRAINING LOGIC    ##
##########################

eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=log_dir, eval_freq=model.n_steps,
                              n_eval_episodes=N_EVAL_EPISODES, deterministic=False,
                              render=False)

model.learn(total_timesteps=POLICY_ITERATIONS * model.n_steps,
            callback=eval_callback,
            log_interval = 1,
            tb_log_name = "PPOStanding",
            reset_num_timesteps = True,
            progress_bar = True)

