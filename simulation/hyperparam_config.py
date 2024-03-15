from torch import nn
import torch
from stable_baselines3.common.torch_layers import FlattenExtractor

# TAKEN FROM RL ZOO HYPERPARAMS FOR Humanoid-v4 mujoco environment
default_hyperparams = dict(
    policy = 'MlpPolicy',
    n_timesteps = 1e7,
    batch_size = 256,
    n_steps = 512,
    gamma = 0.95,
    learning_rate = 3.56987e-05,
    ent_coef = 0.00238306,
    clip_range = 0.3,
    n_epochs = 5,
    gae_lambda = 0.9,
    max_grad_norm = 2,
    vf_coef = 0.431892,
    policy_kwargs = dict(
                        log_std_init = -2,
                        ortho_init = False,
                        activation_fn = nn.ReLU,
                        net_arch = dict(pi=[256, 256], vf=[256, 256])
                    )
    )


hyperparams = {
    "CPUHumanoid": default_hyperparams,
    "GPUHumanoid": default_hyperparams
}