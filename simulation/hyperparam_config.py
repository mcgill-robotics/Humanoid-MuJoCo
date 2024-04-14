from torch import nn

# TAKEN FROM RL ZOO HYPERPARAMS FOR Humanoid-v4 mujoco environment

hyperparams = {
    "CPUHumanoid": dict(
        policy="MlpPolicy",
        normalize=False,
        n_timesteps=150000,  # IRRELEVANT, OVERWRITTEN IN TUNING ARGS
        n_envs=64,
        batch_size=64,
        n_steps=64,
        gamma=0.99,
        learning_rate=0.00001,
        ent_coef=0.002,
        clip_range=0.3,
        n_epochs=10,
        gae_lambda=0.95,
        max_grad_norm=2,
        vf_coef=0.431892,
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
            use_expln=True,
        ),
    )
}
