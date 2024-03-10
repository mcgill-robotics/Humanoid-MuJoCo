from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from reward_functions import *
import torch
from stable_baselines3 import PPO

checkpoint = "./data/training_weights/best_model"

env = CPUEnv(
    xml_path=SIM_XML_PATH,
    reward_fn=standingRewardFn,
    randomization_factor=0
)

ppo_agent = PPO.load(
    path=checkpoint,
    env=env,
)
params = ppo_agent.get_parameters()
params["policy"]["log_std"] = torch.full(env.action_space.shape, -5)
ppo_agent.set_parameters(params)

while True:
    done = False
    obs, _ = env.reset()
    while not done:
        action, _ = ppo_agent.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(reward)
        env.render("human")