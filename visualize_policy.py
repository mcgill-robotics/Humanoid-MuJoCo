from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import *
from stable_baselines3 import PPO

checkpoint = "./data/training_results_r1/best_model"

env = CPUEnv(
    xml_path=SIM_XML_PATH,
    reward_fn=standingRewardFn,
    randomization_factor=1
)

ppo_agent = PPO.load(
    path=checkpoint,
    env=env,
)

while True:
    done = False
    obs, _ = env.reset()
    total_reward = 0
    episode_length = 0
    while not done:
        action, _ = ppo_agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if not done:
            episode_length += 1
            total_reward += reward
        print(reward)
        env.render("human")
    print(" >>> Episode Length {}, Total Reward {}".format(episode_length, total_reward))