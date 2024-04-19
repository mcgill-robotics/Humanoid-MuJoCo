from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import *
from stable_baselines3 import PPO, SAC, TD3


MODEL_TYPE = SAC  # TD3 # SAC # PPO
RANDOMIZATION_FACTOR = 0.0
LOG_NAME = "SAC"
CKPT_NAME = "checkpoint_998400_steps"


checkpoint = "./data/{}/training_results_r{}/{}".format(
    LOG_NAME,
    RANDOMIZATION_FACTOR,
    CKPT_NAME,
)
env = CPUEnv(
    xml_path=SIM_XML_PATH,
    reward_fn=controlInputRewardFn,
    randomization_factor=0.0,  # ,RANDOMIZATION_FACTOR,
)
agent = MODEL_TYPE.load(
    path=checkpoint,
    env=env,
)

while True:
    done = False
    obs, _ = env.reset()
    total_reward = 0
    episode_length = 0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if not done:
            episode_length += 1
            total_reward += reward
            print(reward)
        env.render("human")
    print(
        " >>> Episode Length {}, Total Reward {}".format(episode_length, total_reward)
    )
