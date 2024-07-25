from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from stable_baselines3 import SAC


MODEL_TYPE = SAC
RANDOMIZATION_FACTOR = 0.1
CKPT = "data\SAC_CPU/ckpt_18600000_steps.zip"

# env = GPUVecEnv(
#     num_envs=1,
#     xml_path=SIM_XML_PATH,
#     reward_fn=SELECTED_REWARD_FUNCTION,
#     randomization_factor=RANDOMIZATION_FACTOR,
#     enable_rendering=True,
# )
env = CPUEnv(
    xml_path=SIM_XML_PATH,
    reward_fn=SELECTED_REWARD_FUNCTION,
    randomization_factor=RANDOMIZATION_FACTOR,
    enable_rendering=True,
)
agent = MODEL_TYPE.load(
    path=CKPT,
    env=env,
)

while True:
    done = False
    total_reward = 0
    episode_length = 0
    try:
        obs, _ = env.reset()
    except:
        obs = env.reset()
    try:
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            # print(action)
            try:
                obs, reward, done, _, _ = env.step(action)
            except:
                obs, reward, done, _ = env.step(action, reset_if_terminal=False)
            if not done:
                episode_length += 1
                total_reward += reward
                # print(reward)
            env.render("human")
    except KeyboardInterrupt:
        print(
            " >>> Episode Length {}, Total Reward {}".format(
                episode_length, total_reward
            )
        )
