from simulation.cpu_env import CPUEnv
from simulation import GREEN_SCREEN_SIM_XML_PATH
from simulation.reward_functions import *
from stable_baselines3 import SAC
import os
import cv2


MODEL_TYPE = SAC
LOG_NAME = "SAC"
RANDOMIZATION_FACTOR = 0
CKPT_NAME = "best_model"

eval_dir = "./data/{}/training_results_r{}/".format(LOG_NAME, RANDOMIZATION_FACTOR)
num_videos = 9
video_duration = 5  # seconds

env = CPUEnv(
    xml_path=GREEN_SCREEN_SIM_XML_PATH,
    reward_fn=controlInputRewardFn,
    randomization_factor=RANDOMIZATION_FACTOR,
    enable_rendering=True,
)

agent = MODEL_TYPE.load(
    path=eval_dir + CKPT_NAME,
    env=env,
)

if not os.path.exists(eval_dir + "policy_videos/"):
    os.makedirs(eval_dir + "policy_videos/")
video_file_name = os.path.splitext(os.path.basename(eval_dir + CKPT_NAME))[0]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

print("0%", end="\r")
for v in range(num_videos):
    video_writer = cv2.VideoWriter(
        eval_dir + "policy_videos/" + video_file_name + "_" + str(v) + ".mp4",
        fourcc,
        30,
        (1080, 720),
    )
    done = False
    obs, _ = env.reset()
    while env.data.time < video_duration:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        frame = env.render("rgb_array")
        video_writer.write(frame)
    video_writer.release()
    print("{}%                    ".format(100 * (v + 1) / num_videos), end="\r")
