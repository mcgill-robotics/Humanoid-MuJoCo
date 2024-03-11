from simulation.cpu_env import CPUEnv
from simulation import GREEN_SCREEN_SIM_XML_PATH
from reward_functions import *
import torch
from stable_baselines3 import PPO
import os
import cv2


checkpoint = "./data/training_weights/best_model"
num_videos = 9
video_duration = 5 # seconds
video_dir = "data/policy_videos/"

env = CPUEnv(
    xml_path=GREEN_SCREEN_SIM_XML_PATH,
    reward_fn=standingRewardFn,
    randomization_factor=1
)

ppo_agent = PPO.load(
    path=checkpoint,
    env=env,
)

if not os.path.exists(video_dir):
    os.makedirs(video_dir)
video_file_name = os.path.splitext(os.path.basename(checkpoint))[0]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

print("0%", end='\r')
for v in range(num_videos):
    video_writer = cv2.VideoWriter(video_dir + video_file_name + "_" + str(v) + ".mp4", fourcc, 30, (1080, 720))
    done = False
    obs, _ = env.reset()
    while env.data.time < video_duration:
        action, _ = ppo_agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        frame = env.render("rgb_array")
        video_writer.write(frame)
    video_writer.release()
    print("{}%                    ".format(100*(v+1)/num_videos), end='\r')