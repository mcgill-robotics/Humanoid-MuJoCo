from humanoid.simulation.cpu_simulation import CPUSimulation
from humanoid import GREEN_SCREEN_SIM_XML_PATH
from humanoid.rl.reward_functions import *
import numpy as np
from humanoid.rl.ppo import PPO
import os
import cv2


checkpoint = "data/trained_weights/Standing/PPO_Standing_0_0_episode_700_64.pth"
num_videos = 9
video_duration = 5 # seconds


video_dir = "data/policy_videos/"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
video_file_name = os.path.splitext(os.path.basename(checkpoint))[0]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

env = CPUSimulation(xml_path=GREEN_SCREEN_SIM_XML_PATH, reward_fn=standingRewardFn, timestep=0.001, randomization_factor=1)

state_history_length = 5
state_dim = (env.observation_shape[1] + env.action_shape[1]) * state_history_length
action_dim = env.action_shape[1]

K_epochs = 10              # update policy for K epochs in one PPO update
eps_clip = 0.1          # clip parameter for PPO
gamma = 0.99            # discount factor
lr_actor = 0.0001       # learning rate for actor network
lr_critic = 0.0001  
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, 0.001)
ppo_agent.load(checkpoint)

for v in range(num_videos):
    env.reset()
    action = env.lastAction
    state = env.getObs()
    state_history = [np.concatenate((state, action), axis=1)] * state_history_length
    done = False
    
    video_writer = cv2.VideoWriter(video_dir + video_file_name + "_" + str(v) + ".mp4", fourcc, 30, (1080, 720))
    
    while env.data.time < video_duration:
        # select action with policy
        action = ppo_agent.select_action(np.concatenate(state_history, axis=1).reshape(-1))
        env.step(action)
        obs = env.getObs()
        state = np.concatenate((obs, action), axis=1)
        state_history.pop(0)
        state_history.append(state)
        reward, done = env.computeReward()

        if np.isnan(state).any() or np.isnan(reward).any() or np.isnan(done).any():
            print("ERROR: NaN value in observations. Skipping to next episode.")
            break
        
        frame = env.render(display=False)
        video_writer.write(frame)
        
    video_writer.release()
    