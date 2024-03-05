from humanoid.simulation.cpu_simulation import CPUSimulation
from humanoid import SIM_XML_PATH
from humanoid.rl.reward_functions import *
import numpy as np
from humanoid.rl.ppo import PPO

checkpoint = "data/trained_weights/Standing/PPO_Standing_0_0_episode_69.pth"

env = CPUSimulation(xml_path=SIM_XML_PATH, reward_fn=standingRewardFn, timestep=0.001, randomization_factor=0)

state_history_length = 5
state_dim = (env.observation_shape[1] + env.action_shape[1]) * state_history_length
action_dim = env.action_shape[1]

ppo_agent = PPO(state_dim, action_dim, 0, 0, 0.99, 1, 0.1, True, 0.001)
ppo_agent.load(checkpoint)

while True:
    env.reset()
    action = env.lastAction
    state = env.getObs()
    state_history = [np.concatenate((state, action), axis=1)] * state_history_length
    done = False
    
    while not done:
        # select action with policy
        action = ppo_agent.select_action(np.concatenate(state_history, axis=1).reshape(-1))
        env.step(action)
        obs = env.getObs()
        state = np.concatenate((obs, action), axis=1)
        state_history.pop(0)
        state_history.append(state)
        reward, done = env.computeReward()
        print(reward[0])
        done = done[0]

        if np.isnan(state).any() or np.isnan(reward).any() or np.isnan(done).any():
            print("ERROR: NaN value in observations. Skipping to next episode.")
            break
        
        env.render()