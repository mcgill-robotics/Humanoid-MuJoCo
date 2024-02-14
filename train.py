import os
from datetime import datetime
import torch
import numpy as np
from humanoid.rl.ppo import PPO
from humanoid.rl.training_parameters import *

# STATE INFO FROM https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=HlRhFs_d3WLP

# STATE
    # joint positions     5 · 20          Joint positions in radians (stacked last 5 timesteps)
    # linear acceleration 5 · 3           Linear acceleration from IMU (stacked)
    # angular velocity    5 · 3           Angular velocity (roll, pitch, yaw) from IMU (stacked)
    # foot pressure       5 · 8           Pressure values from foot sensors (stacked)
    # gravity             5 · 3           Gravity direction, derived from angular velocity using Madgwick filter (stacked)
    # agent velocity      5 · 2           X and Y velocity of robot torso (stacked)
    # previous action     5 · 20          Action filter state (stacked)    
    
################################### Training ###################################
def train():
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "data/PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_avg_reward = 0
    print_running_episodes = 0

    log_running_avg_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    try: # wrap in try-except so we can stop training with Ctrl+C and not lose the trained model
        while time_step <= max_training_timesteps:

            # TODO -> implement increasing randomization factor as time goes on (when avg. reward stagnates or something)
            current_avg_ep_reward = 0
            
            env.reset()
            action = env.lastAction
            state = env.getObs()
            state_history = [np.concatenate((state, action), axis=1)] * state_history_length
            
            for t in range(1, max_ep_len+1):

                # select action with policy
                action = ppo_agent.select_action(np.concatenate(state_history, axis=1))
                env.step(action)
                state = np.concatenate((env.getObs(), action), axis=1)
                state_history.pop(0)
                state_history.append(state)
                reward, done = env.computeReward()

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_avg_ep_reward += np.mean(reward)

                # update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # log in logging file
                if time_step % log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_avg_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_avg_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_avg_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_avg_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if np.all(done):
                    break

            print_running_avg_reward += current_avg_ep_reward
            print_running_episodes += 1

            log_running_avg_reward += current_avg_ep_reward
            log_running_episodes += 1

            i_episode += 1
    except:
        print("\nUser ended training early. Saving current state of model.\n")
        
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        ppo_agent.save(checkpoint_path)
        print("model saved")
        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        print("--------------------------------------------------------------------------------------------")

    log_f.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
    
    
    
    
    
    
    