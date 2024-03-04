from humanoid.simulation.reward_functions import *
from humanoid.simulation.gpu_batch_simulation import GPUBatchSimulation
from humanoid.simulation.cpu_simulation import CPUSimulation
from humanoid import SIM_XML_PATH
import os

# Set environment variable to disable rendering
os.environ["RENDER_SIM"] = "False"

# STATE INFO FROM https://arxiv.org/pdf/2304.13653.pdf

# STATE
    # joint positions     5 · 20          Joint positions in radians (stacked last 5 timesteps)
    # linear acceleration 5 · 3           Linear acceleration from IMU (stacked)
    # angular velocity    5 · 3           Angular velocity (roll, pitch, yaw) from IMU (stacked)
    # foot pressure       5 · 8           Pressure values from foot sensors (stacked)
    # gravity             5 · 3           Gravity direction, derived from angular velocity using Madgwick filter (stacked)
    # agent velocity      5 · 2           X and Y velocity of robot torso (stacked)
    # previous action     5 · 20          Action filter state (stacked)    


####### initialize environment hyperparameters ######
has_continuous_action_space = True  # continuous action space; else discrete

print_freq = 1        # print avg reward in the interval (in num episodes)
log_freq = 10           # log avg reward in the interval (in num episodes)
save_model_freq = 100  # save model frequency (in num episodes)

action_std = 0.025                   # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.001        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.0001                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = 5000  # action_std decay frequency (in num timesteps)

state_history_length = 5 # how many iterations of the history of state observations is included in the current state observation
physics_steps_per_control_step = 5

max_reward_for_randomization = 0.3 # if average reward of an episode is greater than this, increase randomization of environment
randomization_increment = 0.1

#####################################################

env = GPUBatchSimulation(count=64,
                        xml_path=SIM_XML_PATH,
                        reward_fn=standingRewardFn,
                        physics_steps_per_control_step=physics_steps_per_control_step,
                        timestep=0.001,
                        randomization_factor=0)

# env = CPUSimulation(xml_path=SIM_XML_PATH, reward_fn=standingRewardFn, timestep=0.005, randomization_factor=0)

max_ep_len = int(5.0 / (physics_steps_per_control_step * env.timestep))                   # max timesteps in one episode
max_training_timesteps = max_ep_len * 200000000   # break training loop if timeteps > max_training_timesteps

env_name = "Standing"

print("training environment name : " + env_name)

# state space dimension
state_dim = (env.observation_shape[1] + env.action_shape[1]) * state_history_length

# action space dimension
action_dim = env.action_shape[1]
#####################################################

# HYPERPARAMETER INFO FROM https://arxiv.org/pdf/1910.10620.pdf#page=3&zoom=100,433,952
################ PPO hyperparameters ################
update_timesteps = 32 # int(4096 / env.count)    # update policy every n timesteps
K_epochs = 10               # update policy for K epochs in one PPO update
batch_size = 64

eps_clip = 0.1          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0001       # learning rate for actor network
lr_critic = 0.0001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
#####################################################