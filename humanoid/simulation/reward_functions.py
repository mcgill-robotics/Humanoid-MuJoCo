from .simulation_parameters import *
from jax import numpy as jp

# REWARD INFO FROM https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=HlRhFs_d3WLP
    
# REWARD
# Velocity The magnitude of the player’s forward velocity. - 0.1
# Termination A penalty, equal to −1 if the player is on the ground - 0.5
# Upright 0 if the robot is upside down or if the tilt angle is greater
    # than 0.4 radians. Increases linearly, and is equal to +1 if the
    # tilt angle is less than 0.2 radians. - 0.02
# Joint torque A penalty, equal to the magnitude of the torque measured at
    # the player’s knees. This discourages the player from learning
    # gaits which cause high forces on the knees, for example
    # during ground impacts, which can damage a physical robot. - 0.01

def standingRewardFn(velocity, z_pos, quat, joint_torques):
    ### REWARD PARAMETERS
    # Velocity The magnitude of the player’s velocity. - 0.1
    HORIZONTAL_VELOCITY_REWARD_WEIGHT = -0.1
    VERTICAL_VELOCITY_REWARD_WEIGHT = -0.05
    # Termination A penalty, equal to −1 if the player is on the ground - 0.5
    GROUNDED_PENALTY_WEIGHT = -0.5
    MIN_Z_BEFORE_GROUNDED = -0.35
    # Upright 0 if the robot is upside down or if the tilt angle is greater
        # than 0.4 radians. Increases linearly, and is equal to +1 if the
        # tilt angle is less than 0.2 radians. - 0.02
    UPRIGHT_REWARD_WEIGHT = 0.02
    MIN_TILT_FOR_REWARD = 0.2
    MAX_TILT_FOR_REWARD = 0.4
    # Joint torque A penalty, equal to the magnitude of the torque measured at
        # the player’s knees. This discourages the player from learning
        # gaits which cause high forces on the knees, for example
        # during ground impacts, which can damage a physical robot. - 0.01
    JOINT_TORQUE_PENALTY_WEIGHT = -0.01 / 20 # divide by 20 since there are 20 joints and we consider the sum of joint torques
    
    ### COMPUTE REWARD
    reward = 0
    
    # Velocity
    abs_velocity = jp.abs(velocity)
    reward += HORIZONTAL_VELOCITY_REWARD_WEIGHT * jp.linalg.norm(abs_velocity[0:2])
    reward += VERTICAL_VELOCITY_REWARD_WEIGHT * abs_velocity[2]
    
    # Termination
    grounded_penalty = jp.where(z_pos > MIN_Z_BEFORE_GROUNDED, 0, GROUNDED_PENALTY_WEIGHT)
    reward += grounded_penalty
    isTouchingGround = jp.where(z_pos > MIN_Z_BEFORE_GROUNDED, False, True)
      
    # Upright
    tilt_angle = jp.abs(2 * jp.arccos(quat[0]))
    tilt_reward = jp.interp(tilt_angle, jp.array([MIN_TILT_FOR_REWARD, MAX_TILT_FOR_REWARD]), jp.array([0, UPRIGHT_REWARD_WEIGHT]))
    reward += tilt_reward
      
    # Joint torque
    total_joint_torque = jp.sum(jp.abs(joint_torques))
    reward += total_joint_torque * JOINT_TORQUE_PENALTY_WEIGHT
    
    return reward, isTouchingGround