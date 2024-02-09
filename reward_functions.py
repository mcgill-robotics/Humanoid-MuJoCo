from simulation_parameters import *
import numpy as np

def standingRewardFn(data):
    ### REWARD PARAMETERS
    # Velocity The magnitude of the player’s velocity. - 0.1
    HORIZONTAL_VELOCITY_REWARD_WEIGHT = -0.1
    VERTICAL_VELOCITY_REWARD_WEIGHT = -0.05
    # Termination A penalty, equal to −1 if the player is on the ground - 0.5
    GROUNDED_PENALTY_WEIGHT = -0.5
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
    JOINT_TORQUE_PENALTY_WEIGHT = -0.01
    
    
    reward = 0
    
    # Velocity The magnitude of the player's forward velocity. - 0.1
    delta_pos = data.sensor("IMU_vel").data.copy() # LOCAL FRAME
    delta_pos[0] = abs(delta_pos[0])
    delta_pos[1] = abs(delta_pos[1])
    delta_pos[2] = abs(delta_pos[2])
    reward += HORIZONTAL_VELOCITY_REWARD_WEIGHT * np.sqrt(delta_pos[0]**2 + delta_pos[1]**2)
    reward += HORIZONTAL_VELOCITY_REWARD_WEIGHT
    reward += VERTICAL_VELOCITY_REWARD_WEIGHT * delta_pos[2]
    
    # Termination A penalty, equal to −1 if the player is on the ground - 0.5
    isTouchingGround = data.body('humanoid').xpos[2] < 0
    if isTouchingGround: reward += GROUNDED_PENALTY_WEIGHT
      
    # Upright 0 if the robot is upside down or if the tilt angle is greater
        # than 0.4 radians. Increases linearly, and is equal to +1 if the
        # tilt angle is less than 0.2 radians. - 0.02
    IMU_quat = np.quaternion(*data.sensor("IMU_quat").data.copy())
    tilt_angle = abs(2 * np.arccos(IMU_quat.w))
    if tilt_angle < MIN_TILT_FOR_REWARD:
      reward += UPRIGHT_REWARD_WEIGHT
    elif tilt_angle < MAX_TILT_FOR_REWARD:
      tilt_reward = (MAX_TILT_FOR_REWARD - tilt_angle) / (MAX_TILT_FOR_REWARD-MIN_TILT_FOR_REWARD)
      reward += tilt_reward * UPRIGHT_REWARD_WEIGHT
      
    # Joint torque A penalty, equal to the magnitude of the torque measured at
        # the player's knees. This discourages the player from learning
        # gaits which cause high forces on the knees, for example
        # during ground impacts, which can damage a physical robot. - 0.01
    total_joint_torque = 0
    for joint in JOINT_ACTUATOR_NAMES:
      joint_torque = np.linalg.norm(np.array(data.joint(joint).qfrc_constraint + data.joint(joint).qfrc_smooth))
      total_joint_torque = max(total_joint_torque, joint_torque)
    reward += total_joint_torque * JOINT_TORQUE_PENALTY_WEIGHT
    
    return reward