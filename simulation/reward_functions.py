from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation

# REWARD INFO FROM https://arxiv.org/pdf/2304.13653.pdf
    
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
# CUSTOM: a penalty for how much the joint control differs from previous joint control, to reward "smoother" motions - 0.1

def standingRewardFn(velocity, z_pos, torso_quat, joint_torques, ctrl_change, isSelfColliding):
    # ALL IN NWU
    # velocity in m/s
    # z_pos in meters
    # quat is quaternion of torso (wxyz)
    # joint torques in N m
    
    ### REWARD PARAMETERS
    # Velocity The magnitude of the player’s velocity. - 0.1
    HORIZONTAL_VELOCITY_REWARD_WEIGHT = -0.1
    VERTICAL_VELOCITY_REWARD_WEIGHT = -0.1
    # Termination A penalty, equal to −1 if the player is on the ground - 0.5
    # MODIFICATION: +0.5 reward for torso being above or at Z=0, linearly interpolated to -0.5 if the torso is under -0.4
    GROUNDED_PENALTY_WEIGHT = -0.5
    NOT_GROUNDED_REWARD_WEIGHT = 0.5 # added this so that staying not grounded is rewarded (rather than terminating quickly to avoid future penalties)
    MIN_Z_BEFORE_GROUNDED = -0.3
    MIN_Z_FOR_REWARD = -0.2
    MAX_Z = -0.1
    # Joint torque A penalty, equal to the magnitude of the torque measured at
        # the player’s knees. This discourages the player from learning
        # gaits which cause high forces on the knees, for example
        # during ground impacts, which can damage a physical robot. - 0.01
    MAX_JOINT_TORQUE = 1.5
    JOINT_TORQUE_PENALTY_WEIGHT = -0.1 / 16 # divide by N since there are N joints and we consider the sum of joint torques
    # penalty term to minimize the time integral of torque peaks
    # (thresholded above 5 N m)
    # CUSTOM: a penalty for how much the joint control differs from previous joint control, to reward "smoother" motions (std is 0 to 2)
    CONTROL_STD_PENALTY = -0.01
    # Upright: a reward which scales linearly with how close the torso is to perfectly upright
    UPRIGHT_REWARD_MAX_PENALTY_TILT = 0.8 # CUSTOM -> paper does not penalize high tilts
    UPRIGHT_REWARD_MAX_TILT_FOR_REWARD = 0.4
    UPRIGHT_REWARD_MIN_TILT = 0.2
    UPRIGHT_MAX_REWARD = 0.2 # CUSTOM -> paper has it at 0.02
    UPRIGHT_MAX_PENALTY = -0.1 # CUSTOM -> paper does not penalize high tilts
    # CUSTOM: add a constant offset to the reward
    CONSTANT_REWARD_OFFSET = 0.0
    # CUSTOM: penalize self-collisions
    TERMINATE_ON_SELF_COLLISION = False
    SELF_COLLISION_PENALTY = -0.1
    
    ### COMPUTE REWARD
    reward = 0
    
    # Velocity
    abs_velocity = jp.abs(velocity)
    reward += HORIZONTAL_VELOCITY_REWARD_WEIGHT * jp.linalg.norm(abs_velocity[0:2])
    reward += VERTICAL_VELOCITY_REWARD_WEIGHT * abs_velocity[2]
    
    # Torso height reward
    z_pos_penalty = jp.interp(z_pos, jp.array([MIN_Z_FOR_REWARD, MAX_Z]), jp.array([GROUNDED_PENALTY_WEIGHT, NOT_GROUNDED_REWARD_WEIGHT]))
    reward += z_pos_penalty
    
    # Termination
    isTouchingGround = jp.where(z_pos > MIN_Z_BEFORE_GROUNDED, False, True)

    # Joint torque
    thresholded_joint_torques = jp.clip(jp.abs(joint_torques), 0.0, MAX_JOINT_TORQUE) / MAX_JOINT_TORQUE
    # thresholded_joint_torques = jp.minimum(jp.abs(joint_torques), jp.full(joint_torques.shape, 5.0))
    total_joint_torque = jp.sum(thresholded_joint_torques)
    reward += total_joint_torque * JOINT_TORQUE_PENALTY_WEIGHT
    
    # Control change
    control_std = jp.std(ctrl_change)
    reward += CONTROL_STD_PENALTY * control_std
    
    reward += CONSTANT_REWARD_OFFSET
    
    # Upright
    tilt_vector = Rotation.from_quat([torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]).inv().apply(jp.array([0,0,1]))
    tilt_amt = jp.arcsin(jp.max(jp.abs(tilt_vector[0:2])))
    tilt_amt = jp.where(tilt_vector[2] <= 0, UPRIGHT_REWARD_MAX_PENALTY_TILT, tilt_amt)
    tilt_reward = jp.interp(tilt_amt, jp.array([UPRIGHT_REWARD_MIN_TILT, UPRIGHT_REWARD_MAX_TILT_FOR_REWARD, UPRIGHT_REWARD_MAX_PENALTY_TILT]), jp.array([UPRIGHT_MAX_REWARD, 0, UPRIGHT_MAX_PENALTY]))
    reward += tilt_reward

    if TERMINATE_ON_SELF_COLLISION: reward = jp.where(isSelfColliding, 0, reward)
    else: reward = jp.where(isSelfColliding, reward + SELF_COLLISION_PENALTY, reward)
    
    reward = jp.where(reward < 0, 0, reward)
    reward = jp.where(isTouchingGround, 0, reward)
    
    terminal = jp.where(isTouchingGround, True, False)
    if TERMINATE_ON_SELF_COLLISION: terminal = jp.where(isSelfColliding, True, terminal)
    
    return reward, terminal