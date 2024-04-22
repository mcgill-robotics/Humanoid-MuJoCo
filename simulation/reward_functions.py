from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation

# REWARD INFO FROM https://arxiv.org/pdf/2307.10142.pdf

# can i assume constant time between actions? TODO 0.01 s

def rewardControlInputFn(torso_local_velocity,
                        local_ang_vel,
                        control_input_velocity,
                        control_input_yaw,
                        torso_z_pos,
                        joint_torques,
                        action_to_take,
                        lastAction,
                        lastLastAction,
                        self_collision,
                        local_gravity_vector,
                        joint_qpos_idx):

    # Velocities Rewards
    lin_vel_reward = track_lin_vel_r(control_input_velocity, torso_local_velocity)
    ang_vel_reward = track_ang_vel_r(control_input_yaw, local_ang_vel)
    # Torque Rewards
    torque_reward = torques_r(joint_torques)
    torque_limit_reward = torques_limits_r(joint_torques)
    joint_limit_reward = joint_limits_r(joint_torques)
    # Termination
    isTerminated = termination_r(torso_local_velocity, torso_z_pos, local_ang_vel, self_collision, local_gravity_vector)
    termination_reward = int(isTerminated)
    # Height 
    height_reward = height_r(torso_z_pos)
    # Orientation
    orientation_reward = orientation_r(local_gravity_vector)
    # Change in action for smoothness
    first_order_reward = first_order_action_r(action_to_take, lastAction)
    second_order_reward = second_order_action_r(action_to_take, lastAction, lastLastAction)
    # Joint Regularization
    joint_reg_reward = joint_regularization_r(joint_qpos_idx)
    # Ankle Regularization
    ankle_reg_reward = ankle_regularization_r(joint_qpos_idx)
    
    final_reward =  10.0 * lin_vel_reward +\
                    5.0 * ang_vel_reward +\
                    -1e-4 * torque_reward +\
                    -0.01 * torque_limit_reward +\
                    -10.0 * joint_limit_reward +\
                    -100.0 * termination_reward +\
                    2.0 * height_reward +\
                    5.0 * orientation_reward +\
                    -1e-3 * first_order_reward +\
                    -1e-4 * second_order_reward +\
                    1.0 * joint_reg_reward +\
                    1.0 * ankle_reg_reward
    
    return final_reward[0], isTerminated
    
    
def joint_regularization_r(joint_qpos_idx):
    # Reward joint poses and symmetry
    DOF_POS = 1.0
    error = 0.
    # Yaw joints regularization around 0
    # left hip yaw
    error += sqrdexp(
        (joint_qpos_idx[4]) / DOF_POS)
    # right hip yaw
    error += sqrdexp(
        (joint_qpos_idx[9]) / DOF_POS)
    # torso yaw
    error += sqrdexp(
        (joint_qpos_idx[14]) / DOF_POS)
    
    # Roll joint symmetry, left and right hip roll
    error += sqrdexp(
        (joint_qpos_idx[5] - joint_qpos_idx[10])
        / DOF_POS)
    
    # Pitch joint symmetry, left and right knee pitch
    error += sqrdexp(
        (joint_qpos_idx[7] + joint_qpos_idx[12])
        / DOF_POS)
    # left and right hip pitch
    error += sqrdexp(
        (joint_qpos_idx[6] + joint_qpos_idx[11])
        / DOF_POS)
    # left and right shoulder pitch
    error += sqrdexp(
        (joint_qpos_idx[0] + joint_qpos_idx[2])
        / DOF_POS)
    # left and right elbow pitch
    error += sqrdexp(
        (joint_qpos_idx[1] + joint_qpos_idx[3])
        / DOF_POS)
    
    return error/8

def ankle_regularization_r(joint_qpos_idx):
    DOF_POS = 1.0
    # Ankle joint regularization around 0
    error = 0
    error += sqrdexp(
        (joint_qpos_idx[9]) / DOF_POS)
    # right hip yaw
    error += sqrdexp(
        (joint_qpos_idx[13]) / DOF_POS)
    return error
    
def first_order_action_r(action_to_take, lastAction):
    FIXED_TIME = 0.01
    # Penalize changes in actions
    diff = jp.array(action_to_take - lastAction) / FIXED_TIME
    return jp.sum(jp.square(diff))

def second_order_action_r(action_to_take, lastAction, lastLastAction):
    FIXED_TIME = 0.01
    # Penalize changes in actions
    diff = jp.array(action_to_take - 2*lastAction + lastLastAction) / FIXED_TIME
    return jp.sum(jp.square(diff))

def orientation_r(local_gravity_vector):
    # Reward tracking upright orientation
    SIGMA = 0.5
    error = jp.sum(jp.square(local_gravity_vector[:2]))
    return jp.exp(-error/SIGMA)

def height_r(torso_z_pos):
    # Reward tracking specified base height
    HEIGHT_TARGET = -0.1
    SIGMA = 0.5
    error = (torso_z_pos - HEIGHT_TARGET)
    return jp.exp(-jp.square(error)/SIGMA)

def termination_r(torso_local_velocity, torso_z_pos, local_ang_vel, self_collision, local_gravity_vector):
    MIN_Z_BEFORE_GROUNDED = -0.3
    MAX_LOCAL_VELOCITY_NORM = 10
    MAX_ANGULAR_VELOCITY_NORM = 5
    
    if jp.linalg.norm(torso_local_velocity) >= MAX_LOCAL_VELOCITY_NORM or\
        jp.linalg.norm(local_ang_vel) >= MAX_ANGULAR_VELOCITY_NORM or \
        local_gravity_vector[0]>=0.7 or local_gravity_vector[1] >=0.7 or\
        self_collision or \
        torso_z_pos < MIN_Z_BEFORE_GROUNDED:
            return True
    return False
    
def joint_limits_r(joint_torques):
    MAX_JOINT_TORQUE = 1.5
    BETA_Q = 0.8
    return jp.max(jp.abs(joint_torques)- BETA_Q* MAX_JOINT_TORQUE, 0)

def torques_limits_r(joint_torques):
    MAX_JOINT_TORQUE = 1.5
    BETA_T = 0.8
    return jp.max(jp.abs(joint_torques)- BETA_T* MAX_JOINT_TORQUE, 0)

def torques_r(joint_torques):
    return jp.sum(jp.square(joint_torques))

def track_lin_vel_r(control_input_velocity, torso_local_velocity):
    SIGMA = 0.5
    # Reward tracking specified linear velocity command
    error = jp.array(control_input_velocity - torso_local_velocity[:2])
    error *= 1./(1. + jp.abs(control_input_velocity))
    error = jp.sum(jp.square(error))
    return jp.exp(-error/SIGMA)

def track_ang_vel_r(control_input_yaw, local_ang_vel):
    SIGMA = 0.5
    # Reward tracking yaw angular velocity command
    ang_vel_error = jp.square(
        (control_input_yaw - local_ang_vel[2])*2/jp.pi)
    return jp.exp(-ang_vel_error/SIGMA)

# ##################### HELPER FUNCTIONS ################################## #

def sqrdexp(x):
    SIGMA = 0.5
    return jp.exp(-jp.square(x)/SIGMA)


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
    

def controlInputRewardFn(
    velocity,
    target_velocity,
    torso_quat,
    target_yaw,
    z_pos,
    joint_torques,
    ctrl_change,
    isSelfColliding,
):
    # ALL IN NWU
    # velocity in m/s (x/y/z)
    # target_velocity in m/s (x/y)
    # quat is quaternion of torso (wxyz)
    # target_yaw in radians
    # z_pos in meters
    # joint torques in N m
    # ctrl_change is (latest_action - previous_action)
    # isSelfColliding is a boolean (is robot colliding with itself?)

    ### REWARD PARAMETERS
    # Velocity The magnitude of the player’s velocity minus the target velocity. - 0.1
    VELOCITY_ERROR_REWARD_WEIGHT = -0.1
    # Velocity The magnitude of the player's vertical velocity
    VERTICAL_VELOCITY_REWARD_WEIGHT = -0.1
    # Termination A penalty, equal to −1 if the player is on the ground - 0.5
    # MODIFICATION: +0.5 reward for torso being above or at Z=0, linearly interpolated to -0.5 if the torso is under -0.4
    GROUNDED_PENALTY_WEIGHT = -0.1
    NOT_GROUNDED_REWARD_WEIGHT = 0.1  # added this so that staying not grounded is rewarded (rather than terminating quickly to avoid future penalties)
    MIN_Z_BEFORE_GROUNDED = -0.3
    MIN_Z_FOR_REWARD = -0.2
    MAX_Z = -0.1
    # Joint torque A penalty, equal to the magnitude of the torque measured at
    # the player’s knees. This discourages the player from learning
    # gaits which cause high forces on the knees, for example
    # during ground impacts, which can damage a physical robot. - 0.01
    MAX_JOINT_TORQUE = 1.5
    JOINT_TORQUE_PENALTY_WEIGHT = (
        -0.1 / 16
    )  # divide by N since there are N joints and we consider the sum of joint torques
    # penalty term to minimize the time integral of torque peaks
    # (thresholded above 5 N m)
    # CUSTOM: a penalty for how much the joint control differs from previous joint control, to reward "smoother" motions (std is 0 to 2)
    CONTROL_STD_PENALTY = 0  # -0.01
    # Upright: a reward which scales linearly with how close the torso is to perfectly upright
    ORIENTATION_REWARD_MAX_ERROR_RAD = (
        0.6  # CUSTOM -> paper does not penalize high tilts
    )
    ORIENTATION_REWARD_MAX_ERROR_RAD_FOR_REWARD = 0.3  # at this error rad reward is 0
    ORIENTATION_REWARD_MIN_ERROR_RAD = 0.1  # at this error rad reward is at MAX_REWARD
    ORIENTATION_MAX_REWARD = 0.2  # CUSTOM -> paper has it at 0.02
    ORIENTATION_MAX_PENALTY = -0.1  # CUSTOM -> paper does not penalize high tilts
    # CUSTOM: add a constant offset to the reward
    CONSTANT_REWARD_OFFSET = 1.0
    # CUSTOM: penalize self-collisions
    TERMINATE_ON_SELF_COLLISION = False
    SELF_COLLISION_PENALTY = -0.1
    # CUSTOM hyperparameters
    ALLOW_NEGATIVE_REWARDS = True
    ALLOW_EARLY_TERMINATION = True
    OVERRIDE_TERMINAL_REWARD = True
    TERMINAL_REWARD = -10 if ALLOW_NEGATIVE_REWARDS else 0

    ### COMPUTE REWARD
    reward = 0

    # Velocity error
    abs_velocity_error = jp.abs(velocity[0:2] - target_velocity)
    hvelocity_reward = VELOCITY_ERROR_REWARD_WEIGHT * (
        jp.linalg.norm(abs_velocity_error) ** 2
    )
    reward += hvelocity_reward
    # print("hvelocity_reward", hvelocity_reward)

    # Vertical velocity
    abs_vertical_velocity = jp.abs(velocity)[2]
    vvelocity_reward = VERTICAL_VELOCITY_REWARD_WEIGHT * (abs_vertical_velocity**2)
    reward += vvelocity_reward
    # print("vvelocity_reward", vvelocity_reward)

    # Torso height reward
    z_pos_penalty = jp.interp(
        z_pos,
        jp.array([MIN_Z_FOR_REWARD, MAX_Z]),
        jp.array([GROUNDED_PENALTY_WEIGHT, NOT_GROUNDED_REWARD_WEIGHT]),
    )
    reward += z_pos_penalty
    # print("z_pos_penalty", z_pos_penalty)

    # Joint torque
    thresholded_joint_torques = (
        jp.clip(jp.abs(joint_torques), 0.0, MAX_JOINT_TORQUE) / MAX_JOINT_TORQUE
    )
    total_joint_torque = jp.sum(thresholded_joint_torques)
    joint_torque_reward = total_joint_torque * JOINT_TORQUE_PENALTY_WEIGHT
    reward += joint_torque_reward
    # print("joint_torque_reward", joint_torque_reward)

    # Control change
    control_std = jp.std(ctrl_change)
    control_std_reward = CONTROL_STD_PENALTY * control_std
    reward += control_std_reward
    # print("control_std_reward", control_std_reward)

    # constant reward offset
    reward += CONSTANT_REWARD_OFFSET

    # Rewards using torso_quat
    torso_quat_obj = Rotation.from_quat(
        [torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]
    )

    # Upright
    tilt_vector = torso_quat_obj.inv().apply(jp.array([0, 0, 1]))
    tilt_vector = jp.clip(
        tilt_vector,
        a_min=-1 * jp.ones(tilt_vector.shape),
        a_max=jp.ones(tilt_vector.shape),
    )
    tilt_amt = jp.arcsin(jp.max(jp.abs(tilt_vector[0:2])))
    tilt_amt = jp.where(tilt_vector[2] <= 0, ORIENTATION_REWARD_MAX_ERROR_RAD, tilt_amt)
    tilt_reward = jp.interp(
        tilt_amt,
        jp.array(
            [
                ORIENTATION_REWARD_MIN_ERROR_RAD,
                ORIENTATION_REWARD_MAX_ERROR_RAD_FOR_REWARD,
                ORIENTATION_REWARD_MAX_ERROR_RAD,
            ]
        ),
        jp.array([ORIENTATION_MAX_REWARD / 2, 0, ORIENTATION_MAX_PENALTY / 2]),
    )
    reward += tilt_reward

    # Target orientation
    rot_vector = torso_quat_obj.inv().apply(jp.array([1, 0, 0]))
    target_rot_vector = jp.array([jp.cos(target_yaw)[0], jp.sin(target_yaw)[0], 0])
    rot_error = jp.abs(target_rot_vector - rot_vector)
    rot_error_amt = jp.arcsin(jp.max(rot_error) / 2)
    rot_reward = jp.interp(
        rot_error_amt,
        jp.array(
            [
                ORIENTATION_REWARD_MIN_ERROR_RAD,
                ORIENTATION_REWARD_MAX_ERROR_RAD_FOR_REWARD,
                ORIENTATION_REWARD_MAX_ERROR_RAD,
            ]
        ),
        jp.array([ORIENTATION_MAX_REWARD / 2, 0, ORIENTATION_MAX_PENALTY / 2]),
    )
    reward += rot_reward
    # print("rot_reward", rot_reward)

    # Self collision penalty
    self_collision_reward = jp.where(isSelfColliding, SELF_COLLISION_PENALTY, 0)
    reward = reward + self_collision_reward
    # print("self_collision_reward", self_collision_reward)

    # make reward positive if necessary
    if not ALLOW_NEGATIVE_REWARDS:
        reward = jp.where(reward < 0, 0, reward)

    # Termination
    isTouchingGround = jp.where(z_pos > MIN_Z_BEFORE_GROUNDED, False, True)
    terminal = jp.where(isTouchingGround, ALLOW_EARLY_TERMINATION, False)
    if TERMINATE_ON_SELF_COLLISION:
        terminal = jp.where(isSelfColliding, ALLOW_EARLY_TERMINATION, terminal)

    if OVERRIDE_TERMINAL_REWARD:
        reward = jp.where(terminal, TERMINAL_REWARD, reward)
        
    return reward, terminal


def standingRewardFn(
    velocity,
    target_velocity,
    torso_quat,
    target_yaw,
    z_pos,
    joint_torques,
    ctrl_change,
    isSelfColliding,
):
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
    NOT_GROUNDED_REWARD_WEIGHT = 0.5  # added this so that staying not grounded is rewarded (rather than terminating quickly to avoid future penalties)
    MIN_Z_BEFORE_GROUNDED = -0.3
    MIN_Z_FOR_REWARD = -0.2
    MAX_Z = -0.1
    # Joint torque A penalty, equal to the magnitude of the torque measured at
    # the player’s knees. This discourages the player from learning
    # gaits which cause high forces on the knees, for example
    # during ground impacts, which can damage a physical robot. - 0.01
    MAX_JOINT_TORQUE = 1.5
    JOINT_TORQUE_PENALTY_WEIGHT = (
        -0.1 / 16
    )  # divide by N since there are N joints and we consider the sum of joint torques
    # penalty term to minimize the time integral of torque peaks
    # (thresholded above 5 N m)
    # CUSTOM: a penalty for how much the joint control differs from previous joint control, to reward "smoother" motions (std is 0 to 2)
    CONTROL_STD_PENALTY = -0.01
    # Upright: a reward which scales linearly with how close the torso is to perfectly upright
    UPRIGHT_REWARD_MAX_PENALTY_TILT = (
        0.8  # CUSTOM -> paper does not penalize high tilts
    )
    UPRIGHT_REWARD_MAX_TILT_FOR_REWARD = 0.4
    UPRIGHT_REWARD_MIN_TILT = 0.2
    UPRIGHT_MAX_REWARD = 0.2  # CUSTOM -> paper has it at 0.02
    UPRIGHT_MAX_PENALTY = -0.1  # CUSTOM -> paper does not penalize high tilts
    # CUSTOM: add a constant offset to the reward
    CONSTANT_REWARD_OFFSET = 0.5
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
    z_pos_penalty = jp.interp(
        z_pos,
        jp.array([MIN_Z_FOR_REWARD, MAX_Z]),
        jp.array([GROUNDED_PENALTY_WEIGHT, NOT_GROUNDED_REWARD_WEIGHT]),
    )
    reward += z_pos_penalty

    # Termination
    isTouchingGround = jp.where(z_pos > MIN_Z_BEFORE_GROUNDED, False, True)

    # Joint torque
    thresholded_joint_torques = (
        jp.clip(jp.abs(joint_torques), 0.0, MAX_JOINT_TORQUE) / MAX_JOINT_TORQUE
    )
    # thresholded_joint_torques = jp.minimum(jp.abs(joint_torques), jp.full(joint_torques.shape, 5.0))
    total_joint_torque = jp.sum(thresholded_joint_torques)
    reward += total_joint_torque * JOINT_TORQUE_PENALTY_WEIGHT

    # Control change
    control_std = jp.std(ctrl_change)
    reward += CONTROL_STD_PENALTY * control_std

    reward += CONSTANT_REWARD_OFFSET

    # Upright
    tilt_vector = (
        Rotation.from_quat([torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]])
        .inv()
        .apply(jp.array([0, 0, 1]))
    )
    tilt_amt = jp.arcsin(jp.max(jp.abs(tilt_vector[0:2])))
    tilt_amt = jp.where(tilt_vector[2] <= 0, UPRIGHT_REWARD_MAX_PENALTY_TILT, tilt_amt)
    tilt_reward = jp.interp(
        tilt_amt,
        jp.array(
            [
                UPRIGHT_REWARD_MIN_TILT,
                UPRIGHT_REWARD_MAX_TILT_FOR_REWARD,
                UPRIGHT_REWARD_MAX_PENALTY_TILT,
            ]
        ),
        jp.array([UPRIGHT_MAX_REWARD, 0, UPRIGHT_MAX_PENALTY]),
    )
    reward += tilt_reward

    if TERMINATE_ON_SELF_COLLISION:
        reward = jp.where(isSelfColliding, 0, reward)
    else:
        reward = jp.where(isSelfColliding, reward + SELF_COLLISION_PENALTY, reward)

    reward = jp.where(reward < 0, 0, reward)
    reward = jp.where(isTouchingGround, 0, reward)

    terminal = jp.where(isTouchingGround, True, False)
    if TERMINATE_ON_SELF_COLLISION:
        terminal = jp.where(isSelfColliding, True, terminal)

    return reward, terminal
