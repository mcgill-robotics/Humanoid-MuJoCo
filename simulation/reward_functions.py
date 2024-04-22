from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
from simulation.simulation_parameters import *

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


inverseRotateVectors = (
    lambda q, v: Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv().apply(v)
)


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

    EXP_SCALING_PARAM = 0.5

    ### COMPUTE REWARD
    reward = 0

    ### HORIZONTAL VELOCITY REWARD
    VELOCITY_ERROR_REWARD_WEIGHT = 10
    hvelocity_reward = VELOCITY_ERROR_REWARD_WEIGHT * jp.exp(
        -1 * jp.linalg.norm(velocity[0:2] - target_velocity) / EXP_SCALING_PARAM
    )
    reward += hvelocity_reward
    # print("hvelocity_reward", hvelocity_reward)

    # TARGET ORIENTATION REWARD
    ORIENTATION_REWARD_WEIGHT = 5  # CUSTOM -> paper does not penalize high tilts
    torso_quat_obj = Rotation.from_quat(
        [torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]
    )
    rot_vector = torso_quat_obj.inv().apply(jp.array([1, 0, 0]))
    target_rot_vector = jp.array([jp.cos(target_yaw)[0], jp.sin(target_yaw)[0], 0])
    rot_reward = ORIENTATION_REWARD_WEIGHT * jp.exp(
        -1 * jp.linalg.norm(rot_vector - target_rot_vector) / EXP_SCALING_PARAM
    )
    reward += rot_reward
    # print("rot_reward", rot_reward)

    ### VERTICAL VELOCITY REWARD
    VERTICAL_VELOCITY_REWARD_WEIGHT = 1
    vvelocity_reward = VERTICAL_VELOCITY_REWARD_WEIGHT * jp.exp(
        -1 * jp.abs(velocity)[2] / EXP_SCALING_PARAM
    )
    reward += vvelocity_reward
    # print("vvelocity_reward", vvelocity_reward)

    ### TORSO HEIGHT REWARD
    TORSO_HEIGHT_REWARD_WEIGHT = 1
    TARGET_Z_POS = (
        -0.1
    )  # above this position, the reward is the same (only penalize the torso for being too low)
    z_pos_reward = TORSO_HEIGHT_REWARD_WEIGHT * jp.exp(
        jp.clip(z_pos - TARGET_Z_POS, -jp.inf, 0) / EXP_SCALING_PARAM
    )
    reward += z_pos_reward
    # print("z_pos_reward", z_pos_reward)

    # JOINT TORQUE REWARD
    MAX_JOINT_TORQUE = 1.5
    JOINT_TORQUE_REWARD_WEIGHT = -0.01
    joint_torque_reward = JOINT_TORQUE_REWARD_WEIGHT * jp.linalg.norm(
        jp.clip(jp.abs(joint_torques) - MAX_JOINT_TORQUE, 0, jp.inf)
    )
    reward += joint_torque_reward
    print("joint_torque_reward", joint_torque_reward)

    # ACTION CHANGE REWARD
    CONTROL_CHANGE_REWARD_WEIGHT = 0  # -1e-3
    control_change_reward = CONTROL_CHANGE_REWARD_WEIGHT * jp.linalg.norm(
        desired_control_frequency * ctrl_change
    )
    reward += control_change_reward
    # print("control_std_reward", control_std_reward)

    # SELF COLLISION REWARD
    SELF_COLLISION_PENALTY = 0
    self_collision_reward = jp.where(isSelfColliding, SELF_COLLISION_PENALTY, 0)
    reward += self_collision_reward
    # print("self_collision_reward", self_collision_reward)

    # CONSTANT REWARD OFFSET
    CONSTANT_REWARD_OFFSET = 0.0
    reward += CONSTANT_REWARD_OFFSET

    # FORCE REWARD TO BE POSITIVE IF DESIRED
    ALLOW_NEGATIVE_REWARDS = True
    if not ALLOW_NEGATIVE_REWARDS:
        reward = jp.where(reward < 0, 0, reward)

    # CHECK TERMINATION CONDITION AND REWARD
    TERMINATE_ON_SELF_COLLISION = True
    ALLOW_EARLY_TERMINATION = True
    MIN_Z_BEFORE_GROUNDED = -0.3
    isTouchingGround = jp.where(z_pos > MIN_Z_BEFORE_GROUNDED, False, True)
    local_gravity_vector = inverseRotateVectors(torso_quat, jp.array([0, 0, -1]))
    isNotUpright = jp.where(
        jp.max(jp.abs(local_gravity_vector[0:2])) < 0.7, isTouchingGround, True
    )
    terminal = jp.where(isNotUpright, ALLOW_EARLY_TERMINATION, False)
    if TERMINATE_ON_SELF_COLLISION:
        terminal = jp.where(isSelfColliding, ALLOW_EARLY_TERMINATION, terminal)

    # OVERRIDE TERMINAL REWARD IF DESIRED
    OVERRIDE_TERMINAL_REWARD = True
    TERMINAL_REWARD = -100.0
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
