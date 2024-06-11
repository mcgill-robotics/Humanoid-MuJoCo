from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
from simulation.simulation_parameters import *

# REWARD INFO FROM https://arxiv.org/pdf/2304.13653.pdf and https://arxiv.org/abs/2307.10142


def scaled_exp(x):
    EXP_SCALING_PARAM = 0.5
    return jp.exp(-x / EXP_SCALING_PARAM)


def sqr(x):
    return jp.square(x)


def horizontal_velocity_reward(velocity, target_velocity):
    VELOCITY_ERROR_REWARD_WEIGHT = 10
    hvelocity_reward = VELOCITY_ERROR_REWARD_WEIGHT * scaled_exp(
        sqr(jp.linalg.norm(velocity[0:2] - target_velocity))
    )
    # print("hvelocity_reward", hvelocity_reward)
    return hvelocity_reward


def target_orientation_reward(torso_quat_obj, target_yaw):
    YAW_REWARD_WEIGHT = 0
    UPRIGHT_REWARD_WEIGHT = 10
    fwd_rot_vector = torso_quat_obj.inv().apply(jp.array([1, 0, 0]))
    target_fwd_rot_vector = jp.array([jp.cos(target_yaw)[0], jp.sin(target_yaw)[0], 0])
    down_rot_vector = torso_quat_obj.inv().apply(jp.array([0, 0, -1]))
    target_down_rot_vector = jp.array([0, 0, -1])
    upright_reward = UPRIGHT_REWARD_WEIGHT * scaled_exp(
        (sqr(jp.linalg.norm(down_rot_vector - target_down_rot_vector)))
    )
    yaw_reward = YAW_REWARD_WEIGHT * scaled_exp(
        (sqr(jp.linalg.norm(fwd_rot_vector - target_fwd_rot_vector)))
    )
    rot_reward = upright_reward + yaw_reward
    # print("rot_reward", rot_reward)
    return rot_reward


def vertical_velocity_reward(velocity):
    VERTICAL_VELOCITY_REWARD_WEIGHT = 5
    vvelocity_reward = VERTICAL_VELOCITY_REWARD_WEIGHT * scaled_exp(sqr(velocity[2]))
    # print("vvelocity_reward", vvelocity_reward)
    return vvelocity_reward


def torso_height_reward(z_pos):
    TORSO_HEIGHT_REWARD_WEIGHT = 5
    TARGET_Z_POS = -0.1
    MIN_Z_POS_FOR_REWARD = -0.3
    z_pos_reward = jp.interp(
        z_pos,
        jp.array([MIN_Z_POS_FOR_REWARD, TARGET_Z_POS]),
        jp.array([0, TORSO_HEIGHT_REWARD_WEIGHT]),
    )
    # print("z_pos_reward", z_pos_reward)
    return z_pos_reward


def joint_torque_reward(joint_torques):
    JOINT_TORQUE_REWARD_WEIGHT = 1
    MAX__SAFE_JOINT_TORQUE = 1.25
    joint_torque_reward = (
        JOINT_TORQUE_REWARD_WEIGHT
        * jp.sum(
            scaled_exp(
                sqr(jp.clip(jp.abs(joint_torques) - MAX__SAFE_JOINT_TORQUE, 0, jp.inf))
            )
        )
        / len(joint_torques)
    )
    # print("joint_torque_reward", joint_torque_reward)
    return joint_torque_reward


def control_change_reward(previous_ctrl, latest_ctrl):
    CONTROL_CHANGE_REWARD_WEIGHT = 0.5
    ctrl_change = latest_ctrl - previous_ctrl
    control_change_reward = (
        CONTROL_CHANGE_REWARD_WEIGHT
        * jp.sum(scaled_exp(sqr(ctrl_change * CONTROL_FREQUENCY)))
        / len(ctrl_change)
    )
    # print("control_change_reward", control_change_reward)
    return control_change_reward


def control_regularization_reward(ctrl):
    CONTROL_REG_REWARD_WEIGHT = 0.5
    control_regularization_reward = (
        CONTROL_REG_REWARD_WEIGHT * jp.sum(scaled_exp(sqr(ctrl))) / len(ctrl)
    )
    # print("control_regularization_reward", control_regularization_reward)
    return control_regularization_reward


def symmetry_reward(ctrl):
    SYMMETRY_REWARD_WEIGHT = 1
    EQUAL_REG_JOINTS = [
        # ("left_hip_yaw", "right_hip_yaw"),
        ("left_elbow", "right_elbow"),
    ]
    OPPOSITE_REG_JOINTS = [
        ("left_hip_roll", "right_hip_roll"),
        ("left_hip_pitch", "right_hip_pitch"),
        ("left_knee", "right_knee"),
        # ("left_ankle_pitch", "right_ankle_pitch"),
        ("left_shoulder_pitch", "right_shoulder_pitch"),
        ("left_shoulder_roll", "right_shoulder_roll"),
    ]
    symmetry_reward = 0
    for joint_a, joint_b in EQUAL_REG_JOINTS:
        i, j = JOINT_ACTUATOR_NAMES.index(joint_a), JOINT_ACTUATOR_NAMES.index(joint_b)
        symmetry_reward += scaled_exp(sqr(ctrl[i] - ctrl[j]))
    for joint_a, joint_b in OPPOSITE_REG_JOINTS:
        i, j = JOINT_ACTUATOR_NAMES.index(joint_a), JOINT_ACTUATOR_NAMES.index(joint_b)
        symmetry_reward += scaled_exp(sqr(ctrl[i] + ctrl[j]))

    symmetry_reward = (
        SYMMETRY_REWARD_WEIGHT
        * symmetry_reward
        / (len(EQUAL_REG_JOINTS) + len(OPPOSITE_REG_JOINTS))
    )
    # print("symmetry_reward", symmetry_reward)
    return symmetry_reward


def self_collision_reward(isSelfColliding):
    SELF_COLLISION_PENALTY = -10
    self_collision_reward = jp.where(isSelfColliding, SELF_COLLISION_PENALTY, 0)
    # print("self_collision_reward", self_collision_reward)
    return self_collision_reward


def controlInputRewardFn(
    velocity,
    target_velocity,
    torso_quat,
    target_yaw,
    z_pos,
    joint_torques,
    previous_ctrl,
    latest_ctrl,
    isSelfColliding,
    timestep,
):
    # ALL IN NWU
    # velocity in m/s (x/y/z)
    # target_velocity in m/s (x/y)
    # quat is quaternion of torso (wxyz)
    # target_yaw in radians
    # z_pos in meters
    # joint torques in N m
    # isSelfColliding is a boolean (is robot colliding with itself?)

    ### COMPUTE REWARD
    reward = 0

    ### HORIZONTAL VELOCITY REWARD
    reward += horizontal_velocity_reward(velocity, target_velocity)

    # TARGET ORIENTATION REWARD
    torso_quat_obj = Rotation.from_quat(
        [torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]
    )
    reward += target_orientation_reward(torso_quat_obj, target_yaw)

    ### VERTICAL VELOCITY REWARD
    reward += vertical_velocity_reward(velocity)

    ### TORSO HEIGHT REWARD
    reward += torso_height_reward(z_pos)

    # JOINT TORQUE REWARD
    reward += joint_torque_reward(joint_torques)

    # CONTROL CHANGE REWARD
    reward += control_change_reward(previous_ctrl, latest_ctrl)

    # CONTROL REGULARIZATION REWARD
    reward += control_regularization_reward(latest_ctrl)

    # SYMMETRY REWARD
    reward += symmetry_reward(latest_ctrl)

    # SELF COLLISION REWARD
    reward += self_collision_reward(isSelfColliding)

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
    local_gravity_vector = torso_quat_obj.inv().apply(jp.array([0, 0, -1]))
    isNotUpright = jp.where(
        jp.max(jp.abs(local_gravity_vector[0:2])) < 0.7, isTouchingGround, True
    )
    terminal = jp.where(isNotUpright, ALLOW_EARLY_TERMINATION, False)
    if TERMINATE_ON_SELF_COLLISION:
        terminal = jp.where(isSelfColliding, ALLOW_EARLY_TERMINATION, terminal)

    # DO NOT TERMINATE IF WITHIN GRACE PERIOD
    terminal = jp.where(timestep < GRACE_PERIOD_AFTER_RESET, False, terminal)

    # OVERRIDE TERMINAL REWARD IF DESIRED
    OVERRIDE_TERMINAL_REWARD = True
    TERMINAL_REWARD = -100.0
    if OVERRIDE_TERMINAL_REWARD:
        reward = jp.where(terminal, TERMINAL_REWARD, reward)

    return reward, terminal
