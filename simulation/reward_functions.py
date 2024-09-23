# import jax

import numpy as jp

# if jax.default_backend() != "gpu":
#     print(" >> Using CPU backend for NumPy.")
#     import numpy as jp
# else:
#     print(" >> Using GPU backend for NumPy.")
#     from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
from simulation.simulation_parameters import *

# REWARD INFO FROM https://arxiv.org/pdf/2304.13653.pdf and https://arxiv.org/abs/2307.10142


def scaled_exp(x):
    EXP_SCALING_PARAM = 0.5
    return jp.exp(-x / EXP_SCALING_PARAM)


def sqr(x):
    return jp.square(x)


def horizontal_velocity_penalty(velocity, target_velocity):
    hvelocity_reward = HORIZONTAL_VELOCITY_PENALTY_WEIGHT * scaled_exp(
        sqr(jp.linalg.norm(velocity[0:2] - target_velocity))
    )
    # print("hvelocity_reward", hvelocity_reward)
    return hvelocity_reward


def target_orientation_reward(torso_quat_obj_inv, target_yaw, down_rot_vector):
    fwd_rot_vector = torso_quat_obj_inv.apply(jp.array([1, 0, 0]))
    target_fwd_rot_vector = jp.array([jp.cos(target_yaw)[0], jp.sin(target_yaw)[0], 0])
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


def vertical_velocity_penalty(velocity):
    vvelocity_reward = VERTICAL_VELOCITY_PENALTY_WEIGHT * scaled_exp(sqr(velocity[2]))
    # print("vvelocity_reward", vvelocity_reward)
    return vvelocity_reward


def torso_height_reward(z_pos):
    z_pos_reward = jp.interp(
        z_pos,
        jp.array([MIN_Z_POS_FOR_REWARD, TARGET_Z_POS]),
        jp.array([0, TORSO_HEIGHT_REWARD_WEIGHT]),
    )
    # print("z_pos_reward", z_pos_reward)
    return z_pos_reward


def joint_torque_penalty(joint_torques):
    joint_torque_penalty = (
        JOINT_TORQUE_PENALTY_WEIGHT
        * jp.sum(
            scaled_exp(
                sqr(jp.clip(jp.abs(joint_torques) - MAX__SAFE_JOINT_TORQUE, 0, jp.inf))
            )
        )
        / len(joint_torques)
    )
    # print("joint_torque_penalty", joint_torque_penalty)
    return joint_torque_penalty


def control_change_penalty(previous_ctrl, latest_ctrl):
    ctrl_change = latest_ctrl - previous_ctrl
    control_change_penalty = (
        CONTROL_CHANGE_PENALTY_WEIGHT
        * jp.sum(scaled_exp(sqr(ctrl_change * CONTROL_FREQUENCY)))
        / len(ctrl_change)
    )
    # print("control_change_penalty", control_change_penalty)
    return control_change_penalty


def control_regularization_reward(ctrl):
    control_regularization_reward = (
        CONTROL_REG_REWARD_WEIGHT * jp.sum(scaled_exp(sqr(ctrl))) / len(ctrl)
    )
    # print("control_regularization_reward", control_regularization_reward)
    return control_regularization_reward


def symmetry_reward(ctrl):
    symmetry_reward = 0
    for joint_a, joint_b in EQUAL_REG_JOINTS:
        i, j = JOINT_NAMES.index(joint_a), JOINT_NAMES.index(joint_b)
        symmetry_reward += scaled_exp(sqr(ctrl[i] - ctrl[j]))
    for joint_a, joint_b in OPPOSITE_REG_JOINTS:
        i, j = JOINT_NAMES.index(joint_a), JOINT_NAMES.index(joint_b)
        symmetry_reward += scaled_exp(sqr(ctrl[i] + ctrl[j]))

    symmetry_reward = (
        SYMMETRY_REWARD_WEIGHT
        * symmetry_reward
        / (len(EQUAL_REG_JOINTS) + len(OPPOSITE_REG_JOINTS))
    )
    # print("symmetry_reward", symmetry_reward)
    return symmetry_reward


def controlInputReward(
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
    max_simulation_time,
):
    # ALL IN NWU
    # velocity in m/s (x/y/z)
    # target_velocity in m/s (x/y)
    # quat is quaternion of torso (wxyz)
    # target_yaw in radians
    # z_pos in meters
    # joint torques in N m
    # isSelfColliding is a boolean (is robot colliding with itself?)

    global HORIZONTAL_VELOCITY_PENALTY_WEIGHT
    global YAW_REWARD_WEIGHT
    global UPRIGHT_REWARD_WEIGHT
    global VERTICAL_VELOCITY_PENALTY_WEIGHT
    global TORSO_HEIGHT_REWARD_WEIGHT
    global TARGET_Z_POS
    global MIN_Z_POS_FOR_REWARD
    global JOINT_TORQUE_PENALTY_WEIGHT
    global MAX__SAFE_JOINT_TORQUE
    global CONTROL_CHANGE_PENALTY_WEIGHT
    global CONTROL_REG_REWARD_WEIGHT
    global EQUAL_REG_JOINTS
    global OPPOSITE_REG_JOINTS
    global SYMMETRY_REWARD_WEIGHT
    global SELF_COLLISION_PENALTY

    ### COMPUTE REWARD
    reward = 0

    ### HORIZONTAL VELOCITY REWARD
    HORIZONTAL_VELOCITY_PENALTY_WEIGHT = 15
    reward += horizontal_velocity_penalty(velocity, target_velocity)

    # TARGET ORIENTATION REWARD
    YAW_REWARD_WEIGHT = 0
    UPRIGHT_REWARD_WEIGHT = 10
    torso_quat_obj_inv = Rotation.from_quat(
        [torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]
    ).inv()
    local_gravity_vector = torso_quat_obj_inv.apply(jp.array([0, 0, -1]))
    reward += target_orientation_reward(
        torso_quat_obj_inv, target_yaw, local_gravity_vector
    )

    ### VERTICAL VELOCITY REWARD
    VERTICAL_VELOCITY_PENALTY_WEIGHT = 5
    reward += vertical_velocity_penalty(velocity)

    ### TORSO HEIGHT REWARD
    TORSO_HEIGHT_REWARD_WEIGHT = 5
    TARGET_Z_POS = Z_INITIAL_POS
    MIN_Z_POS_FOR_REWARD = Z_INITIAL_POS - 0.75
    reward += torso_height_reward(z_pos)

    # JOINT TORQUE REWARD
    JOINT_TORQUE_PENALTY_WEIGHT = 2.5
    MAX__SAFE_JOINT_TORQUE = 1.0
    reward += joint_torque_penalty(joint_torques)

    # CONTROL CHANGE REWARD
    CONTROL_CHANGE_PENALTY_WEIGHT = 2
    reward += control_change_penalty(previous_ctrl, latest_ctrl)

    # CONTROL REGULARIZATION REWARD
    CONTROL_REG_REWARD_WEIGHT = 0.5
    reward += control_regularization_reward(latest_ctrl)

    # SYMMETRY REWARD
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
    SYMMETRY_REWARD_WEIGHT = 1
    reward += symmetry_reward(latest_ctrl)

    # SELF COLLISION REWARD
    SELF_COLLISION_PENALTY = -20
    if isSelfColliding:
        reward += SELF_COLLISION_PENALTY

    # CONSTANT REWARD OFFSET
    CONSTANT_REWARD_OFFSET = 0.0
    reward += CONSTANT_REWARD_OFFSET

    # FORCE REWARD TO BE POSITIVE IF DESIRED
    ALLOW_NEGATIVE_REWARDS = True
    if not ALLOW_NEGATIVE_REWARDS and reward < 0:
        reward = 0

    # CHECK TERMINATION CONDITION AND REWARD
    TERMINATE_ON_SELF_COLLISION = False
    ALLOW_EARLY_TERMINATION = True
    MIN_Z_BEFORE_GROUNDED = -0.5
    isTouchingGround = z_pos < MIN_Z_BEFORE_GROUNDED
    isUpright = jp.max(jp.abs(local_gravity_vector[0:2])) < 0.7
    terminal = ALLOW_EARLY_TERMINATION and (
        (not isUpright)
        or isTouchingGround
        or (TERMINATE_ON_SELF_COLLISION and isSelfColliding)
    )
    # OVERRIDE TERMINAL REWARD IF DESIRED
    OVERRIDE_TERMINAL_REWARD = True
    TERMINAL_REWARD = -100.0
    if OVERRIDE_TERMINAL_REWARD and terminal:
        reward = TERMINAL_REWARD

    truncated = timestep >= max_simulation_time

    return reward, terminal, truncated


def standupReward(
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
    max_simulation_time,
):
    # ALL IN NWU
    # velocity in m/s (x/y/z)
    # target_velocity in m/s (x/y)
    # quat is quaternion of torso (wxyz)
    # target_yaw in radians
    # z_pos in meters
    # joint torques in N m
    # isSelfColliding is a boolean (is robot colliding with itself?)

    global HORIZONTAL_VELOCITY_PENALTY_WEIGHT
    global YAW_REWARD_WEIGHT
    global UPRIGHT_REWARD_WEIGHT
    global VERTICAL_VELOCITY_PENALTY_WEIGHT
    global TORSO_HEIGHT_REWARD_WEIGHT
    global TARGET_Z_POS
    global MIN_Z_POS_FOR_REWARD
    global JOINT_TORQUE_PENALTY_WEIGHT
    global MAX__SAFE_JOINT_TORQUE
    global CONTROL_CHANGE_PENALTY_WEIGHT
    global CONTROL_REG_REWARD_WEIGHT
    global EQUAL_REG_JOINTS
    global OPPOSITE_REG_JOINTS
    global SYMMETRY_REWARD_WEIGHT
    global SELF_COLLISION_PENALTY

    ### COMPUTE REWARD
    reward = 0

    ### HORIZONTAL VELOCITY REWARD
    HORIZONTAL_VELOCITY_PENALTY_WEIGHT = 5
    reward += horizontal_velocity_penalty(velocity, target_velocity)

    # TARGET ORIENTATION REWARD
    YAW_REWARD_WEIGHT = 0
    UPRIGHT_REWARD_WEIGHT = 10
    torso_quat_obj_inv = Rotation.from_quat(
        [torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]
    ).inv()
    local_gravity_vector = torso_quat_obj_inv.apply(jp.array([0, 0, -1]))
    reward += target_orientation_reward(
        torso_quat_obj_inv, target_yaw, local_gravity_vector
    )

    ### VERTICAL VELOCITY REWARD
    VERTICAL_VELOCITY_PENALTY_WEIGHT = 0
    reward += vertical_velocity_penalty(velocity)

    ### TORSO HEIGHT REWARD
    TORSO_HEIGHT_REWARD_WEIGHT = 15
    TARGET_Z_POS = Z_INITIAL_POS
    MIN_Z_POS_FOR_REWARD = -0.6
    reward += torso_height_reward(z_pos)

    # JOINT TORQUE REWARD
    JOINT_TORQUE_PENALTY_WEIGHT = 2.5
    MAX__SAFE_JOINT_TORQUE = 1.0
    reward += joint_torque_penalty(joint_torques)

    # CONTROL CHANGE REWARD
    CONTROL_CHANGE_PENALTY_WEIGHT = 2
    reward += control_change_penalty(previous_ctrl, latest_ctrl)

    # CONTROL REGULARIZATION REWARD
    CONTROL_REG_REWARD_WEIGHT = 0.5
    reward += control_regularization_reward(latest_ctrl)

    # SYMMETRY REWARD
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
    SYMMETRY_REWARD_WEIGHT = 1
    reward += symmetry_reward(latest_ctrl)

    # SELF COLLISION REWARD
    SELF_COLLISION_PENALTY = -20
    if isSelfColliding:
        reward += SELF_COLLISION_PENALTY

    # CONSTANT REWARD OFFSET
    CONSTANT_REWARD_OFFSET = 0.0
    reward += CONSTANT_REWARD_OFFSET

    # FORCE REWARD TO BE POSITIVE IF DESIRED
    ALLOW_NEGATIVE_REWARDS = True
    if not ALLOW_NEGATIVE_REWARDS and reward < 0:
        reward = 0

    # CHECK TERMINATION CONDITION AND REWARD
    TERMINATE_ON_SELF_COLLISION = False
    terminal = False
    if timestep >= max_simulation_time or (
        TERMINATE_ON_SELF_COLLISION and isSelfColliding
    ):
        terminal = True

    # OVERRIDE TERMINAL REWARD IF DESIRED
    OVERRIDE_TERMINAL_REWARD = True
    TERMINAL_REWARD = -100.0
    if OVERRIDE_TERMINAL_REWARD and terminal:
        reward = TERMINAL_REWARD

    # check if successful
    isUpright = jp.max(jp.abs(local_gravity_vector[0:2])) < 0.7
    truncated = z_pos >= TARGET_Z_POS and isUpright

    return reward, terminal, truncated


SELECTED_REWARD_FUNCTION = standupReward
# SELECTED_REWARD_FUNCTION = controlInputReward
