from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
from simulation.simulation_parameters import *

# REWARD INFO FROM https://arxiv.org/pdf/2304.13653.pdf and https://arxiv.org/abs/2307.10142


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
    fwd_rot_vector = torso_quat_obj.inv().apply(jp.array([1, 0, 0]))
    target_fwd_rot_vector = jp.array([jp.cos(target_yaw)[0], jp.sin(target_yaw)[0], 0])
    down_rot_vector = torso_quat_obj.inv().apply(jp.array([0, 0, -1]))
    target_down_rot_vector = jp.array([0, 0, -1])
    rot_reward = ORIENTATION_REWARD_WEIGHT * jp.exp(
        -1
        * (
            jp.linalg.norm(fwd_rot_vector - target_fwd_rot_vector)
            + jp.linalg.norm(down_rot_vector - target_down_rot_vector)
        )
        / EXP_SCALING_PARAM
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
    # print("joint_torque_reward", joint_torque_reward)

    # ACTION CHANGE REWARD
    CONTROL_CHANGE_REWARD_WEIGHT = 0  # -1e-3
    control_change_reward = CONTROL_CHANGE_REWARD_WEIGHT * jp.linalg.norm(
        CONTROL_FREQUENCY * ctrl_change
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
    local_gravity_vector = torso_quat_obj.inv().apply(jp.array([0, 0, -1]))
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
