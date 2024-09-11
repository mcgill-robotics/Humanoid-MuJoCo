import numpy as np
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.simulation_parameters import (
    TIMESTEP,
    PHYSICS_STEPS_PER_CONTROL_STEP,
    CONTROL_FREQUENCY,
    TORSO_BODY_NAME,
    JOINT_NAMES,
)
import time
import cv2
import mujoco
import scipy


# Function to render current state of mujoco (for debugging)
def render(mj_data):
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    renderer.update_scene(mj_data, camera="track", scene_option=scene_option)
    frame = renderer.render()
    time.sleep(1 / CONTROL_FREQUENCY)
    cv2.imshow("CPU Sim View", frame)
    cv2.waitKey(1)


# side effect: resets data
def calculate_q_matrix(model, data, qpos0):
    BALANCE_COST = 1000  # Balancing.
    STABLE_JOINT_COST = 1.0  # Joints that should remain stable
    OTHER_JOINT_COST = 0.1  # Other joints.
    # Get the Jacobian for the root body (torso) CoM.
    mujoco.mj_resetData(model, data)
    data.qpos = qpos0
    mujoco.mj_forward(model, data)
    jac_com = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body("humanoid_world_link").id)

    # Get the Jacobian for the left foot.
    jac_left_foot = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(
        model, data, jac_left_foot, None, model.body("left_knee_pitch_link").id
    )
    jac_diff_left = jac_com - jac_left_foot
    Qbalance_left = jac_diff_left.T @ jac_diff_left

    # Get the Jacobian for the right foot.
    jac_right_foot = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(
        model, data, jac_right_foot, None, model.body("right_knee_pitch_link").id
    )
    jac_diff_right = jac_com - jac_right_foot
    Qbalance_right = jac_diff_right.T @ jac_diff_right

    # Get all joint names.
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    # Get indices into relevant sets of joints.
    root_dofs = range(6)
    body_dofs = range(6, model.nv)
    stable_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if ("hip" in name or "knee" in name)
    ]
    other_dofs = np.setdiff1d(body_dofs, stable_dofs)

    # Construct the Qjoint matrix.
    Qjoint = np.eye(model.nv)
    Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
    Qjoint[stable_dofs, stable_dofs] *= STABLE_JOINT_COST
    Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST
    # Construct the Q matrix for position DoFs.
    Qpos = (
        (BALANCE_COST / 2) * Qbalance_right
        + (BALANCE_COST / 2) * Qbalance_left
        + Qjoint
    )
    # penalty for velocity
    Qvel = np.zeros((model.nv, model.nv))
    Q = np.block([[Qpos, Qvel], [np.zeros((model.nv, 2 * model.nv))]])
    return Q


def calculate_initial_conditions(model, data):
    TARGET_Z_POS = -0.38
    TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
    TARGET_ANG_VEL = np.zeros(3)
    TARGET_JOINT_POSITIONS = np.zeros(len(JOINT_NAMES))
    TARGET_JOINT_VELOCITIES = np.zeros(len(JOINT_NAMES))
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    qpos0 = data.qpos.copy()
    mujoco.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
    ctrl0 = ctrl0.flatten()
    return qpos0, ctrl0


# function to calculate the distance the robot should be offset vertically so it is touching the ground
# This is a workaround since we do not have full state observations: the assumption being the robot will always be touching the ground
def find_ideal_distance_to_ground(
    mj_model, mj_data, joint_positions, torso_orientation
):
    height_offsets = np.linspace(-0.5, 0.5, 100)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
        mujoco.mj_forward(mj_model, mj_data)
        mj_data.qacc = 0
        mj_data.qpos[3:7] = torso_orientation
        mj_data.qpos[JOINT_QPOS_IDX] = joint_positions
        mj_data.qpos[2] += offset
        mujoco.mj_inverse(mj_model, mj_data)
        vertical_forces.append(mj_data.qfrc_inverse[2])

    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]
    return best_offset


def set_mujoco_state(state, mj_model, mj_data):
    joint_positions, joint_velocities, torso_ang_vel, torso_orientation = state
    dist_to_ground = find_ideal_distance_to_ground(
        mj_model, mj_data, joint_positions, torso_orientation
    )
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
    mj_data.qacc = 0
    # update mujoco state with observations
    mj_data.qpos[3:7] = torso_orientation
    mj_data.qpos[JOINT_QPOS_IDX] = joint_positions
    mj_data.qpos[2] += dist_to_ground
    mj_data.qvel[JOINT_DOF_IDX] = joint_velocities
    mj_data.qvel[3:6] = torso_ang_vel
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data


# Function which takes in partial system observations and returns optimal joint targets
def control(state, mj_model, mj_data, Q, qpos0, ctrl0):
    mj_model, mj_data = set_mujoco_state(state, mj_model, mj_data)

    A = np.zeros((2 * mj_model.nv, 2 * mj_model.nv))
    B = np.zeros((2 * mj_model.nv, mj_model.nu))
    epsilon = 1e-8
    flg_centered = True
    mujoco.mjd_transitionFD(mj_model, mj_data, epsilon, flg_centered, A, B, None, None)
    try:
        # Solve discrete Riccati equation.
        P = scipy.linalg.solve_discrete_are(A, B, Q, np.eye(mj_model.nu))
        # Compute the feedback gain matrix K.
        K = np.linalg.inv(np.eye(mj_model.nu) + B.T @ P @ B) @ B.T @ P @ A
    except Exception as e:
        print(str(e))
        return ctrl0
    # Allocate position difference dq.
    dq = np.zeros(mj_model.nv)
    # Get state difference dx.
    mujoco.mj_differentiatePos(
        mj_model, dq, PHYSICS_STEPS_PER_CONTROL_STEP, qpos0, mj_data.qpos
    )
    dx = np.hstack((dq, mj_data.qvel)).T
    # LQR control law.
    return np.clip(ctrl0 - K @ dx, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)


if __name__ == "__main__":
    # SETUP MUJOCO DATA
    mj_model = mujoco.MjModel.from_xml_path(SIM_XML_PATH)
    renderer = mujoco.Renderer(mj_model, 720, 1080)
    mj_model.opt.timestep = TIMESTEP
    mj_data = mujoco.MjData(mj_model)
    TORSO_BODY_ID = mj_model.body(TORSO_BODY_NAME).id
    NU = mj_model.nu  # Alias for the number of actuators.
    NV = mj_model.nv  # Shortcut for the number of DoFs.
    JOINT_QPOS_IDX = []
    JOINT_DOF_IDX = []
    JOINT_LOWER_LIMITS = []
    JOINT_UPPER_LIMITS = []
    for joint in JOINT_NAMES:
        JOINT_DOF_IDX.append(mj_model.jnt_dofadr[mj_model.joint(joint).id])
        JOINT_QPOS_IDX.append(mj_model.jnt_qposadr[mj_model.joint(joint).id])
        JOINT_LOWER_LIMITS.append(mj_model.actuator(joint).forcerange[0])
        JOINT_UPPER_LIMITS.append(mj_model.actuator(joint).forcerange[1])
    # CALCULATE CTRL0
    qpos0, ctrl0 = calculate_initial_conditions(mj_model, mj_data)
    # CALCULATE Q MATRIX
    Q = calculate_q_matrix(mj_model, mj_data, qpos0)
    # ----------- SETUP ENVIRONMENT -----------
    env = CPUEnv(
        xml_path=SIM_XML_PATH,
        randomization_factor=0.1,
        enable_rendering=True,
    )
    done = False
    torque_ctrl = np.zeros(12)
    env.reset()
    while env.data.time < 2:
        obs, _, done, _, _ = env.step(torque_ctrl)
        env.render("human")

        joint_positions = obs[: len(JOINT_NAMES)]
        joint_velocities = obs[len(JOINT_NAMES) : 2 * len(JOINT_NAMES)]
        torso_ang_vel = obs[2 * len(JOINT_NAMES) : 3 + 2 * len(JOINT_NAMES)]
        torso_quat = env.torso_quat

        state = (
            joint_positions,  # radians
            joint_velocities,  # radians / s
            torso_ang_vel,  # local angular velocity, rad / s
            torso_quat,  # quaternion in WXYZ form of torso
        )

        torque_ctrl = control(state, mj_model, mj_data, Q, qpos0, ctrl0)
        # print(torque_ctrl)

    renderer.close()
