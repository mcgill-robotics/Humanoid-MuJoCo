import numpy as np
import mujoco
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation.simulation_parameters import CONTROL_FREQUENCY, JOINT_NAMES
import time
import cv2
import scipy

scene_option = mujoco.MjvOption()
mujoco.mjv_defaultOption(scene_option)
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False


# function to calculate the distance the robot should be offset vertically so it is touching the ground
# This is a workaround since we do not have full state observations: the assumption being the robot will always be touching the ground
def find_ideal_distance_to_ground():
    height_offsets = np.linspace(-0.05, 0.05, 2001)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(model, data, 1)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        data.qpos[2] += offset
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[2])

    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]
    return best_offset


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


def control(model, data, Q, qpos0, ctrl0):
    A = np.zeros((2 * model.nv, 2 * model.nv))
    B = np.zeros((2 * model.nv, model.nu))
    epsilon = 1e-8
    flg_centered = True
    # global A, B, K
    mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)
    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, np.eye(model.nu))
    # Compute the feedback gain matrix K.
    K = np.linalg.inv(np.eye(model.nu) + B.T @ P @ B) @ B.T @ P @ A
    # Allocate position difference dq.
    dq = np.zeros(model.nv)
    # Get state difference dx.
    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T
    # LQR control law.
    return ctrl0 - K @ dx


def calculate_initial_conditions(model, data):
    TARGET_Z_POS = data.qpos[2] + find_ideal_distance_to_ground()
    TARGET_QUAT = np.array([0.7, 0.0, 0.0, -0.7])
    TARGET_JOINT_POSITIONS = np.zeros(len(JOINT_NAMES))
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    # update mujoco state with observations
    data.qpos[3:7] = TARGET_QUAT
    data.qpos[JOINT_QPOS_IDX] = TARGET_JOINT_POSITIONS
    data.qpos[2] = TARGET_Z_POS
    qpos0 = data.qpos.copy()
    mujoco.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
    ctrl0 = ctrl0.flatten()
    return qpos0, ctrl0


def render():
    renderer.update_scene(data, camera="track", scene_option=scene_option)
    frame = renderer.render()
    cv2.imshow("CPU Sim View", frame)
    cv2.waitKey(1)


if __name__ == "__main__":
    # Cost coefficients.
    NUM_TRIALS = 1000
    RENDER = True
    MAX_SIM_TIME = 10.0  # s

    model = mujoco.MjModel.from_xml_path(SIM_XML_PATH)
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 720, 1080)
    JOINT_QPOS_IDX = []
    JOINT_DOF_IDX = []
    for joint in JOINT_NAMES:
        JOINT_DOF_IDX.append(model.jnt_dofadr[model.joint(joint).id])
        JOINT_QPOS_IDX.append(model.jnt_qposadr[model.joint(joint).id])

    # CALCULATE CTRL0
    qpos0, ctrl0 = calculate_initial_conditions(model, data)

    # CALCULATE Q MATRIX
    Q = calculate_q_matrix(model, data, qpos0)

    # SIMULATE THE CONTROLLER
    # Initialize score to 0
    score = 0
    for N in range(NUM_TRIALS):
        print("Trial", N)
        # Set the initial state and control.
        mujoco.mj_resetData(model, data)
        while data.time < MAX_SIM_TIME:
            score += 1
            try:
                start_comp_time = time.time()
                desired_torques = control(model, data, Q, qpos0, ctrl0)
                comp_time = time.time() - start_comp_time
                print("Computation HZ: ", 1.0 / comp_time, end="\r")
            except Exception:
                print(".                                            ")
                desired_torques = ctrl0
            data.ctrl = desired_torques

            mujoco.mj_step(model, data)
            if RENDER:
                render()
