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

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(SIM_XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 720, 1080)
    JOINT_QPOS_IDX = []
    JOINT_DOF_IDX = []
    for joint in JOINT_NAMES:
        JOINT_DOF_IDX.append(model.jnt_dofadr[model.joint(joint).id])
        JOINT_QPOS_IDX.append(model.jnt_qposadr[model.joint(joint).id])

    def render():
        renderer.update_scene(data, camera="track", scene_option=scene_option)
        frame = renderer.render()
        # time.sleep(1 / CONTROL_FREQUENCY)
        cv2.imshow("CPU Sim View", frame)
        cv2.waitKey(1)

    def torques_to_positions(init_qpos, init_qvel, torques):
        # mujoco.mj_resetDataKeyframe(model, data, 1)
        data.qpos = init_qpos
        data.qvel = init_qvel
        data.qfrc_applied[6:] = torques
        data.qfrc_actuator[6:] = torques
        for _ in range(10):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            mujoco.mj_inverse(model, data)
        print(torques)
        print(data.qfrc_actuator)
        print(data.actuator_force)
        print(data.qfrc_applied)
        print(data.qfrc_inverse)
        print(init_qpos - data.qpos)
        return data.qpos[JOINT_QPOS_IDX]

    # CALCULATE CTRL0
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    qpos0 = data.qpos.copy()
    mujoco.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
    ctrl0 = ctrl0.flatten()
    print(ctrl0)

    nu = model.nu  # Alias for the number of actuators.
    nv = model.nv  # Shortcut for the number of DoFs.

    # CREATE R MATRIX
    R = np.eye(nu)

    # CALCULATE Q MATRIX
    # Get the Jacobian for the root body (torso) CoM.
    mujoco.mj_resetData(model, data)
    data.qpos = qpos0
    mujoco.mj_forward(model, data)
    jac_com = np.zeros((3, nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body("humanoid_world_link").id)
    # Get the Jacobian for the left foot.
    jac_left_foot = np.zeros((3, nv))
    mujoco.mj_jacBodyCom(
        model, data, jac_left_foot, None, model.body("left_knee_pitch_link").id
    )
    # Get the Jacobian for the right foot.
    jac_right_foot = np.zeros((3, nv))
    mujoco.mj_jacBodyCom(
        model, data, jac_right_foot, None, model.body("right_knee_pitch_link").id
    )
    # Calculate the QBalance matrix to keep both feet below the CoM.
    jac_diff_left = jac_com - jac_left_foot
    QbalanceL = jac_diff_left.T @ jac_diff_left
    jac_diff_right = jac_com - jac_right_foot
    QbalanceR = jac_diff_right.T @ jac_diff_right
    # Get DOFs for the root, joints, and balance
    root_dofs = range(6)
    body_dofs = range(6, nv)
    # Cost coefficients.
    BALANCE_COST = 1e3  # Balancing.
    JOINT_COST = 3  # Non-root joints.

    # Construct the Qjoint matrix.
    Qjoint = np.eye(nv)
    Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
    Qjoint[body_dofs, body_dofs] *= JOINT_COST
    # Construct the Q matrix for position DoFs.
    Qpos = BALANCE_COST * QbalanceL + BALANCE_COST * QbalanceR + Qjoint
    # No explicit penalty for velocities.
    Q = np.block([[Qpos, np.zeros((nv, nv))], [np.zeros((nv, 2 * nv))]])

    # SIMULATE THE CONTROLLER
    # Constants
    NUM_TRIALS = 1
    RENDER = True
    RECALCULATE_A_B = False
    # Initialize score to 0
    score = 0
    for N in range(NUM_TRIALS):
        print("Trial", N)
        INIT_A_B = False
        # Set the initial state and control.
        mujoco.mj_resetData(model, data)
        data.ctrl = ctrl0
        data.qpos = qpos0
        while True:  # data.qpos[2] > -0.6:
            score += 1
            # Allocate the A and B matrices, compute them.
            try:
                if RECALCULATE_A_B or not INIT_A_B:
                    A = np.zeros((2 * nv, 2 * nv))
                    B = np.zeros((2 * nv, nu))
                    epsilon = 1e-6
                    flg_centered = True
                    mujoco.mjd_transitionFD(
                        model, data, epsilon, flg_centered, A, B, None, None
                    )

                    # Solve discrete Riccati equation.
                    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

                    # Compute the feedback gain matrix K.
                    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
                    # Allocate position difference dq.
                    dq = np.zeros(model.nv)
                    INIT_A_B = True
                # Get state difference dx.
                mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
                dx = np.hstack((dq, data.qvel)).T
                # LQR control law.
                desired_torques = ctrl0 - K @ dx
                positions = torques_to_positions(data.qpos, data.qvel, desired_torques)
                data.ctrl = positions
            except Exception as e:
                print(e)
                pass

            mujoco.mj_step(model, data)
            if RENDER:
                render()

    print("score:", score)
    print("RECALCULATE_A_B:", RECALCULATE_A_B)
