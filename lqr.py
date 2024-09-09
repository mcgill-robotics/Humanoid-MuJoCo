import numpy as np
import mujoco
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation.simulation_parameters import CONTROL_FREQUENCY, JOINT_NAMES
import time
import cv2
import scipy

def find_ideal_start_height():
    height_offsets = np.linspace(-0.001, 0.001, 2001)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(model, data, 1)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        # Offset the height by `offset`.
        data.qpos[2] += offset
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[2])

    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]
    return best_offset

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(SIM_XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 720, 1080)
    

    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    def render():
        renderer.update_scene(
            data, camera="track", scene_option=scene_option
        )
        frame = renderer.render()
        # time.sleep(1 / CONTROL_FREQUENCY)
        cv2.imshow("CPU Sim View", frame)
        cv2.waitKey(1)
        
    best_offset = find_ideal_start_height()
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    data.qpos[2] += best_offset
    qpos0 = data.qpos.copy()
    mujoco.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    print('desired forces:', qfrc0)
    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
    ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
    print('control setpoint:', ctrl0)
    
    nu = model.nu  # Alias for the number of actuators.
    R = np.eye(nu)
    nv = model.nv  # Shortcut for the number of DoFs.
    # Get the Jacobian for the root body (torso) CoM.
    mujoco.mj_resetData(model, data)
    data.qpos = qpos0
    mujoco.mj_forward(model, data)
    jac_com = np.zeros((3, nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body('torso').id)

    # Get the Jacobian for the left foot.
    jac_foot = np.zeros((3, nv))
    mujoco.mj_jacBodyCom(model, data, jac_foot, None, model.body('left_knee_pitch_link').id)

    jac_diff = jac_com - jac_foot
    Qbalance = jac_diff.T @ jac_diff
    
    BALANCE_JOINT_NAMES = [
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee"
    ]
    root_dofs = range(6)
    body_dofs = range(6, nv)
    balance_dofs = [
        model.joint(name).dofadr[0]
        for name in BALANCE_JOINT_NAMES
    ]
    other_dofs = np.setdiff1d(body_dofs, balance_dofs)
    
    # Cost coefficients.
    BALANCE_COST        = 1000  # Balancing.
    BALANCE_JOINT_COST  = 3     # Joints required for balancing.
    OTHER_JOINT_COST    = .3    # Other joints.

    # Construct the Qjoint matrix.
    Qjoint = np.eye(nv)
    Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
    Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
    Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

    # Construct the Q matrix for position DoFs.
    Qpos = BALANCE_COST * Qbalance + Qjoint

    # No explicit penalty for velocities.
    Q = np.block([[Qpos, np.zeros((nv, nv))],
                [np.zeros((nv, 2*nv))]])
    
    # Set the initial state and control.
    mujoco.mj_resetData(model, data)
    data.ctrl = ctrl0
    data.qpos = qpos0

    
    NUM_TRIALS = 50

    score = 0
    for _ in range(NUM_TRIALS):

        # Allocate position difference dq.
        dq = np.zeros(model.nv)

        INIT_A_B = False
        
        RECALCULATE_A_B = True
        RECALCULATE_CTRL_0 = False


        mujoco.mj_resetData(model, data)
        data.ctrl = ctrl0
        data.qpos = qpos0

        while data.qpos[2] > -0.4:
            
            score += 1
            
            # Allocate the A and B matrices, compute them.
            try:
                if RECALCULATE_A_B or not INIT_A_B:
                    A = np.zeros((2*nv, 2*nv))
                    B = np.zeros((2*nv, nu))
                    epsilon = 1e-6
                    flg_centered = True
                    mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)
                    
                    # Solve discrete Riccati equation.
                    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

                    # Compute the feedback gain matrix K.
                    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
                    # Get state difference dx.
                    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
                    dx = np.hstack((dq, data.qvel)).T
                    INIT_A_B = True

                if RECALCULATE_CTRL_0:
                    mujoco.mj_inverse(model, data)
                    qfrc0 = data.qfrc_inverse.copy()
                    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
                    ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
                # LQR control law.
                data.ctrl = ctrl0 - K @ dx
            except:
                pass

            mujoco.mj_step(model, data)
            # render()
        
    print('score:', score)
    print("RECALCULATE_A_B:", RECALCULATE_A_B)
    print("RECALCULATE_CTRL_0:", RECALCULATE_CTRL_0)