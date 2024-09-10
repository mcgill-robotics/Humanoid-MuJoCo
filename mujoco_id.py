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

# PARAMETERS
TARGET_Z_POS = -0.38
TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
TARGET_ANG_VEL = np.zeros(3)
TARGET_JOINT_POSITIONS = np.zeros(len(JOINT_NAMES))
TARGET_JOINT_VELOCITIES = np.zeros(len(JOINT_NAMES))
JOINT_VEL_COST = 0.1
BALANCE_COST = 1e3  # Balancing.
JOINT_COST = 3  # Non-root joints.

# SETUP MUJOCO DATA
mj_model = mujoco.MjModel.from_xml_path(SIM_XML_PATH)
mj_model.opt.timestep = TIMESTEP
mj_data = mujoco.MjData(mj_model)
TORSO_BODY_ID = mj_model.body(TORSO_BODY_NAME).id
NU = mj_model.nu  # Alias for the number of actuators.
NV = mj_model.nv  # Shortcut for the number of DoFs.
JOINT_QPOS_IDX = []
JOINT_DOF_IDX = []
for joint in JOINT_NAMES:
    JOINT_DOF_IDX.append(mj_model.jnt_dofadr[mj_model.joint(joint).id])
    JOINT_QPOS_IDX.append(mj_model.jnt_qposadr[mj_model.joint(joint).id])
INIT_DYNAMICS = False


# Function to render current state of mujoco (for debugging)
def render():
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


def torques_to_positions(init_qpos, init_qvel, torques):
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
    mj_data.qacc = 0
    mj_data.qpos = init_qpos
    mj_data.qvel = init_qvel
    mj_data.qfrc_actuator[6:] = torques
    mujoco.mj_step(mj_model, mj_data)
    return mj_data.qpos[JOINT_QPOS_IDX]


# function to calculate the distance the robot should be offset vertically so it is touching the ground
# This is a workaround since we do not have full state observations: the assumption being the robot will always be touching the ground
def find_ideal_distance_to_ground(joint_positions, torso_orientation):
    height_offsets = np.linspace(-0.5, 0.5, 100)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
        mujoco.mj_forward(mj_model, mj_data)
        mj_data.qpos[JOINT_QPOS_IDX] = joint_positions
        mj_data.qpos[3:7] = torso_orientation
        mj_data.qacc = 0
        mj_data.qpos[2] += offset
        mujoco.mj_inverse(mj_model, mj_data)
        vertical_forces.append(mj_data.qfrc_inverse[2])

    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]
    return best_offset


def set_mujoco_state(
    joint_positions, joint_velocities, torso_ang_vel, torso_orientation
):
    global mj_data
    global mj_model
    dist_to_ground = find_ideal_distance_to_ground(joint_positions, torso_orientation)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
    mj_data.qacc = 0
    # update mujoco state with observations
    mj_data.qpos[3:7] = torso_orientation
    mj_data.qpos[JOINT_QPOS_IDX] = joint_positions
    mj_data.qpos[2] += dist_to_ground
    mj_data.qvel[JOINT_DOF_IDX] = joint_velocities
    mj_data.qvel[3:6] = torso_ang_vel
    mujoco.mj_forward(mj_model, mj_data)


def compute_dynamics():
    global INIT_DYNAMICS
    global K
    global ctrl0
    # mujoco.mj_forward(mj_model, mj_data)
    mj_data.qacc = 0
    # calculate inverse dynamics
    mujoco.mj_inverse(mj_model, mj_data)
    qfrc0 = mj_data.qfrc_inverse.copy()
    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(mj_data.actuator_moment)
    ctrl0 = ctrl0.flatten()
    # Calculate COM Jacobians
    jac_com = np.zeros((3, NV))
    mujoco.mj_jacSubtreeCom(mj_model, mj_data, jac_com, TORSO_BODY_ID)
    # Get the Jacobian for the left foot.
    jac_left_foot = np.zeros((3, NV))
    mujoco.mj_jacBodyCom(
        mj_model, mj_data, jac_left_foot, None, mj_model.body("left_knee_pitch_link").id
    )
    # Get the Jacobian for the right foot.
    jac_right_foot = np.zeros((3, NV))
    mujoco.mj_jacBodyCom(
        mj_model,
        mj_data,
        jac_right_foot,
        None,
        mj_model.body("right_knee_pitch_link").id,
    )
    # Calculate the QBalance matrix to keep both feet below the CoM.
    jac_diff_left = jac_com - jac_left_foot
    QbalanceL = jac_diff_left.T @ jac_diff_left
    jac_diff_right = jac_com - jac_right_foot
    QbalanceR = jac_diff_right.T @ jac_diff_right

    # Get DOFs for the root, joints, and balance
    body_dofs = range(6, NV)
    # Construct the Qjoint matrix.
    Qjoint = np.zeros((NV, NV))
    Qjoint[body_dofs, body_dofs] = JOINT_COST
    # Construct the Q matrix for position DoFs.
    Qpos = Qjoint
    Qpos = BALANCE_COST * QbalanceL + BALANCE_COST * QbalanceR + Qjoint
    # Add penalty for velocities.
    Qvel = np.zeros((NV, NV))
    # Qvel[range(NV), range(NV)] = JOINT_VEL_COST
    Q = np.block([[Qpos, Qvel], [np.zeros((NV, 2 * NV))]])
    # CREATE R MATRIX
    R = np.eye(NU)
    A = np.zeros((2 * NV, 2 * NV))
    B = np.zeros((2 * NV, NU))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(mj_model, mj_data, epsilon, flg_centered, A, B, None, None)

    try:
        # Solve discrete Riccati equation.
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Compute the feedback gain matrix K.
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        INIT_DYNAMICS = True
    except Exception as e:
        print(e)


# Function which takes in partial system observations and returns optimal joint targets
def compute_optimal_joint_targets(
    joint_positions, joint_velocities, torso_ang_vel, torso_orientation
):
    if not INIT_DYNAMICS:
        set_mujoco_state(
            TARGET_JOINT_POSITIONS, TARGET_JOINT_VELOCITIES, TARGET_ANG_VEL, TARGET_QUAT
        )
        compute_dynamics()

    set_mujoco_state(
        joint_positions, joint_velocities, torso_ang_vel, torso_orientation
    )

    # For calculating position difference dq.
    dq = np.zeros(NV)
    TARGET_QPOS = np.concatenate(
        [
            mj_data.qpos[:2],
            np.array([TARGET_Z_POS]),
            TARGET_QUAT,
            np.zeros(len(JOINT_NAMES)),
        ]
    )
    TARGET_QPOS[JOINT_QPOS_IDX] = TARGET_JOINT_POSITIONS

    render()
    if INIT_DYNAMICS:
        mujoco.mj_differentiatePos(mj_model, dq, 1, TARGET_QPOS, mj_data.qpos)
        QVEL_ERROR = mj_data.qvel
        QVEL_ERROR[3:6] = TARGET_ANG_VEL - mj_data.qvel[3:6]
        QVEL_ERROR[JOINT_DOF_IDX] = (
            TARGET_JOINT_VELOCITIES - mj_data.qvel[JOINT_DOF_IDX]
        )
        dx = np.hstack((dq, QVEL_ERROR)).T
        desired_torques = ctrl0 - K @ dx

        new_positions = torques_to_positions(
            mj_data.qpos, mj_data.qvel, desired_torques
        )
        return new_positions
    else:
        return joint_positions


if __name__ == "__main__":
    # ----------- SETUP ENVIRONMENT -----------
    env = CPUEnv(
        xml_path=SIM_XML_PATH,
        randomization_factor=0,
        enable_rendering=True,
    )

    renderer = mujoco.Renderer(mj_model, 720, 1080)
    done = False
    joint_position_targets = np.zeros(12)
    env.reset()
    while env.data.time < 2:
        obs, _, done, _, _ = env.step(joint_position_targets)
        env.render("human")

        joint_positions = obs[: len(JOINT_NAMES)]
        joint_velocities = obs[len(JOINT_NAMES) : 2 * len(JOINT_NAMES)]
        torso_ang_vel = obs[2 * len(JOINT_NAMES) : 3 + 2 * len(JOINT_NAMES)]
        torso_quat = env.torso_quat

        joint_position_targets = compute_optimal_joint_targets(
            joint_positions,  # radians
            joint_velocities,  # radians / s
            torso_ang_vel,  # local angular velocity, rad / s
            torso_quat,  # quaternion in WXYZ form of torso
        )

    renderer.close()
