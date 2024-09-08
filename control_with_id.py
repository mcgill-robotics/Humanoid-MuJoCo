import pinocchio as pin
import numpy as np
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation.simulation_parameters import CONTROL_FREQUENCY, JOINT_NAMES
from scipy.optimize import minimize

# GLOBAL VARIABLES
PLANNING_HORIZON = 50  # steps
DT = 1 / CONTROL_FREQUENCY
VISUALIZE_URDF = True
INVERTED_JOINTS = ["left_knee"]
PRINT_JOINT_INFO = True
JOINT_VELOCITY_LIMIT = 10

# Load the robot model
robot = pin.RobotWrapper.BuildFromURDF("id_urdf/humanoid.urdf")
TORSO_FRAME_ID = robot.model.getFrameId("torso")
q = np.concatenate([np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(12)])
pin.forwardKinematics(robot.model, robot.data, q, np.zeros(robot.model.nv))
pin.updateFramePlacements(robot.model, robot.data)
geometry_model = pin.buildGeomFromUrdf(
    robot.model, "id_urdf/humanoid.urdf", "id_urdf/", pin.GeometryType.COLLISION
)
geometry_model.addAllCollisionPairs()
geometry_data = pin.GeometryData(geometry_model)
FLOOR_GEOM_ID = None
for geom_id in range(len(geometry_model.geometryObjects)):
    geom_object = geometry_model.geometryObjects[geom_id]
    if "floor" in geom_object.name:
        FLOOR_GEOM_ID = geom_id
        break
if VISUALIZE_URDF:
    from pinocchio.visualize import MeshcatVisualizer

    robot.setVisualizer(MeshcatVisualizer())
    robot.initViewer(open=True)
    robot.loadViewerModel()
    # while True:
    #     time.sleep(1)


def get_joint_info():
    global INVERTED_JOINT_Q_IDX
    global INVERTED_JOINT_V_IDX
    global JOINT_NAMES_URDF_Q
    global JOINT_NAMES_URDF_V
    JOINT_NAMES_URDF_Q = [""] * robot.model.nq
    JOINT_NAMES_URDF_V = [""] * robot.model.nv
    if PRINT_JOINT_INFO:
        print("Joint Information:")
        print("------------------")
    for joint_id in range(
        1, robot.model.njoints
    ):  # Start from 1 to skip the "universe" joint
        joint = robot.model.joints[joint_id]
        joint_name = robot.model.names[joint_id]
        joint_type = joint.shortname()
        nq = joint.nq  # Number of position variables
        nv = joint.nv  # Number of velocity variables

        JOINT_NAMES_URDF_Q[joint.idx_q] = joint_name
        JOINT_NAMES_URDF_V[joint.idx_v] = joint_name

        if PRINT_JOINT_INFO:
            print(f"Joint {joint_id}:")
            print(f"  Name: {joint_name}")
            print(f"  Type: {joint_type}")
            print(f"  DOFs: {nv} (nq: {nq}, nv: {nv})")
            print(f"  Joint Index: {joint.idx_q}")
            print(f"  Joint Velocity Index: {joint.idx_v}")
            print()

    INVERTED_JOINT_Q_IDX = [JOINT_NAMES_URDF_Q.index(j) for j in INVERTED_JOINTS]
    INVERTED_JOINT_V_IDX = [JOINT_NAMES_URDF_V.index(j) for j in INVERTED_JOINTS]

    if PRINT_JOINT_INFO:
        print(f"Total DOFs: {robot.model.nv}")
        print(f"Configuration vector size: {robot.model.nq}")


def get_dist_from_ground(q):
    min_dist = np.inf
    pin.computeDistances(robot.model, robot.data, geometry_model, geometry_data, q)
    for i in range(len(geometry_model.collisionPairs)):
        c1 = geometry_model.collisionPairs[i].first
        c2 = geometry_model.collisionPairs[i].second
        dist = geometry_data.distanceResults[i].min_distance
        if (c1 == FLOOR_GEOM_ID or c2 == FLOOR_GEOM_ID) and c1 != c2:
            min_dist = min(min_dist, dist)
    return min_dist


def optimal_control(q, v, target_torso_trajectory):
    """
    q: current joint configuration of the robot
    v: current velocities of the robot
    target_torso_trajectory: desired trajectory of the torso of the robot of shape [HORIZON, 7]
                              where the second axis represents position (3) and quaternion (4)
    robot: Pinocchio robot model
    """
    # compute joint velocities that minimize deviation from the target torso trajectory over the next HORIZON steps
    # use pinocchio to calculate forward/inverse dynamics
    # constrain joint velocities to be within -JOINT_VELOCITY_LIMIT and JOINT_VELOCITY_LIMIT
    # minimize control effort
    # keep the implementation simple
    # global variables available:
        # robot, DT, PLANNING_HORIZON, JOINT_VELOCITY_LIMIT

def compute_optimal_joint_targets(
    joint_positions, joint_velocities, torso_ang_vel, torso_orientation
):
    q = np.concatenate(
        [
            np.zeros(3),
            np.array(
                [
                    -torso_orientation[2],
                    torso_orientation[1],
                    torso_orientation[3],
                    torso_orientation[0],
                ]
            ),
            np.zeros(12),
        ]
    )
    v = np.concatenate([torso_ang_vel, np.zeros(3), np.zeros(12)])
    # convert mujoco to URDF
    q[INVERTED_JOINT_Q_IDX] *= -1
    v[INVERTED_JOINT_V_IDX] *= -1
    for joint in JOINT_NAMES:
        q[JOINT_NAMES_URDF_Q.index(joint)] = joint_positions[JOINT_NAMES.index(joint)]
        v[JOINT_NAMES_URDF_V.index(joint)] = joint_velocities[JOINT_NAMES.index(joint)]
    q[2] -= get_dist_from_ground(q)
    TORSO_TARGET_TRAJECTORY = np.array([
        np.concatenate([q[0:2], np.zeros(1), np.array([0, 0, 0, 1])])
    ] * PLANNING_HORIZON)

    if VISUALIZE_URDF:
        robot.display(q)

    optimal_joint_velocities = optimal_control(q, v, TORSO_TARGET_TRAJECTORY)

    optimal_joint_velocities[np.array([i - 6 for i in INVERTED_JOINT_V_IDX])] *= -1

    return joint_positions + optimal_joint_velocities * DT


if __name__ == "__main__":
    get_joint_info()
    # ----------- SETUP ENVIRONMENT -----------
    env = CPUEnv(
        xml_path=SIM_XML_PATH,
        reward_fn=SELECTED_REWARD_FUNCTION,
        randomization_factor=0,
        enable_rendering=True,
    )
    try:
        obs, _ = env.reset()
    except:
        obs = env.reset()

    while True:
        done = False
        obs, _ = env.reset()
        while env.data.time < 2:
            joint_positions = obs[:12]  # rad
            joint_velocities = obs[12:24]  # rad / s
            torso_ang_vel = obs[
                24:27
            ]  # rad / s, local angular velocity of the torso of the robot

            joint_position_targets = compute_optimal_joint_targets(
                joint_positions,
                joint_velocities,
                torso_ang_vel,
                env.torso_quat,
            )
            obs, reward, done, _, _ = env.step(joint_position_targets)
            env.render("human")
