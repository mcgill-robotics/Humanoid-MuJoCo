import pinocchio as pin
import numpy as np
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation.simulation_parameters import CONTROL_FREQUENCY
from scipy.optimize import minimize

# Constants
TORSO_ANG_VEL_TARGET = np.array([0, 0, 0])  # Euler angles, radians (roll, pitch, yaw) (NWU)
JOINT_VELOCITY_TARGETS = [0] * 12  # All zeros for standing position
URDF_PATH = "id_urdf/world.urdf"

# Load the robot model
robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH)


def print_joint_info():
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

        print(f"Joint {joint_id}:")
        print(f"  Name: {joint_name}")
        print(f"  Type: {joint_type}")
        print(f"  DOFs: {nv} (nq: {nq}, nv: {nv})")
        print(f"  Joint Index: {joint.idx_q}")
        print(f"  Joint Velocity Index: {joint.idx_v}")
        print()

    print(f"Total DOFs: {robot.model.nv}")
    print(f"Configuration vector size: {robot.model.nq}")


def compute_optimal_joint_targets(
    joint_positions,
    joint_velocities,
    torso_ang_vel,
    torso_orientation,
):
    q = np.concatenate([np.concatenate([np.zeros(3), np.array([torso_orientation[0], torso_orientation[1], torso_orientation[2], torso_orientation[3]])]), joint_positions])  # Assume floating base
    v = np.concatenate([torso_ang_vel, np.zeros(3), joint_velocities])

    # Set Pinocchio model state
    robot.q = q
    robot.v = v

    # Calculate forward kinematics
    pin.forwardKinematics(robot.model, robot.data, q, v)
    pin.centerOfMass(robot.model, robot.data, q, v)

    # Jacobian for torso orientation and center of mass
    jac_torso = pin.getFrameJacobian(
        robot.model,
        robot.data,
        1,  # Assuming frame ID 1 for torso
        pin.ReferenceFrame.LOCAL
    )[:3, 6:]  # Floating base

    jac_com = pin.jacobianCenterOfMass(robot.model, robot.data)[0:2, 6:]  # XY CoM

    def objective(x):
        joint_vel_error = x[:12] - JOINT_VELOCITY_TARGETS
        torso_ang_vel_error = x[12:] - TORSO_ANG_VEL_TARGET
        return np.sum(joint_vel_error**2) + np.sum(torso_ang_vel_error**2)

    def constraints(x):
        joint_vel = x[:12]
        torso_ang_vel = x[12:]
        
        torso_constraint = jac_torso @ joint_vel + torso_ang_vel - TORSO_ANG_VEL_TARGET
        com_constraint = jac_com @ joint_vel + pin.centerOfMass(robot.model, robot.data)[:2]
        
        return np.concatenate([torso_constraint, com_constraint])

    # Initial guess
    x0 = np.concatenate([joint_velocities, torso_ang_vel])

    # Solve the optimization problem
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraints},
        options={'ftol': 1e-6, 'disp': False}
    )

    if result.success:
        optimal_joint_velocities = result.x[:12]
        optimal_torso_ang_vel = result.x[12:]
    else:
        print("Optimization failed")
        optimal_joint_velocities = joint_velocities
        optimal_torso_ang_vel = torso_ang_vel

    # Integrate joint velocities to obtain joint positions
    dt = 1 / CONTROL_FREQUENCY
    print(optimal_joint_velocities)
    return joint_positions + -optimal_joint_velocities * dt


if __name__ == "__main__":
    print_joint_info()
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
        while not done:
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
