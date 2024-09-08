import numpy as np
import pinocchio as pin
from qpsolvers import solve_qp
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION

# Constants
TORSO_ORIENTATION_TARGET = np.array([0, 0, 0])  # euler angles, radians (roll, pitch, yaw) (NWU)
JOINT_POSITION_TARGETS = np.zeros(12)  # all zeros for standing position
URDF_PATH = "id_urdf/world.urdf"

# Load the robot model
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()

def print_joint_info():
    print("Joint Information:")
    print("------------------")
    for joint_id in range(1, model.njoints):  # Start from 1 to skip the "universe" joint
        joint = model.joints[joint_id]
        joint_name = model.names[joint_id]
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

    print(f"Total DOFs: {model.nv}")
    print(f"Configuration vector size: {model.nq}")

def compute_optimal_joint_targets(joint_positions, joint_velocities, local_ang_vel, local_gravity_direction, torso_orientation_target, joint_position_targets):
    # Update the robot configuration
    q = np.concatenate([np.zeros(7), joint_positions])  # Assume floating base
    v = np.concatenate([local_ang_vel, np.zeros(3), joint_velocities])
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    
    # Compute Jacobians
    J = pin.computeJointJacobians(model, data, q)
    
    # Compute the center of mass Jacobian
    Jcom = pin.jacobianCenterOfMass(model, data, q)
    
    # Compute the current COM position
    com = pin.centerOfMass(model, data, q)
    
    # Compute the current orientation error
    current_orientation = pin.rpy.matrixToRpy(data.oMf[model.getFrameId("torso")].rotation)
    orientation_error = pin.difference(current_orientation, torso_orientation_target)
    
    # Compute the current joint position error
    joint_position_error = joint_positions - joint_position_targets
    
    # Set up the QP problem
    n = model.nv - 6  # Number of actuated joints
    
    # Objective function: min 0.5 * x^T * P * x + q^T * x
    P = np.eye(n)
    q = np.zeros(n)
    
    # Constraints: G * x <= h
    G = np.zeros((n, n))
    h = np.zeros(n)
    
    # Joint limits
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.concatenate([model.upperPositionLimit[7:] - joint_positions,
                        joint_positions - model.lowerPositionLimit[7:]])
    
    # Solve the QP problem
    joint_accelerations = solve_qp(P, q, G, h)
    
    # Compute the new joint positions
    dt = 0.01  # Assuming a control frequency of 100 Hz
    new_joint_positions = joint_positions + joint_velocities * dt + 0.5 * joint_accelerations * dt**2
    new_joint_velocities = joint_velocities + joint_accelerations * dt
    
    return new_joint_positions

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
            local_ang_vel = obs[24:27]  # rad / s, local angular velocity of the torso of the robot
            local_gravity_direction = obs[27:30]  # unit vector which points in the direction of the force of gravity, relative to the torso of the robot
            
            joint_position_targets = compute_optimal_joint_targets(
                joint_positions, joint_velocities, local_ang_vel, local_gravity_direction, TORSO_ORIENTATION_TARGET, JOINT_POSITION_TARGETS
            )
            obs, reward, done, _, _ = env.step(joint_position_targets)
            env.render("human")