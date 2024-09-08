import pinocchio as pin
import numpy as np
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION
from simulation.simulation_parameters import CONTROL_FREQUENCY
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Constants
TORSO_ANG_VEL_TARGET = np.array([0, 0, 0])  # Euler angles, radians (roll, pitch, yaw) (NWU)
JOINT_VELOCITY_TARGETS = [0] * 12  # All zeros for standing position
MAX_ZMP = np.array([0.5, 0.5])  # Adjust based on your robot's support polygon
MIN_ZMP = np.array([-0.5, -0.5])
DT = 1 / CONTROL_FREQUENCY
DEBUG = False

# Load the robot model
robot = pin.RobotWrapper.BuildFromURDF("id_urdf/world.urdf")
TORSO_FRAME_ID = robot.model.getFrameId("torso")

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

def compute_zmp(q, v, a):
    pin.forwardKinematics(robot.model, robot.data, q, v)
    pin.centerOfMass(robot.model, robot.data, q, v, a)
    c = robot.data.com[0]
    cdd = pin.getFrameAcceleration(robot.model, robot.data, TORSO_FRAME_ID).linear
    g = np.array([0, 0, -9.81])
    f = robot.data.mass[0] * (cdd - g)
    zmp = c - np.array([f[1] * (c[2] / f[2]), -f[0] * (c[2] / f[2]), 0])
    return zmp[:2]

def objective(x, q, v, joint_velocities):
    joint_vel = x
    joint_vel_error = joint_vel - JOINT_VELOCITY_TARGETS
    
    # Compute the resulting torso angular velocity
    J_torso = pin.computeFrameJacobian(robot.model, robot.data, q, TORSO_FRAME_ID, pin.ReferenceFrame.LOCAL)[:3, 6:]
    torso_ang_vel = J_torso @ joint_vel + v[:3]  # v[:3] is the current torso angular velocity
    torso_ang_vel_error = torso_ang_vel - TORSO_ANG_VEL_TARGET
    
    # Weighted sum of joint velocity error and torso angular velocity error
    return np.sum(joint_vel_error**2) + 10 * np.sum(torso_ang_vel_error**2)

def constraints(x, q, v, joint_velocities):
    joint_vel = x
    joint_acc = (joint_vel - joint_velocities) / DT
    
    new_q = pin.integrate(robot.model, q, np.concatenate([np.zeros(6), joint_vel * DT]))
    J_torso = pin.computeFrameJacobian(robot.model, robot.data, q, TORSO_FRAME_ID, pin.ReferenceFrame.LOCAL)[:3, 6:]
    new_torso_ang_vel = J_torso @ joint_vel + v[:3]
    new_v = np.concatenate([new_torso_ang_vel, v[3:6], joint_vel])
    
    zmp = compute_zmp(new_q, new_v, np.concatenate([np.zeros(6), joint_acc]))
    
    zmp_constraint = np.array([
        MAX_ZMP[0] - zmp[0],
        zmp[0] - MIN_ZMP[0],
        MAX_ZMP[1] - zmp[1],
        zmp[1] - MIN_ZMP[1]
    ])
    
    return zmp_constraint

def compute_optimal_joint_targets(joint_positions, joint_velocities, torso_ang_vel, torso_orientation):
    q = np.concatenate([np.zeros(3), torso_orientation, joint_positions])
    v = np.concatenate([torso_ang_vel, np.zeros(3), joint_velocities])

    robot.q = q
    robot.v = v
    pin.forwardKinematics(robot.model, robot.data, q, v)
    pin.computeCentroidalMomentum(robot.model, robot.data, q, v)

    x0 = joint_velocities
    
    bounds = [(-10, 10) for _ in range(12)]  # Adjust bounds as needed

    result = minimize(
        objective,
        x0,
        args=(q, v, joint_velocities),
        method='SLSQP',
        bounds=bounds,
        constraints={
            'type': 'ineq',
            'fun': constraints,
            'args': (q, v, joint_velocities)
        },
        options={'ftol': 1e-6, 'disp': False, 'maxiter': 100}
    )

    if result.success:
        optimal_joint_velocities = result.x
    else:
        print(f"Optimization result: {result.message}")
        print(f"Final objective value: {result.fun}")
        print(f"Constraint violations: {constraints(result.x, q, v, joint_velocities)}")
        optimal_joint_velocities = np.zeros(12)

    if DEBUG:
        J_torso = pin.computeFrameJacobian(robot.model, robot.data, q, TORSO_FRAME_ID, pin.ReferenceFrame.LOCAL)[:3, 6:]
        resulting_torso_ang_vel = J_torso @ optimal_joint_velocities + torso_ang_vel
        debug_info = {
            'initial_state': {'q': q, 'v': v},
            'optimal_velocities': {'joint': optimal_joint_velocities, 'torso': resulting_torso_ang_vel},
            'zmp': compute_zmp(q, v, np.concatenate([np.zeros(6), (optimal_joint_velocities - joint_velocities) / DT])),
            'objective_value': result.fun,
            'constraint_violation': np.max(np.abs(constraints(result.x, q, v, joint_velocities)))
        }
        plot_debug_info(debug_info)
    
    print(optimal_joint_velocities)

    return joint_positions + optimal_joint_velocities * DT

def plot_debug_info(debug_info):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    ax1.plot(debug_info['zmp'][0], debug_info['zmp'][1], 'ro', label='ZMP')
    ax1.plot([ZMP_MIN[0], ZMP_MAX[0], ZMP_MAX[0], ZMP_MIN[0], ZMP_MIN[0]],
             [ZMP_MIN[1], ZMP_MIN[1], ZMP_MAX[1], ZMP_MAX[1], ZMP_MIN[1]], 'b-', label='Support Polygon')
    ax1.set_title('ZMP and Support Polygon')
    ax1.legend()
    ax1.set_aspect('equal')
    
    ax2.bar(range(12), debug_info['optimal_velocities']['joint'], label='Optimal')
    ax2.bar(range(12), JOINT_VELOCITY_TARGETS, alpha=0.5, label='Target')
    ax2.set_title('Joint Velocities')
    ax2.set_xlabel('Joint Index')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    
    ax3.bar(range(3), debug_info['optimal_velocities']['torso'], label='Resulting')
    ax3.bar(range(3), TORSO_ANG_VEL_TARGET, alpha=0.5, label='Target')
    ax3.set_title('Torso Angular Velocity')
    ax3.set_xlabel('Axis')
    ax3.set_ylabel('Angular Velocity')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


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
        while env.data.time < 2:
            joint_positions = obs[:12]  # rad
            joint_velocities = obs[12:24]  # rad / s
            torso_ang_vel = obs[
                24:27
            ]  # rad / s, local angular velocity of the torso of the robot
            
            JOINT_VELOCITY_TARGETS = -joint_positions
            TORSO_ANG_VEL_TARGET = -torso_ang_vel
            print(torso_ang_vel)
            joint_position_targets = compute_optimal_joint_targets(
                joint_positions,
                joint_velocities,
                torso_ang_vel,
                env.torso_quat,
            )
            obs, reward, done, _, _ = env.step(joint_position_targets)
            env.render("human")
