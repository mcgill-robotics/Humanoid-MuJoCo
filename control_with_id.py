import pinocchio as pin
import numpy as np
from pinocchio.utils import zero
import gym
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.reward_functions import SELECTED_REWARD_FUNCTION

# ----------- HYPERPARAMETERS -----------
KP = 100  # Proportional gain for PD control (if using joint position)
KD = 10  # Derivative gain for PD control (if using joint position)
LAMBDA_REG = 0.01  # Regularization for torque magnitude in cost function
OPTIMIZATION_STEP = 0.01  # Step size for optimizing joint positions
MAX_STEPS = 10000  # Maximum number of simulation steps
TORQUE_SCALE = 0.5  # Scale factor for applying torques (to avoid over-saturation)
RANDOMIZATION_FACTOR = 0

# ----------- SETUP ENVIRONMENT -----------
env = CPUEnv(
    xml_path=SIM_XML_PATH,
    reward_fn=SELECTED_REWARD_FUNCTION,
    randomization_factor=RANDOMIZATION_FACTOR,
    enable_rendering=True,
)
try:
    obs, _ = env.reset()
except:
    obs = env.reset()

# ----------- LOAD URDF WITH PINOCCHIO -----------
model = pin.buildModelFromUrdf("simulation/assets/humanoid_urdf/humanoid.urdf")
data = model.createData()

# Gravity setup for inverse dynamics
gravity = np.array([0, 0, -9.81])
model.gravity.linear = gravity

# Define robot configuration variables
nq = model.nq  # Number of joint positions
nv = model.nv  # Number of joint velocities

# ----------- SPECIFY DESIRED POSE -----------
# Define a desired standing pose (this could be customized as needed)
q_desired = np.zeros(
    nq
)  # Assuming a neutral upright standing pose (all joints at zero)


# ----------- INVERSE DYNAMICS CONTROL FUNCTION -----------
def inverse_dynamics_control(q_current, v_current):
    """
    Computes the joint torques based on inverse dynamics and an optimization step.
    """
    # Assume zero acceleration for a balancing task
    a = zero(nv)

    # Calculate inverse dynamics (torques required to maintain the current state)
    tau = pin.rnea(model, data, q_current, v_current, a)

    # Optimize joint positions to minimize distance from desired pose
    joint_diff = q_current - q_desired
    cost = np.linalg.norm(joint_diff) ** 2 + LAMBDA_REG * np.linalg.norm(tau) ** 2

    # Gradient descent step to minimize joint distance
    q_optimized = q_current - OPTIMIZATION_STEP * joint_diff

    return q_optimized, tau

# Print the names and indices of joints
print("Joint names and indices:")

# Access joint names from the robot model's metadata
for joint_id in range(model.njoints):
    joint_name = model.names[joint_id] if joint_id < len(model.names) else f"Joint_{joint_id}"
    print(f"Joint index {joint_id}: {joint_name}")

# For each joint, you can also print the position (nq) and velocity (nv) DOFs
print("\nJoint DOFs:")
for joint_id in range(model.njoints):
    joint = model.joints[joint_id]
    nq = joint.nq
    nv = joint.nv
    print(f"Joint index {joint_id} (name {model.names[joint_id]}):")
    print(f"  Number of position DOFs: {nq}")
    print(f"  Number of velocity DOFs: {nv}")

# ----------- CONTROL LOOP -----------
for step in range(MAX_STEPS):
    # Get the current state from the environment
    q_current = obs[:12]
    v_current = obs[12:24]

    # Compute the optimized joint positions and inverse dynamics torques
    q_new, tau = inverse_dynamics_control(q_current, v_current)

    # Apply the torques to the robot (scaled to avoid large torques)
    action = TORQUE_SCALE * tau

    # Step the environment with the calculated torques
    obs, reward, done, info = env.step(action)
    env.render("human")

    # Break the loop if the environment signals done
    if done:
        break

# Close the environment after the loop
env.close()
