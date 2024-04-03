# DOMAIN RANDOMIZATION TECHNIQUES FROM https://arxiv.org/pdf/2304.13653.pdf

# Specifically, we randomized the floor friction (0.5 to
# 1.0), joint angular offsets (±2.9°), and varied the orientation (up to 2°) and position (up to 5 mm)
# of the IMU, and attached a random external mass (up to 0.5 kg) to a randomly chosen location on
# the robot torso. We also added random time delays (10 ms to 50 ms) to the observations to emulate
# latency in the control loop. The values were resampled in the beginning of each episode, and then
# kept constant for the whole episode. In addition to domain randomization, we found that applying
# random perturbations to the robot during training substantially improved the robustness of the agent,
# leading to better transfer. Specifically, we applied an external impulse force 5 N m to 15 N m lasting
# for 0.05 s to 0.15 s to a randomly selected point on the torso every 1 s to 3 s

# MUJOCO REF FRAME: NWU

### STATE/DOMAIN RANDOMIZATION PARAMETERS
FLOOR_FRICTION_MIN_MULTIPLIER = 0.5
FLOOR_FRICTION_MAX_MULTIPLIER = 1.0
MIN_DELAY = 0.01 #s, applies to observations and actions
MAX_DELAY = 0.05 #s, applies to observations and actions
MAX_MASS_CHANGE_PER_LIMB = 0.25 #kg
MAX_EXTERNAL_MASS_ADDED = 0.5 #kg
MIN_EXTERNAL_FORCE_DURATION = 0.05 #s
MAX_EXTERNAL_FORCE_DURATION = 0.15 #s
MIN_EXTERNAL_FORCE_MAGNITUDE = 5 #N
MAX_EXTERNAL_FORCE_MAGNITUDE = 15 #N
MIN_EXTERNAL_FORCE_INTERVAL = 1 #s
MAX_EXTERNAL_FORCE_INTERVAL = 3 #s
JOINT_INITIAL_CTRL_OFFSET_MIN = 0.1 # degrees 
JOINT_INITIAL_CTRL_OFFSET_MAX = 0.5 # degrees
JOINT_ARMATURE_MAX_CHANGE = 0.0005 # kg m2
JOINT_RANGE_MAX_CHANGE = 0.05 # radians
JOINT_STIFFNESS_MAX_CHANGE = 0.05 # unit ? TODO
JOINT_MARGIN_MAX_CHANGE = 0.05 # radians
JOINT_FORCE_LIMIT_MAX_CHANGE = 0.025 # N m
JOINT_ANGLE_NOISE_STDDEV = 2 # degrees
JOINT_VELOCITY_NOISE_STDDEV = 5 # deg/s
IMU_NOISE_STDDEV = 2 # degrees
GYRO_NOISE_STDDEV = 1 # degrees / s
ACCELEROMETER_NOISE_STDDEV = 0.05 # m/s^2
PRESSURE_SENSOR_NOISE_STDDEV = 1 #N
VELOCIMETER_NOISE_STDDEV = 0.05 # m/s

desired_control_frequency = 100
timestep = 0.001
physics_steps_per_control_step = int((1.0/desired_control_frequency) // timestep)
max_simulation_time = 10 # seconds
MAX_CONTROL_INPUT_VELOCITY = 2 # m/s ~ 7.2 mph (4.5 km/h)
USE_CONTROL_INPUTS = True # if false, only the standing behaviour is trained
USE_POTENTIAL_REWARDS = False

### URDF REFERENCE NAMES

JOINT_NAMES = ["right_shoulder_pitch", "right_elbow", "left_shoulder_pitch", "left_elbow", "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "torso_yaw", "torso_roll"]
JOINT_ACTUATOR_NAMES = JOINT_NAMES
PRESSURE_GEOM_NAMES = ["pressure_geom_LLB", "pressure_geom_LRB", "pressure_geom_LRF", "pressure_geom_LLF", "pressure_geom_RLB", "pressure_geom_RRB", "pressure_geom_RRF", "pressure_geom_RLF"]
TORSO_BODY_NAME = 'humanoid'
NON_ROBOT_GEOMS = ['floor']