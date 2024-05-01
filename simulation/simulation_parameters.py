# DOMAIN RANDOMIZATION TECHNIQUES FROM https://arxiv.org/pdf/2304.13653.pdf

### STATE/DOMAIN RANDOMIZATION PARAMETERS
# FRICTION
FLOOR_FRICTION_MIN_MULTIPLIER = 1.0  # 0.5
FLOOR_FRICTION_MAX_MULTIPLIER = 1.0
# SENSOR/ACTION DELAY
MIN_DELAY = 0  # 0.01  # s, applies to observations and actions
MAX_DELAY = 0  # 0.05  # s, applies to observations and actions
# MASS/DYNAMICS
MAX_MASS_CHANGE_PER_LIMB = 0  # 0.25  # kg
MAX_EXTERNAL_MASS_ADDED = 0  # 0.5  # kg
# FORCE PERTURBATIONS
MIN_EXTERNAL_FORCE_DURATION = 0.05  # s
MAX_EXTERNAL_FORCE_DURATION = 0.15  # s
MIN_EXTERNAL_FORCE_MAGNITUDE = 0  # 5  # N
MAX_EXTERNAL_FORCE_MAGNITUDE = 0  # 15  # N
MIN_EXTERNAL_FORCE_INTERVAL = 1  # s
MAX_EXTERNAL_FORCE_INTERVAL = 3  # s
# INITIAL OFFSETS
JOINT_INITIAL_OFFSET_MIN = 0.05  # rad
JOINT_INITIAL_OFFSET_MAX = 0.1  # rad
# JOINT PROPERTIES
JOINT_ARMATURE_MAX_CHANGE = 0  # 0.0005  # kg m2
JOINT_RANGE_MAX_CHANGE = 0  # 0.05  # radians
JOINT_STIFFNESS_MAX_CHANGE = 0  # 0.05  # unit ?
JOINT_MARGIN_MAX_CHANGE = 0  # 0.05  # radians
JOINT_FORCE_LIMIT_MAX_CHANGE = 0  # 0.025  # N m
# SENSOR NOISE
JOINT_ANGLE_NOISE_STDDEV = 0  # 2  # degrees
JOINT_VELOCITY_NOISE_STDDEV = 0  # 5  # deg/s
IMU_NOISE_STDDEV = 0  # 2  # degrees
GYRO_NOISE_STDDEV = 0  # 1  # degrees / s
ACCELEROMETER_NOISE_STDDEV = 0  # 0.05  # m/s^2
VELOCIMETER_NOISE_STDDEV = 0  # 0.05  # m/s

# SIMULATION PARAMETERS
CONTROL_FREQUENCY = 100
TIMESTEP = 0.001
MAX_SIM_TIME = -1  # let the simulation run indefinitely
TERMINAL_FRACTION_RESET_THRESHOLD = 0.1  # applies only to GPUVecEnv NOTE: setting this >= 1 will make the environment terminate only from time limit
MIN_FORCE_FOR_CONTACT = 0.0  # N
USE_POTENTIAL_REWARDS = False

# CONTROL INPUT PARAMETERS
CONTROL_INPUT_MAX_VELOCITY = 2  # m/s ~ 7.2 mph (4.5 km/h), can go from -2 to 2 m/s
CONTROL_INPUT_MAX_YAW = 3.14159  # radians, can go from -180 to 180 degrees
USE_CONTROL_INPUTS = False  # if false, all zero control inputs
RANDOMIZATION_FACTOR_AFFECTS_CONTROL_INPUT = False


#####################
### IGNORE  BELOW ###
#####################


# DO NOT CHANGE
PHYSICS_STEPS_PER_CONTROL_STEP = int((1.0 / CONTROL_FREQUENCY) // TIMESTEP)
### URDF REFERENCE NAMES (DEFINED FROM MUJOCO .xml FILES, DO NOT CHANGE UNLESS .xml CHANGES)
JOINT_NAMES = [
    "right_shoulder_pitch",
    "right_elbow",
    "left_shoulder_pitch",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "torso_yaw",
    "torso_roll",
]
JOINT_ACTUATOR_NAMES = JOINT_NAMES
PRESSURE_GEOM_NAMES = [
    "pressure_geom_LLB",
    "pressure_geom_LRB",
    "pressure_geom_LRF",
    "pressure_geom_LLF",
    "pressure_geom_RLB",
    "pressure_geom_RRB",
    "pressure_geom_RRF",
    "pressure_geom_RLF",
]
TORSO_BODY_NAME = "humanoid"
NON_ROBOT_GEOMS = ["floor"]
