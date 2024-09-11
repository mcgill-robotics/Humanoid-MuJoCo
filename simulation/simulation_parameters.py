# DOMAIN RANDOMIZATION TECHNIQUES FROM https://arxiv.org/pdf/2304.13653.pdf

### STATE/DOMAIN RANDOMIZATION PARAMETERS
# FRICTION
FLOOR_FRICTION_MIN_MULTIPLIER = 0.5
FLOOR_FRICTION_MAX_MULTIPLIER = 1.0
# SENSOR/ACTION DELAY
MIN_DELAY = 0.01  # s, applies to observations and actions
MAX_DELAY = 0.05  # s, applies to observations and actions
# MASS/DYNAMICS
MAX_MASS_CHANGE_PER_LIMB = 0.05  # kg
MAX_EXTERNAL_MASS_ADDED = 0.2  # kg
# FORCE PERTURBATIONS
MIN_EXTERNAL_FORCE_DURATION = 0.05  # s
MAX_EXTERNAL_FORCE_DURATION = 0.15  # s
MIN_EXTERNAL_FORCE_MAGNITUDE = 5  # N
MAX_EXTERNAL_FORCE_MAGNITUDE = 15  # N
MIN_EXTERNAL_FORCE_INTERVAL = 1  # s
MAX_EXTERNAL_FORCE_INTERVAL = 3  # s
# INITIAL OFFSETS
JOINT_INITIAL_OFFSET_MIN = 0.0  # rad
JOINT_INITIAL_OFFSET_MAX = 0.2  # rad
# POSITIONAL INITIAL OFFSETS
XY_POSITION_INITIAL_OFFSET_MIN = 0.0  # m
XY_POSITION_INITIAL_OFFSET_MAX = 0.0  # m
Z_POSITION_INITIAL_OFFSET_MIN = 0.0  # m
Z_POSITION_INITIAL_OFFSET_MAX = 0.1  # m
QUAT_INITIAL_OFFSET_MIN = 0.0
QUAT_INITIAL_OFFSET_MAX = 0.1
# JOINT PROPERTIES
JOINT_ARMATURE_MAX_CHANGE = 0.0005  # kg m2
JOINT_RANGE_MAX_CHANGE = 0.1  # radians
JOINT_STIFFNESS_MAX_CHANGE = 0  # 5  # unit ?
JOINT_MARGIN_MAX_CHANGE = 0.05  # radians
JOINT_FORCE_LIMIT_MAX_CHANGE = 0.05  # N m
JOINT_P_GAIN = 2
JOINT_P_GAIN_MAX_CHANGE = 0.5
# SENSOR NOISE
JOINT_ANGLE_NOISE_STDDEV = 2  # degrees
JOINT_VELOCITY_NOISE_STDDEV = 5  # deg/s
IMU_NOISE_STDDEV = 5  # degrees
GYRO_NOISE_STDDEV = 2  # degrees / s
VELOCIMETER_NOISE_STDDEV = 0.1  # m/s
# ACTION NOISE
JOINT_ACTION_NOISE_STDDEV = 0.5  # degrees
# FLOOR HEIGHT NOISE
MIN_FLOOR_BUMP_HEIGHT = 0.0
MAX_FLOOR_BUMP_HEIGHT = 0.1

# SIMULATION PARAMETERS
CONTROL_FREQUENCY = 500  # Hz
TIMESTEP = 0.002
MAX_SIM_TIME = -1  # let the simulation run indefinitely
MAX_SIM_TIME_STANDUP = 10.0  # s
MIN_FORCE_FOR_CONTACT = 0.0  # N
USE_POTENTIAL_REWARDS = False

X_INITIAL_POS = 0.0  # m
Y_INITIAL_POS = 0.0  # m
Z_INITIAL_POS = -0.375  # m
Z_INITIAL_POS_STANDUP = -0.6  # m
INITIAL_QUAT = [
    -0.70710678,
    0.0,
    0.0,
    0.70710678,
]
INITIAL_QUAT_STANDUP = [-0.5, -0.5, 0.5, 0.5]

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
    "right_shoulder_roll",
    "right_elbow",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    # "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    # "left_ankle_pitch",
    # "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    # "right_ankle_pitch",
    # "torso_yaw",
    # "torso_roll",
]
PRESSURE_GEOM_NAMES = [
    "pressure_geom_L",
    "pressure_geom_R",
    # "pressure_geom_LLB",
    # "pressure_geom_LRB",
    # "pressure_geom_LRF",
    # "pressure_geom_LLF",
    # "pressure_geom_RLB",
    # "pressure_geom_RRB",
    # "pressure_geom_RRF",
    # "pressure_geom_RLF",
]
JOINT_LIMITS = {
    "right_shoulder_pitch": [-1.5708, -1.5708],
    "right_shoulder_roll": [-1.5708, 0],
    "right_elbow": [-1.5708, 0],
    "left_shoulder_pitch": [-1.5708, -1.5708],
    "left_shoulder_roll": [0, -1.5708],
    "left_elbow": [-1.5708, 0],
    "left_hip_roll": [-1.5708, -1.5708],
    "left_hip_pitch": [-1.5708, -1.5708],
    "left_knee": [0, -1.5708],
    "right_hip_roll": [-1.5708, -1.5708],
    "right_hip_pitch": [-1.5708, -1.5708],
    "right_knee": [0, -1.5708],
}
TORSO_BODY_NAME = "humanoid_world_link"
FREE_JOINT_NAME = "humanoid_free"
NON_ROBOT_GEOMS = ["floor", "X_axis_ref", "Y_axis_ref", "Z_axis_ref"]
