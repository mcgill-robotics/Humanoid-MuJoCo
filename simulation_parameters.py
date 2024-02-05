# DOMAIN RANDOMIZATION TECHNIQUES FROM https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=HlRhFs_d3WLP

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
MAX_MASS_CHANGE_PER_LIMB = 0.05 #kg
MAX_EXTERNAL_MASS_ADDED = 0.1 #kg
MIN_EXTERNAL_FORCE_DURATION = 0.05 #s
MAX_EXTERNAL_FORCE_DURATION = 0.15 #s
MIN_EXTERNAL_FORCE_MAGNITUDE = 50 #N m ? TODO
MAX_EXTERNAL_FORCE_MAGNITUDE = 150 #N m ? TODO
MIN_EXTERNAL_FORCE_INTERVAL = 1 #s
MAX_EXTERNAL_FORCE_INTERVAL = 3 #s
JOINT_INITIAL_STATE_OFFSET_MAX = 7 # degrees
IMU_ORIENTATION_OFFSET_MAX = 10 # degrees
IMU_POS_OFFSET_MAX = 0.05 # meters
PRESSURE_SENSOR_POS_OFFSET_MAX = 0.025 # meters
JOINT_ANGLE_NOISE_STDDEV = 2 # degrees
IMU_NOISE_STDDEV = 2 # degrees
GYRO_NOISE_STDDEV = 1 # degrees
ACCELEROMETER_NOISE_STDDEV = 0.05 # G
PRESSURE_SENSOR_NOISE_STDDEV = 0.1 #N m ? TODO
VELOCIMETER_NOISE_STDDEV = 0.05 # m/s ? TODO
JOINT_DAMPING_MAX_CHANGE = 0.1 # Nm/(rad/s) ? TODO
JOINT_ARMATURE_MAX_CHANGE = 0.05 # kg m2 ? TODO
JOINT_FRICTION_MAX_CHANGE = 0.05 # ? TODO
JOINT_RANGE_MAX_CHANGE = 1 # degrees
JOINT_STIFFNESS_MAX_CHANGE = 0.05 # unit ? TODO
JOINT_MARGIN_MAX_CHANGE = 0.5 # degrees ? TODO
JOINT_FORCE_LIMIT_MAX_CHANGE = 0.05 # N m ? TODO
JOINT_PID_P_GAIN = 20 # PID gain
JOINT_PID_V_GAIN = 2 # PID gain
JOINT_PID_GAIN_MAX_CHANGE = 0.5 # PID gain delta

### URDF REFERENCE NAMES
JOINT_NAMES = ['jL5S1_rotx', 'jL5S1_roty', 'jL5S1_rotz', 'jL4L3_rotx', 'jL4L3_roty', 'jL4L3_rotz', 'jL1T12_rotx', 'jL1T12_roty', 'jL1T12_rotz', 'jT9T8_rotx', 'jT9T8_roty', 'jT9T8_rotz', 'jT1C7_rotx', 'jT1C7_roty', 'jT1C7_rotz', 'jC1Head_rotx', 'jC1Head_roty', 'jC1Head_rotz', 'jRightC7Shoulder_rotx', 'jRightC7Shoulder_roty', 'jRightC7Shoulder_rotz', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz', 'jRightElbow_rotx', 'jRightElbow_roty', 'jRightElbow_rotz', 'jRightWrist_rotx', 'jRightWrist_roty', 'jRightWrist_rotz', 'jLeftC7Shoulder_rotx', 'jLeftC7Shoulder_roty', 'jLeftC7Shoulder_rotz', 'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftShoulder_rotz', 'jLeftElbow_rotx', 'jLeftElbow_roty', 'jLeftElbow_rotz', 'jLeftWrist_rotx', 'jLeftWrist_roty', 'jLeftWrist_rotz', 'jRightHip_rotx', 'jRightHip_roty', 'jRightHip_rotz', 'jRightKnee_rotx', 'jRightKnee_roty', 'jRightKnee_rotz', 'jRightAnkle_rotx', 'jRightAnkle_roty', 'jRightAnkle_rotz', 'jRightBallFoot_rotx', 'jRightBallFoot_roty', 'jRightBallFoot_rotz', 'jLeftHip_rotx', 'jLeftHip_roty', 'jLeftHip_rotz', 'jLeftKnee_rotx', 'jLeftKnee_roty', 'jLeftKnee_rotz', 'jLeftAnkle_rotx', 'jLeftAnkle_roty', 'jLeftAnkle_rotz', 'jLeftBallFoot_rotx', 'jLeftBallFoot_roty', 'jLeftBallFoot_rotz', 'jLeftBallFoot_rotz', 'jLeftBallFoot_roty', 'jLeftBallFoot_rotx', 'jLeftAnkle_rotz']
# gotten from print([name for name in str(self.model.names).split('\\x00') if 'rot' in name])
JOINT_SENSOR_NAMES = ['jLeftBallFoot_rotz'] # TODO -> can be replaced by JOINT_NAMES when robot URDF is complete
JOINT_ACTUATOR_NAMES = ['jLeftBallFoot_rotz', 'jLeftBallFoot_roty', 'jLeftBallFoot_rotx', 'jLeftAnkle_rotz'] # TODO -> can be replaced by JOINT_SENSOR_NAMES = JOINT_NAMES when robot URDF is complete
PRESSURE_SENSOR_SITE_NAMES = ["foot_LLB"] # TODO
PRESSURE_SENSOR_NAMES = ["pressure_LLB"] # TODO

### REWARD PARAMETERS
# Velocity The magnitude of the player’s forward velocity. - 0.1
X_VELOCITY_REWARD_WEIGHT = 0.1
Y_VELOCITY_REWARD_WEIGHT = 0
Z_VELOCITY_REWARD_WEIGHT = 0
# For Velocity: set these to true to reward/penalize movement any movement on the axis (by applying abs() to velocity)
ABS_X_VELOCITY = False
ABS_Y_VELOCITY = False
ABS_Z_VELOCITY = False
# Termination A penalty, equal to −1 if the player is on the ground - 0.5
GROUNDED_PENALTY_WEIGHT = 0.5
# Upright 0 if the robot is upside down or if the tilt angle is greater
    # than 0.4 radians. Increases linearly, and is equal to +1 if the
    # tilt angle is less than 0.2 radians. - 0.02
UPRIGHT_REWARD_WEIGHT = 0.02
MIN_TILT_FOR_REWARD = 0.2
MAX_TILT_FOR_REWARD = 0.4
# Joint torque A penalty, equal to the magnitude of the torque measured at
    # the player’s knees. This discourages the player from learning
    # gaits which cause high forces on the knees, for example
    # during ground impacts, which can damage a physical robot. - 0.01
JOINT_TORQUE_PENALTY_WEIGHT = 0.01