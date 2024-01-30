import mujoco
import cv2
import numpy as np
import random
import quaternion

# STATE INFO/DOMAIN RANDOMIZATION TECHNIQUES FROM https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=HlRhFs_d3WLP

# MUJOCO REF FRAME: ENU

### PARAMETERS
FLOOR_FRICTION_MIN_MULTIPLIER = 0.5
FLOOR_FRICTION_MAX_MULTIPLIER = 1.0
JOINT_ANGLE_SENSOR_OFFSET_MAX = 2 # degrees
JOINT_INITIAL_STATE_OFFSET_MAX = 7 # degrees
MIN_DELAY = 0.01 #s, applies to observations and actions
MAX_DELAY = 0.05 #s, applies to observations and actions
MAX_MASS_CHANGE_PER_LIMB = 0.05 #kg
MAX_EXTERNAL_MASS_ADDED = 0.1 #kg
MIN_EXTERNAL_FORCE_DURATION = 0.05 #s
MAX_EXTERNAL_FORCE_DURATION = 0.15 #s
MIN_EXTERNAL_FORCE_MAGNITUDE = 2.5 #N m ?
MAX_EXTERNAL_FORCE_MAGNITUDE = 7.5 #N m ?
IMU_ORIENTATION_OFFSET_MAX = 10 # degrees
IMU_POS_OFFSET_MAX = 0.05 # meters
GYRO_NOISE_STDDEV = 0 # degrees
ACCELEROMETER_NOISE_STDDEV = 0.05 # G
PRESSURE_SENSOR_NOISE_STDDEV = 0.1 #N m ?
VELOCIMETER_NOISE_STDDEV = 0.05 # m/s
JOINT_DAMPING_MAX_CHANGE = 0.1 # Nm/(rad/s) ?
JOINT_ARMATURE_MAX_CHANGE = 0.05 # kg m2 ?
JOINT_FRICTION_MAX_CHANGE = 0.05 # ?
JOINT_RANGE_MAX_CHANGE = 1 # degrees
JOINT_STIFFNESS_MAX_CHANGE = 0.05
JOINT_MARGIN_MAX_CHANGE = 0.5 # degrees ?
PRESSURE_SENSOR_POS_OFFSET_MAX = 0.025 # meters

JOINT_NAMES = ['jL5S1_rotx', 'jL5S1_roty', 'jL5S1_rotz', 'jL4L3_rotx', 'jL4L3_roty', 'jL4L3_rotz', 'jL1T12_rotx', 'jL1T12_roty', 'jL1T12_rotz', 'jT9T8_rotx', 'jT9T8_roty', 'jT9T8_rotz', 'jT1C7_rotx', 'jT1C7_roty', 'jT1C7_rotz', 'jC1Head_rotx', 'jC1Head_roty', 'jC1Head_rotz', 'jRightC7Shoulder_rotx', 'jRightC7Shoulder_roty', 'jRightC7Shoulder_rotz', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz', 'jRightElbow_rotx', 'jRightElbow_roty', 'jRightElbow_rotz', 'jRightWrist_rotx', 'jRightWrist_roty', 'jRightWrist_rotz', 'jLeftC7Shoulder_rotx', 'jLeftC7Shoulder_roty', 'jLeftC7Shoulder_rotz', 'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftShoulder_rotz', 'jLeftElbow_rotx', 'jLeftElbow_roty', 'jLeftElbow_rotz', 'jLeftWrist_rotx', 'jLeftWrist_roty', 'jLeftWrist_rotz', 'jRightHip_rotx', 'jRightHip_roty', 'jRightHip_rotz', 'jRightKnee_rotx', 'jRightKnee_roty', 'jRightKnee_rotz', 'jRightAnkle_rotx', 'jRightAnkle_roty', 'jRightAnkle_rotz', 'jRightBallFoot_rotx', 'jRightBallFoot_roty', 'jRightBallFoot_rotz', 'jLeftHip_rotx', 'jLeftHip_roty', 'jLeftHip_rotz', 'jLeftKnee_rotx', 'jLeftKnee_roty', 'jLeftKnee_rotz', 'jLeftAnkle_rotx', 'jLeftAnkle_roty', 'jLeftAnkle_rotz', 'jLeftBallFoot_rotx', 'jLeftBallFoot_roty', 'jLeftBallFoot_rotz', 'jLeftBallFoot_rotz', 'jLeftBallFoot_roty', 'jLeftBallFoot_rotx', 'jLeftAnkle_rotz']

JOINT_SENSOR_NAMES = ['jLeftBallFoot_rotz'] # TODO -> can be replaced by JOINT_SENSOR_NAMES = JOINT_NAMES when robot URDF is complete

PRESSURE_SENSOR_SITE_NAMES = ["foot_LLB"] # TODO

PRESSURE_SENSOR_NAMES = ["pressure_LLB"] # TODO

class Simulator:
  def __init__(self, xml_path, timestep=0.001, randomization_factor=0):
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    
  def initialize(self):
    try:
      del self.model
      del self.data
      del self.renderer
    except: pass
    #load model from XML
    self.model = mujoco.MjModel.from_xml_path(self.xml_path)
    self.renderer = mujoco.Renderer(self.model, 720, 1080)
    self.model.opt.timestep = self.timestep
    
    # Visualization Options:
    self.scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(self.scene_option)
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    self.model.vis.scale.contactwidth = 0.1
    self.model.vis.scale.contactheight = 0.03
    self.model.vis.scale.forcewidth = 0.05
    self.model.vis.map.force = 0.3
    
    #initialize instance parameters
    self.next_force_start_time = 0
    self.next_force_direction = [0,0]
    self.next_force_magnitude = 0
    self.next_force_duration = 0
    self.next_force_body = 0
    
    # RANDOMIZATION
    # floor friction (0.5 to 1.0)
    self.model.geom('floor').friction = [coef * (1.0*(1.0-self.randomization_factor) + random.uniform(FLOOR_FRICTION_MIN_MULTIPLIER, FLOOR_FRICTION_MAX_MULTIPLIER)*self.randomization_factor) for coef in self.model.geom('floor').friction]    
    #delays in actions and observations (10ms to 50ms)
    self.action_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.observation_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    #round delays to be multiples of the timestep
    self.observation_delay = round(self.observation_delay / self.timestep) * self.timestep
    self.action_delay = round(self.action_delay / self.timestep) * self.timestep
    #make buffers for observations and actions
    self.observation_buffer = [None] * (int)(self.observation_delay/self.timestep)
    self.action_buffer = [None] * (int)(self.action_delay/self.timestep)
    # vary the mass of all limbs randomly
    for i in range(self.model.nbody-1): self.model.body(i+1).mass[0] += random.uniform(-MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor, MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor)
    # attach a random external mass (up to 0.1 kg) to a randomly chosen limb
    self.model.body(random.randint(0, self.model.nbody - 1)).mass[0] += random.uniform(0, MAX_EXTERNAL_MASS_ADDED*self.randomization_factor)
    # randomize IMU X/Y/Z/Quat
    for i in range(len(self.model.site("IMU").pos)):
      self.model.site("IMU").pos[i] += random.uniform(-IMU_POS_OFFSET_MAX*self.randomization_factor, IMU_POS_OFFSET_MAX*self.randomization_factor)
    random_quat = np.quaternion(*self.model.site("IMU").quat) * quaternion.from_euler_angles([random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor), random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor), random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor)])
    self.model.site("IMU").quat = quaternion.as_float_array(random_quat)
    #add sensor noise
    self.model.sensor("gyro").noise = random.uniform(0, (GYRO_NOISE_STDDEV/180.0*np.pi)*self.randomization_factor)
    self.model.sensor("accelerometer").noise = random.uniform(0, ACCELEROMETER_NOISE_STDDEV*self.randomization_factor)
    self.model.sensor("velocimeter").noise = random.uniform(0, VELOCIMETER_NOISE_STDDEV*self.randomization_factor)
    for pressure_sensor in PRESSURE_SENSOR_NAMES:
      self.model.sensor(pressure_sensor).noise = random.uniform(0, PRESSURE_SENSOR_NOISE_STDDEV*self.randomization_factor)
    for joint in JOINT_SENSOR_NAMES:
      self.model.sensor(joint).noise = random.uniform(0, (JOINT_ANGLE_SENSOR_OFFSET_MAX/180.0*np.pi)*self.randomization_factor)
    # randomize foot sensor X/Y positions
    for pressure_site in PRESSURE_SENSOR_SITE_NAMES:
      self.model.site(pressure_site).pos[0] += random.uniform(-PRESSURE_SENSOR_POS_OFFSET_MAX*self.randomization_factor, PRESSURE_SENSOR_POS_OFFSET_MAX*self.randomization_factor)
      self.model.site(pressure_site).pos[1] += random.uniform(-PRESSURE_SENSOR_POS_OFFSET_MAX*self.randomization_factor, PRESSURE_SENSOR_POS_OFFSET_MAX*self.randomization_factor)
    # randomize joint properties  
    for joint in JOINT_NAMES:
      self.model.joint(joint).damping[0] += random.uniform(-JOINT_DAMPING_MAX_CHANGE, JOINT_DAMPING_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).armature[0] += random.uniform(0, JOINT_ARMATURE_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).frictionloss[0] += random.uniform(0, JOINT_FRICTION_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).stiffness[0] += random.uniform(0, JOINT_STIFFNESS_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).margin[0] += random.uniform(0, JOINT_MARGIN_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).range[0] += random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).range[1] += random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)*self.randomization_factor
    
    # create data from model
    self.data = mujoco.MjData(self.model)
    mujoco.mj_kinematics(self.model, self.data)
    
    # randomize joint initial states
    for joint in JOINT_NAMES:
      for i in range(len(self.data.joint(joint).qpos)):
        self.data.joint(joint).qpos[i] += random.uniform(-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi, JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi)*self.randomization_factor
    
    
  def getState(self): #TODO

    # joint positions     20          Joint positions in radians
    # linear acceleration 3           Linear acceleration from IMU
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU
    # foot pressure       8           Pressure values from foot sensors
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    # agent velocity      2           X and Y velocity of robot torso
    
    state = []
    # print(self.data.sensor('accelerometer').data.copy())
    # print(self.data.sensor('velocimeter').data.copy())
    # print(self.data.sensor('gyro').data.copy())
    # print(self.data.sensor('pressure_LLB').data.copy())
    # print(self.data.sensor('jLeftBallFoot_rotz').data.copy())
    
    # cycle state through observation buffer
    self.observation_buffer.append(state)
    observed_state = self.observation_buffer.pop(0)
    return observed_state
    
  def step(self, action):
    # cycle action through action buffer
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    if action_to_take is not None:
        self.data.ctrl = action_to_take
        
    # apply forces to the robot to destabilise it
    if self.data.time > self.next_force_start_time + self.next_force_duration:
      self.next_force_start_time = self.data.time + random.uniform(1, 3)
      self.next_force_duration = random.uniform(MIN_EXTERNAL_FORCE_DURATION*self.randomization_factor, MAX_EXTERNAL_FORCE_DURATION*self.randomization_factor)
      self.next_force_magnitude = random.uniform(MIN_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor, MAX_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor)
      self.data.xfrc_applied[self.next_force_body][0] = 0
      self.data.xfrc_applied[self.next_force_body][1] = 0
      self.next_force_body = random.randint(1, len(self.data.xfrc_applied) - 1)
      self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      while np.linalg.norm(self.next_force_direction) == 0: self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      self.next_force_direction = self.next_force_direction / np.linalg.norm(self.next_force_direction)
    if self.data.time > self.next_force_start_time and self.data.time < self.next_force_start_time + self.next_force_duration:
      self.data.xfrc_applied[self.next_force_body][0] = self.next_force_direction[0] * self.next_force_magnitude
      self.data.xfrc_applied[self.next_force_body][1] = self.next_force_direction[1] * self.next_force_magnitude
    
    # step simulation
    mujoco.mj_step(self.model, self.data)
    
  def render(self):
    self.renderer.update_scene(self.data, camera="track", scene_option=self.scene_option)
    cv2.imshow("Sim View", self.renderer.render())
    cv2.waitKey(1)

    
if __name__ == "__main__":
    sim = Simulator("assets/flat.xml", timestep=0.005, randomization_factor=1)
    while True:
      sim.initialize()
      while sim.data.time < 2:
        sim.getState()
        sim.step([0]*4)
        sim.render()