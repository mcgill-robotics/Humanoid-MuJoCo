import mujoco
import cv2
import numpy as np
import random
import quaternion
from simulation_parameters import *

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
    self.IMU_offset_quat = np.quaternion(*self.model.site("IMU").quat) * quaternion.from_euler_angles([random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor), random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor), random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*np.pi)*self.randomization_factor)])
    self.model.site("IMU").quat = quaternion.as_float_array(self.IMU_offset_quat)
    #add sensor noise
    self.model.sensor("gyro").noise = random.uniform(0, (GYRO_NOISE_STDDEV/180.0*np.pi)*self.randomization_factor)
    self.model.sensor("accelerometer").noise = random.uniform(0, ACCELEROMETER_NOISE_STDDEV*self.randomization_factor)
    self.model.sensor("velocimeter").noise = random.uniform(0, VELOCIMETER_NOISE_STDDEV*self.randomization_factor)
    for pressure_sensor in PRESSURE_SENSOR_NAMES:
      self.model.sensor(pressure_sensor).noise = random.uniform(0, PRESSURE_SENSOR_NOISE_STDDEV*self.randomization_factor)
    for joint in JOINT_SENSOR_NAMES:
      self.model.sensor(joint).noise = random.uniform(0, (JOINT_ANGLE_NOISE_STDDEV/180.0*np.pi)*self.randomization_factor)
    # randomize foot sensor X/Y positions
    for pressure_site in PRESSURE_SENSOR_NAMES:
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
    for joint in JOINT_ACTUATOR_NAMES:
      self.model.actuator(joint).forcerange[0] += random.uniform(-JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE)*self.randomization_factor
      self.model.actuator(joint).forcerange[1] += random.uniform(-JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE)*self.randomization_factor
      kp = max(0, JOINT_PID_P_GAIN + random.uniform(-JOINT_PID_GAIN_MAX_CHANGE, JOINT_PID_GAIN_MAX_CHANGE)*self.randomization_factor)
      kv = max(0, JOINT_PID_V_GAIN + random.uniform(-JOINT_PID_GAIN_MAX_CHANGE, JOINT_PID_GAIN_MAX_CHANGE)*self.randomization_factor)
      self.model.actuator(joint).gainprm[0] = kp
      self.model.actuator(joint).biasprm[1] = -kp
      self.model.actuator(joint).biasprm[2] = -kv

    # create data from model
    self.data = mujoco.MjData(self.model)
    mujoco.mj_kinematics(self.model, self.data)
    
    # randomize joint initial states
    for joint in JOINT_NAMES:
      for i in range(len(self.data.joint(joint).qpos)):
        self.data.joint(joint).qpos[i] += random.uniform(-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi, JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi)*self.randomization_factor
    
  def computeReward(self):
    reward = 0
    # Velocity The magnitude of the player’s forward velocity. - 0.1
    delta_pos = self.data.sensor("IMU_vel").data.copy() # LOCAL FRAME
    reward += X_VELOCITY_REWARD_WEIGHT * delta_pos[0]
    reward += Y_VELOCITY_REWARD_WEIGHT * delta_pos[1]
    reward += Z_VELOCITY_REWARD_WEIGHT * delta_pos[2]
    
    # Termination A penalty, equal to −1 if the player is on the ground - 0.5
    isTouchingGround = self.data.body('humanoid').xpos[2] < 0
    if isTouchingGround:
      reward += -1 * GROUNDED_PENALTY_WEIGHT
      
    # Upright 0 if the robot is upside down or if the tilt angle is greater
        # than 0.4 radians. Increases linearly, and is equal to +1 if the
        # tilt angle is less than 0.2 radians. - 0.02
    IMU_quat = np.quaternion(*self.data.sensor("IMU_quat").data.copy()) * self.IMU_offset_quat.inverse()
    tilt_angle = abs(2 * np.arccos(IMU_quat.w))
    if tilt_angle < MIN_TILT_FOR_REWARD:
      reward += UPRIGHT_REWARD_WEIGHT
    elif tilt_angle < MAX_TILT_FOR_REWARD:
      tilt_reward = (MAX_TILT_FOR_REWARD - tilt_angle) / (MAX_TILT_FOR_REWARD-MIN_TILT_FOR_REWARD)
      reward += tilt_reward * UPRIGHT_REWARD_WEIGHT
      
    # Joint torque A penalty, equal to the magnitude of the torque measured at
        # the player’s knees. This discourages the player from learning
        # gaits which cause high forces on the knees, for example
        # during ground impacts, which can damage a physical robot. - 0.01
    total_joint_torque = 0
    for joint in JOINT_ACTUATOR_NAMES:
      joint_torque = np.linalg.norm(np.array(self.data.joint(joint).qfrc_constraint + self.data.joint(joint).qfrc_smooth))
      total_joint_torque = max(total_joint_torque, joint_torque)
    reward += -total_joint_torque * JOINT_TORQUE_PENALTY_WEIGHT
    
    return reward
    
  def getState(self):
    state = []
    # joint positions     20          Joint positions in radians
    for joint in JOINT_SENSOR_NAMES:
      state.append(self.data.sensor(joint).data.copy()[0]) # LOCAL FRAME (PER-JOINT)
    # linear acceleration 3           Linear acceleration from IMU
    state.extend(self.data.sensor('accelerometer').data.copy()) # LOCAL FRAME (IMU)
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU
    state.extend(self.data.sensor('gyro').data.copy()) # LOCAL FRAME (IMU)
    # foot pressure       8           Pressure values from foot sensors
    for pressure_sensor in PRESSURE_SENSOR_NAMES:
      state.append(self.data.sensor(pressure_sensor).data.copy()[0])
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    global_gravity_vector = self.model.opt.gravity
    world_IMU_quat_world = self.data.sensor("IMU_quat").data.copy()
    local_gravity_vector_IMU = quaternion.rotate_vectors(np.quaternion(*world_IMU_quat_world).inverse(), global_gravity_vector)
    state.extend(local_gravity_vector_IMU) # LOCAL FRAME (IMU)
    # agent velocity      2           X and Y velocity of robot torso
    state.extend(self.data.sensor('velocimeter').data.copy()) # LOCAL FRAME (IMU)

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
      self.next_force_start_time = self.data.time + random.uniform(MIN_EXTERNAL_FORCE_INTERVAL, MAX_EXTERNAL_FORCE_INTERVAL)
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
    return self.computeReward()
    
  def render(self):
    self.renderer.update_scene(self.data, camera="track", scene_option=self.scene_option)
    cv2.imshow("Sim View", self.renderer.render())
    cv2.waitKey(1)

    
if __name__ == "__main__":
    sim = Simulator("assets/flat.xml", timestep=0.005, randomization_factor=1)
    while True:
      print("reset")
      sim.initialize()
      action = [0]*4
      while sim.data.time < 2:
        sim.getState()
        reward = sim.step(action)
        # print(reward)
        sim.render()