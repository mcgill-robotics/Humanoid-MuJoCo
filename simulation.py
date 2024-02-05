import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import cv2
import random
import quaternion
from simulation_parameters import *
import gc

class Simulation:
  def __init__(self, xml_path, timestep=0.001, randomization_factor=0, run_on_gpu=False):
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.run_on_gpu = run_on_gpu if jax.default_backend() == 'gpu' else False
    if run_on_gpu != self.run_on_gpu:
      print("WARN: failed to find GPU device. Running simulation with CPU.")
    
    # TODO: remove this line when done testing GPU functionality
    self.run_on_gpu = run_on_gpu
    
  def reset(self):
    try:
      del self.model
      del self.data
      del self.renderer
      del self.cpu_model
    except: pass
    gc.collect()
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
    # TODO: ensure changes are reflected when running on GPU
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
    if self.run_on_gpu:
      self.cpu_model = self.model
      self.cpu_data = mujoco.MjData(self.cpu_model)
      mujoco.mj_kinematics(self.cpu_model, self.cpu_data)
      self.model = mjx.put_model(self.cpu_model)
      self.jax_step = jax.jit(mjx.step)
      self.data = mjx.put_data(self.cpu_model, self.cpu_data)
    else:
      self.cpu_model = None
      self.data = mujoco.MjData(self.model)
      mujoco.mj_kinematics(self.model, self.data)

    # TODO: doesn't reflect when running on GPU
    # randomize joint initial states
    for i in range(len(self.data.qpos)):
      if self.run_on_gpu:
        self.data.qpos.at[i].set(self.data.qpos[i] + random.uniform(-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi, JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi)*self.randomization_factor)
      else:
        self.data.qpos[i] += random.uniform(-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi, JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi)*self.randomization_factor

  def computeReward(self):
    if self.run_on_gpu:
      self.cpu_data = mjx.get_data(self.cpu_model, self.data)
    else:
      self.cpu_data = self.data
    
    reward = 0
    # Velocity The magnitude of the player’s forward velocity. - 0.1
    delta_pos = self.cpu_data.sensor("IMU_vel").data.copy() # LOCAL FRAME
    if ABS_X_VELOCITY: delta_pos[0] = abs(delta_pos[0])
    if ABS_X_VELOCITY: delta_pos[1] = abs(delta_pos[1])
    if ABS_Z_VELOCITY: delta_pos[2] = abs(delta_pos[2])
    reward += X_VELOCITY_REWARD_WEIGHT * delta_pos[0]
    reward += Y_VELOCITY_REWARD_WEIGHT * delta_pos[1]
    reward += Z_VELOCITY_REWARD_WEIGHT * delta_pos[2]
    
    # Termination A penalty, equal to −1 if the player is on the ground - 0.5
    isTouchingGround = self.cpu_data.body('humanoid').xpos[2] < 0
    if isTouchingGround:
      reward += -1 * GROUNDED_PENALTY_WEIGHT
      
    # Upright 0 if the robot is upside down or if the tilt angle is greater
        # than 0.4 radians. Increases linearly, and is equal to +1 if the
        # tilt angle is less than 0.2 radians. - 0.02
    IMU_quat = np.quaternion(*self.cpu_data.sensor("IMU_quat").data.copy()) * self.IMU_offset_quat.inverse()
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
      joint_torque = np.linalg.norm(np.array(self.cpu_data.joint(joint).qfrc_constraint + self.cpu_data.joint(joint).qfrc_smooth))
      total_joint_torque = max(total_joint_torque, joint_torque)
    reward += -total_joint_torque * JOINT_TORQUE_PENALTY_WEIGHT
    
    return reward
    
  def getState(self):
    if self.run_on_gpu:
      self.cpu_data = mjx.get_data(self.cpu_model, self.data)
    else:
      self.cpu_data = self.data
    
    state = []

    # joint positions     20          Joint positions in radians
    for joint in JOINT_SENSOR_NAMES:
      state.append(self.cpu_data.sensor(joint).data.copy()[0] + random.gauss(0, JOINT_ANGLE_NOISE_STDDEV/180.0*np.pi)) # LOCAL FRAME (PER-JOINT)
    # linear acceleration 3           Linear acceleration from IMU
    state.extend([val + random.gauss(0, ACCELEROMETER_NOISE_STDDEV) for val in self.cpu_data.sensor('accelerometer').data.copy()]) # LOCAL FRAME (IMU)
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU
    state.extend([val + random.gauss(0, GYRO_NOISE_STDDEV) for val in self.cpu_data.sensor('gyro').data.copy()]) # LOCAL FRAME (IMU)
    # foot pressure       8           Pressure values from foot sensors
    for pressure_sensor in PRESSURE_SENSOR_NAMES:
      state.append(self.cpu_data.sensor(pressure_sensor).data.copy()[0] + random.gauss(0, PRESSURE_SENSOR_NOISE_STDDEV))
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    global_gravity_vector = self.model.opt.gravity
    world_IMU_quat_world = [val + random.gauss(0, IMU_NOISE_STDDEV/180.0*np.pi) for val in self.cpu_data.sensor("IMU_quat").data.copy()]
    local_gravity_vector_IMU = quaternion.rotate_vectors(np.quaternion(*world_IMU_quat_world).inverse(), global_gravity_vector)
    state.extend(local_gravity_vector_IMU) # LOCAL FRAME (IMU)
    # agent velocity      2           X and Y velocity of robot torso
    state.extend([val + random.gauss(0, VELOCIMETER_NOISE_STDDEV) for val in self.cpu_data.sensor('velocimeter').data.copy()]) # LOCAL FRAME (IMU)

    # cycle state through observation buffer
    self.observation_buffer.append(state)
    observed_state = self.observation_buffer.pop(0)
    
    return observed_state
    
  def step(self, action):
    # cycle action through action buffer
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    if action_to_take is not None:
      if self.run_on_gpu:
        for i in range(len(action_to_take)):
          self.data.ctrl.at[i].set(action_to_take[i])
      else:
        self.data.ctrl = action_to_take
        
    # apply forces to the robot to destabilise it
    if self.data.time > self.next_force_start_time + self.next_force_duration:
      self.next_force_start_time = self.data.time + random.uniform(MIN_EXTERNAL_FORCE_INTERVAL, MAX_EXTERNAL_FORCE_INTERVAL)
      self.next_force_duration = random.uniform(MIN_EXTERNAL_FORCE_DURATION*self.randomization_factor, MAX_EXTERNAL_FORCE_DURATION*self.randomization_factor)
      self.next_force_magnitude = random.uniform(MIN_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor, MAX_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor)
      self.next_force_body = random.randint(1, len(self.data.xfrc_applied) - 1)
      self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      if self.run_on_gpu:
        self.data.xfrc_applied.at[self.next_force_body,0].set(0)
        self.data.xfrc_applied.at[self.next_force_body,1].set(0)
      else:
        self.data.xfrc_applied[self.next_force_body][0] = 0
        self.data.xfrc_applied[self.next_force_body][1] = 0
      
      while np.linalg.norm(self.next_force_direction) == 0: self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      self.next_force_direction = self.next_force_direction / np.linalg.norm(self.next_force_direction)
    if self.data.time > self.next_force_start_time and self.data.time < self.next_force_start_time + self.next_force_duration:
      if self.run_on_gpu:
        self.data.xfrc_applied.at[self.next_force_body,0].set(self.next_force_direction[0] * self.next_force_magnitude)
        self.data.xfrc_applied.at[self.next_force_body,1].set(self.next_force_direction[1] * self.next_force_magnitude)
      else:
        self.data.xfrc_applied[self.next_force_body][0] = self.next_force_direction[0] * self.next_force_magnitude
        self.data.xfrc_applied[self.next_force_body][1] = self.next_force_direction[1] * self.next_force_magnitude
    
    # step simulation
    if self.run_on_gpu:
      self.data = self.jax_step(self.model, self.data)
    else:
      mujoco.mj_step(self.model, self.data)
    return self.computeReward()
    
  def render(self, display=True):
    if self.run_on_gpu:
      self.renderer.update_scene(mjx.get_data(self.cpu_model, self.data), scene_option=self.scene_option)
    else:
      self.renderer.update_scene(self.data, camera="track", scene_option=self.scene_option)
    frame = self.renderer.render()
    if display:
      cv2.imshow("Sim View", frame)
      cv2.waitKey(1)
    return frame

    
if __name__ == "__main__":
    sim = Simulation("assets/world.xml",
                     timestep=0.005,
                     randomization_factor=1,
                     run_on_gpu=True)
    while True:
      sim.reset()
      action = [0]*4
      while sim.data.time < 2:
        print(sim.getState())
        reward = sim.step(action)
        # print(reward)
        sim.render()