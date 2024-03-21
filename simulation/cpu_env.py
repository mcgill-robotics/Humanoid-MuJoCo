import jax
from jax import numpy as jp
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
import cv2
import random
from .simulation_parameters import *
from simulation.reward_functions import *
from simulation.simulation_parameters import physics_steps_per_control_step, timestep
from jax.scipy.spatial.transform import Rotation
import gc
import os
from simulation import SIM_XML_PATH, reward_functions

# STATE INFO FROM https://arxiv.org/pdf/2304.13653.pdf

# STATE
    # joint positions     5 · 20          Joint positions in radians (stacked last 5 timesteps)
    # angular velocity    5 · 3           Angular velocity (roll, pitch, yaw) from IMU (stacked)
    # agent velocity      5 · 2           X and Y velocity of robot torso (stacked)
    # linear acceleration 5 · 3           Linear acceleration from IMU (stacked)
    # gravity             5 · 3           Gravity direction, derived from angular velocity using Madgwick filter (stacked)
    # foot pressure       5 · 8           Pressure values from foot sensors (stacked)
    # previous action     5 · 20          Action filter state (stacked)
    
inverseRotateVectors = lambda q, v : Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv().apply(v)

class CPUEnv(gym.Env):
  metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
  def __init__(self, reward_fn, xml_path=SIM_XML_PATH, randomization_factor=0, verbose=False):
    self.platform = "CPU"
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    if type(reward_fn) == str: reward_fn = getattr(reward_functions, reward_fn)
    self.reward_fn = reward_fn
    self.physics_steps_per_control_step = physics_steps_per_control_step
    self.verbose = verbose
    self.num_envs = 1
    self.rng_key = jax.random.PRNGKey(0)

    self.action_space = spaces.Box(-1, 1, shape=(len(JOINT_NAMES),), dtype=np.float32)
    observation_size = len(JOINT_NAMES) + 3 + 2 + 3 + 3 + 8
    self.observation_space = spaces.Box(-10, 10, shape=(observation_size,), dtype=np.float32)
    
  def reset(self, seed=None, options=None):
    try:
      del self.renderer
    except: pass
    
    super().reset(seed=seed)
    if seed is not None: self.rng_key = jax.random.PRNGKey(seed)
    else: self.rng_key = jax.random.PRNGKey(random.randint(0, 100))
    
    if self.verbose: print("Creating new simulation...       ", end='')
    
    #load model from XML
    self.model = mujoco.MjModel.from_xml_path(self.xml_path)
    if os.environ.get('RENDER_SIM', "True") == "True": self.renderer = mujoco.Renderer(self.model, 720, 1080)
    self.model.opt.timestep = self.timestep
    self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self.model.opt.iterations = 15
    self.model.opt.ls_iterations = 15
   
    # Visualization Options:
    self.scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(self.scene_option)
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    self.model.vis.scale.contactwidth = 0.1
    self.model.vis.scale.contactheight = 0.03
    self.model.vis.scale.forcewidth = 0.05
    self.model.vis.map.force = 0.3
    
    #initialize instance parameters
    self.next_force_start_time = 0
    self.next_force_direction = jp.zeros((2))
    self.next_force_magnitude = 0
    self.next_force_duration = 0
    self.next_force_body = 0
    self.previous_torso_local_velocity = jp.zeros((3))
    # save gravity vector
    self.gravity_vector = jp.array([0,0,-1])
    # save torso body index
    self.torso_idx = self.model.body(TORSO_BODY_NAME).id
    # save joint addresses
    self.joint_qpos_idx = []
    self.joint_torque_idx = []
    for joint in JOINT_NAMES:
      self.joint_torque_idx.append(self.model.jnt_dofadr[self.model.joint(joint).id])
      self.joint_qpos_idx.append(self.model.jnt_qposadr[self.model.joint(joint).id])
    self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
    self.joint_torque_idx = jp.array(self.joint_torque_idx)
    # get pressure sensor geometries
    self.pressure_sensor_ids = [self.model.geom(pressure_sensor_geom).id for pressure_sensor_geom in PRESSURE_GEOM_NAMES]
    self.non_robot_geom_ids = [self.model.geom(geom).id for geom in NON_ROBOT_GEOMS]
    
    # RANDOMIZATION
    # floor friction (0.5 to 1.0)
    self.model.geom('floor').friction = [coef * (1.0*(1.0-self.randomization_factor) + random.uniform(FLOOR_FRICTION_MIN_MULTIPLIER, FLOOR_FRICTION_MAX_MULTIPLIER)*self.randomization_factor) for coef in self.model.geom('floor').friction]    
    # vary the mass of all limbs randomly
    for i in range(self.model.nbody-1): self.model.body(i+1).mass[0] = max(0.01, self.model.body(i+1).mass[0] + random.uniform(-MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor, MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor))
    # attach a random external mass (up to 0.1 kg) to a randomly chosen limb
    self.model.body(random.randint(1, self.model.nbody - 1)).mass[0] += random.uniform(0, MAX_EXTERNAL_MASS_ADDED*self.randomization_factor)
    # randomize joint properties  
    for joint in JOINT_NAMES:
      self.model.joint(joint).armature[0] += random.uniform(0, JOINT_ARMATURE_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).stiffness[0] += random.uniform(0, JOINT_STIFFNESS_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).margin[0] += random.uniform(0, JOINT_MARGIN_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).range[0] += random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)*self.randomization_factor
      self.model.joint(joint).range[1] += random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)*self.randomization_factor
    for joint in JOINT_ACTUATOR_NAMES:
      self.model.actuator(joint).forcerange[0] += random.uniform(-JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE)*self.randomization_factor
      self.model.actuator(joint).forcerange[1] += random.uniform(-JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE)*self.randomization_factor
      
    # create data from model
    self.cpu_model = None
    self.data = mujoco.MjData(self.model)
    mujoco.mj_kinematics(self.model, self.data)
    # randomize joint initial states (CPU)
    joint_ctrl_range = JOINT_INITIAL_CTRL_OFFSET_MIN + self.randomization_factor * (JOINT_INITIAL_CTRL_OFFSET_MAX - JOINT_INITIAL_CTRL_OFFSET_MIN)
    for i in range(len(self.data.ctrl)):
      random_val = random.uniform(-joint_ctrl_range, joint_ctrl_range)
      self.data.ctrl[i] = random_val

    #delays in actions and observations (10ms to 50ms)
    #round delays to be multiples of the timestep
    actual_timestep = self.timestep * self.physics_steps_per_control_step
    self.action_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.action_delay = round(self.action_delay / actual_timestep) * actual_timestep
    self.joint_angles_observation_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.joint_angles_observation_delay = round(self.joint_angles_observation_delay / actual_timestep) * actual_timestep
    self.local_ang_vel_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.local_ang_vel_delay = round(self.local_ang_vel_delay / actual_timestep) * actual_timestep
    self.torso_global_velocity_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.torso_global_velocity_delay = round(self.torso_global_velocity_delay / actual_timestep) * actual_timestep
    self.torso_local_accel_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.torso_local_accel_delay = round(self.torso_local_accel_delay / actual_timestep) * actual_timestep
    self.local_gravity_vector_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.local_gravity_vector_delay = round(self.local_gravity_vector_delay / actual_timestep) * actual_timestep
    self.pressure_values_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.pressure_values_delay = round(self.pressure_values_delay / actual_timestep) * actual_timestep
     
    #make buffers for observations and actions   
    self.action_buffer = [self.data.ctrl] * (int)(self.action_delay/actual_timestep)
    self.joint_angles_buffer = []
    self.local_ang_vel_buffer = []
    self.torso_global_velocity_buffer = []
    self.torso_local_accel_buffer = []
    self.local_gravity_vector_buffer = []
    self.pressure_values_buffer = []
    
    for i in range((int)(self.joint_angles_observation_delay/actual_timestep)):
      self.joint_angles_buffer.append(jp.array([0]*len(JOINT_NAMES)))
    for i in range((int)(self.local_ang_vel_delay/actual_timestep)):
      self.local_ang_vel_buffer.append(jp.array([0]*3))
    for i in range((int)(self.torso_global_velocity_delay/actual_timestep)):
      self.torso_global_velocity_buffer.append(jp.array([0]*2))
    for i in range((int)(self.torso_local_accel_delay/actual_timestep)):
      self.torso_local_accel_buffer.append(jp.array([0]*3))
    for i in range((int)(self.local_gravity_vector_delay/actual_timestep)):
      self.local_gravity_vector_buffer.append(self.gravity_vector)
    for i in range((int)(self.pressure_values_delay/actual_timestep)):
      self.pressure_values_buffer.append(jp.array([0]*len(PRESSURE_GEOM_NAMES)))
    
    # initialize environment properties
    self.lastAction = self.data.ctrl
    self.action_change = jp.zeros(self.data.ctrl.shape)
    
    # clean up any unreferenced variables
    gc.collect()
    
    if self.verbose: print("Done")
    
    return self._get_obs(), {}
      
  def _get_obs(self):
    if self.verbose: print("Collecting observations...       ", end='')
    
    torso_quat = self.data.xquat[self.torso_idx]
    torso_global_vel = self.data.cvel[self.torso_idx]
    
    # joint positions     20          Joint positions in radians
    joint_angles = self.data.qpos[self.joint_qpos_idx] + (self.randomization_factor * (JOINT_ANGLE_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=[len(self.joint_qpos_idx)]))
    # normalize
    joint_angles = joint_angles / (jp.pi / 2)

    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU (in torso reference frame)
    torso_global_ang_vel = torso_global_vel[0:3]
    local_ang_vel = inverseRotateVectors(torso_quat, torso_global_ang_vel) + (self.randomization_factor * (GYRO_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(3,)))
    
    # agent velocity      2           X and Y velocity of robot torso (global, NWU)
    torso_global_velocity = torso_global_vel[3:] + (self.randomization_factor * VELOCIMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(3,)))
   
    # linear acceleration 3           Linear acceleration from IMU (local to torso)
    torso_local_velocity = inverseRotateVectors(torso_quat, torso_global_vel[3:])
    torso_local_accel = ((torso_local_velocity - self.previous_torso_local_velocity)/self.timestep) + (self.randomization_factor * ACCELEROMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(3,)))
    self.previous_torso_local_velocity = torso_local_velocity
    
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    noisy_torso_quat = torso_quat + (self.randomization_factor * (IMU_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(4,)))
    local_gravity_vector = inverseRotateVectors(noisy_torso_quat, self.gravity_vector)
    
    # foot pressure       8           Pressure values from foot sensors (N)
    pressure_values = np.zeros((8))
    for i in range(len(self.pressure_sensor_ids)):
      for ci in range(len(self.data.contact.geom1)):
        if self.data.contact.geom1[ci] == self.pressure_sensor_ids[i]:
          pressure_values[i] += abs(self.data.efc_force[self.data.contact.efc_address[ci]])
        if self.data.contact.geom2[ci] == self.pressure_sensor_ids[i]:
          pressure_values[i] += abs(self.data.efc_force[self.data.contact.efc_address[ci]])
    #convert pressure to binary values
    pressure_values = np.where(pressure_values > 0.5, 1.0, 0.0) # above 0.5N is considered a contact
    
    # cycle observations through observation buffers
    self.joint_angles_buffer.append(joint_angles)
    self.local_ang_vel_buffer.append(local_ang_vel)
    self.torso_global_velocity_buffer.append(torso_global_velocity[0:2])
    self.torso_local_accel_buffer.append(torso_local_accel)
    self.local_gravity_vector_buffer.append(local_gravity_vector)
    self.pressure_values_buffer.append(pressure_values)
    
    joint_angles = self.joint_angles_buffer.pop(0)
    local_ang_vel = self.local_ang_vel_buffer.pop(0)
    torso_global_velocity = self.torso_global_velocity_buffer.pop(0)
    torso_local_accel = self.torso_local_accel_buffer.pop(0)
    local_gravity_vector = self.local_gravity_vector_buffer.pop(0)
    pressure_values = self.pressure_values_buffer.pop(0)
    
    delayed_observations = jp.concatenate((joint_angles, local_ang_vel, torso_global_velocity, torso_local_accel, local_gravity_vector, pressure_values))
  
    if self.verbose: print("Done")
    
    return np.array(delayed_observations, dtype=np.float32)
    
  def _get_reward(self):
    if self.verbose: print("Computing reward...              ", end='')
    
    torso_global_velocity = self.data.cvel[self.torso_idx][3:]
    torso_z_pos = self.data.xpos[self.torso_idx, 2]
    torso_quat = self.data.xquat[self.torso_idx]
    joint_torques = self.data.qfrc_constraint[self.joint_torque_idx] + self.data.qfrc_smooth[self.joint_torque_idx]
    self_collision = False
    for i in range(len(self.data.contact.geom1)):
      if self.data.contact.geom1[i] not in self.non_robot_geom_ids and self.data.contact.geom2[i] not in self.non_robot_geom_ids:
        self_collision = True
    
    reward, isTerminal = self.reward_fn(torso_global_velocity, torso_z_pos, torso_quat, joint_torques, self.action_change, self_collision)
    
    if self.verbose: print("Done")

    return float(reward), bool(isTerminal)
    
  def step(self, action=None):
    if self.verbose: print("Stepping simulation...           ", end='')
    # cycle action through action buffer
    if action is None:
      action = self.data.ctrl
    # TODO: actions should be -1 to 1, we need to map each entry to the corresponding joint limits in radians
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    action_to_take = jp.clip(jp.array(action_to_take), -1.0, 1.0)
    self.data.ctrl = action_to_take
    self.action_change = action_to_take - self.lastAction[0]
    self.lastAction = jp.expand_dims(action_to_take, axis=0)
        
    # apply forces to the robot to destabilise it
    if self.data.time >= self.next_force_start_time + self.next_force_duration:
      self.next_force_start_time = self.data.time + random.uniform(MIN_EXTERNAL_FORCE_INTERVAL, MAX_EXTERNAL_FORCE_INTERVAL)
      self.next_force_duration = random.uniform(MIN_EXTERNAL_FORCE_DURATION, MAX_EXTERNAL_FORCE_DURATION)
      self.next_force_magnitude = random.uniform(MIN_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor, MAX_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor)
      self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      self.data.xfrc_applied[self.next_force_body][0] = 0
      self.data.xfrc_applied[self.next_force_body][1] = 0
      self.next_force_body = random.randint(1, len(self.data.xfrc_applied) - 1)
      
      while np.linalg.norm(self.next_force_direction) == 0: self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      self.next_force_direction = self.next_force_direction / np.linalg.norm(self.next_force_direction)
    if self.data.time > self.next_force_start_time and self.data.time < self.next_force_start_time + self.next_force_duration:
      self.data.xfrc_applied[self.next_force_body][0] = self.next_force_direction[0] * self.next_force_magnitude
      self.data.xfrc_applied[self.next_force_body][1] = self.next_force_direction[1] * self.next_force_magnitude
    
    # step simulation
    for _ in range(self.physics_steps_per_control_step - 1):
      mujoco.mj_step(self.model, self.data)
      
    torso_quat = self.data.xquat[self.torso_idx]
    torso_global_vel = self.data.cvel[self.torso_idx]
    self.previous_torso_local_velocity = inverseRotateVectors(torso_quat, torso_global_vel[3:])
    
    mujoco.mj_step(self.model, self.data)
      
    if self.verbose: print("Done")
    
    reward, terminated = self._get_reward()
    
    truncated = False
    if self.data.time >= max_simulation_time:
      truncated = True
    
    if terminated or truncated: info = {"is_success": truncated}
    else: info = {}
    
    return self._get_obs(), reward, terminated, truncated, info
    
  def render(self, mode="rgb_array"):
    if not os.environ.get('RENDER_SIM', "True") == "True": return None
    self.renderer.update_scene(self.data, camera="track", scene_option=self.scene_option)
    frame = self.renderer.render()
    if mode == "human":
      cv2.imshow("Sim View", frame)
      cv2.waitKey(1)
    else:
      return frame
    
if __name__ == "__main__":
  sim = CPUEnv(xml_path=SIM_XML_PATH, reward_fn=standingRewardFn, randomization_factor=1)
  obs = sim.reset()
  
  while True:
    # action = np.random.uniform(-1, 1, 16)
    action = None
    obs, reward, isTerminal, _, _ = sim.step(action)
    print(reward)
    sim.render("human")