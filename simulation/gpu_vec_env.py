import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from .simulation_parameters import *
from reward_functions import *
from simulation.simulation_parameters import physics_steps_per_control_step, timestep
import gc
import random
from .gpu_vec_env_utils import *
from simulation import SIM_XML_PATH

# STATE INFO FROM https://arxiv.org/pdf/2304.13653.pdf

# STATE
    # joint positions     5 · 20          Joint positions in radians (stacked last 5 timesteps)
    # angular velocity    5 · 3           Angular velocity (roll, pitch, yaw) from IMU (stacked)
    # agent velocity      5 · 2           X and Y velocity of robot torso (stacked)
    # linear acceleration 5 · 3           Linear acceleration from IMU (stacked)
    # gravity             5 · 3           Gravity direction, derived from angular velocity using Madgwick filter (stacked)
    # foot pressure       5 · 8           Pressure values from foot sensors (stacked)
    # previous action     5 · 20          Action filter state (stacked)

class GPUVecEnv(VecEnv):
  def __init__(self, num_envs, xml_path, reward_fn, randomization_factor=0, verbose=False):
    if jax.default_backend() != 'gpu':
      print("ERROR: Failed to find GPU device.")
      exit()
    print("INFO: Running MuJoCo on GPU.")
    
    self.platform = "GPU"
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.num_envs = num_envs
    self.reward_fn = jax.jit(jax.vmap(lambda v, z, q, jt, ac, sc : reward_fn(v, z, q, jt, ac, sc)))
    self.physics_steps_per_control_step = physics_steps_per_control_step
    self.rng_key = jax.random.PRNGKey(42)
    self.rng = jax.random.split(self.rng_key, self.num_envs)
    self.verbose = verbose
    self.render_mode = [[None]] * self.num_envs
    
    # define jax step function
    def rollout(m,d):
      for _ in range(self.physics_steps_per_control_step - 1):
        d = mjx.step(m,d)
      return d
    
    self.jax_step_minus_one = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))
    self.jax_single_step = jax.jit(jax.vmap(lambda m,d : mjx.step(m,d), in_axes=(None, 0)))
    
    #define contact force function
    self.getFootForces = jax.jit(jax.vmap(getFootForces, in_axes=(None, 0)))
    
    self.action_space = spaces.Box(-1, 1, shape=(len(JOINT_NAMES),), dtype=np.float32)
    observation_size = len(JOINT_NAMES) + 3 + 2 + 3 + 3 + 8
    self.observation_space = spaces.Box(-1000, 1000, shape=(observation_size,), dtype=np.float32)
    
    super().__init__(self.num_envs, self.observation_space, self.action_space)
  
  def close(self): 
    try: del self.model
    except: pass
    try: del self.cpu_model
    except: pass
    try: del self.base_mjx_data
    except: pass
    try: del self.data_batch
    except: pass
    # clean up any unreferenced variables
    gc.collect()
    
  def render(self, mode=None): pass
  
  def env_is_wrapped(self, wrapper_class, indices=None): return [False]*self.num_envs
    
  def step_async(self, actions):
      self.async_result = self.step(actions)

  def step_wait(self):
      return self.async_result

  def get_attr(self, attr_name, indices=None):
    if indices is None:
      return getattr(self, attr_name)
    else:
      return [getattr(self, attr_name)]*len(indices)
      
  def set_attr(self, attr_name, value, indices=None):
      setattr(self, attr_name, value)

  def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
      if method_name == "reset":
        print("Attempted reset on envs [{}]".format(indices))
      elif method_name == "step":
        print("Attempted step on envs [{}]".format(indices))
      else:
        func = getattr(self, method_name)
        if indices is not None:
          return [func(*method_args)]
        else:
          return func(*method_args)

  def seed(self, seed):
    self.rng_key = jax.random.PRNGKey(seed)
    self.rng = jax.random.split(self.rng_key, self.num_envs)
    
  def reset(self, seed=None):
    self.close()
    
    if self.verbose: print("Initializing simulations...      ", end='')
    
    if seed is not None: 
      self.rng_key = jax.random.PRNGKey(seed)
      self.rng = jax.random.split(self.rng_key, self.num_envs)
    
    #load model from XML
    self.model = mujoco.MjModel.from_xml_path(self.xml_path)
    self.model.opt.timestep = self.timestep
    self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self.model.opt.iterations = 15
    self.model.opt.ls_iterations = 15

    #initialize instance parameters
    self.next_force_start_times = jp.zeros((self.num_envs))
    self.next_force_durations = jp.zeros((self.num_envs))
    self.next_force_magnitudes = jp.zeros((self.num_envs))
    self.next_force_bodies = jp.zeros((self.num_envs))
    self.next_force_directions = jp.zeros((self.num_envs, 2))
    self.previous_torso_local_velocity = jp.zeros((self.num_envs, 3))
    # save joint addresses
    self.joint_qpos_idx = []
    self.joint_torque_idx = []
    for joint in JOINT_NAMES:
      self.joint_torque_idx.append(self.model.jnt_dofadr[self.model.joint(joint).id])
      self.joint_qpos_idx.append(self.model.jnt_qposadr[self.model.joint(joint).id])
    self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
    self.joint_torque_idx = jp.array(self.joint_torque_idx)
    # save gravity vector
    self.gravity_vector = jp.array([0,0,-1]) # normalized, so -1 instead of -9.81
    self.gravity_vector_batch = jp.array([self.gravity_vector]*self.num_envs)
    # save torso body index
    self.torso_idx = self.model.body(TORSO_BODY_NAME).id
    # get pressure sensor geom ids
    self.pressure_sensor_ids = [self.model.geom(pressure_sensor_geom).id for pressure_sensor_geom in PRESSURE_GEOM_NAMES]
    self.non_robot_geom_ids = [self.model.geom(geom).id for geom in NON_ROBOT_GEOMS]
  
    self.isSelfColliding = jax.jit(jax.vmap(checkSelfCollision, in_axes=(None, 0)))
    
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
      
    # create MJX model/data from CPU model/data
    self.cpu_model = self.model
    self.model = mjx.put_model(self.cpu_model)
    mj_data = mujoco.MjData(self.cpu_model)
    mujoco.mj_kinematics(self.cpu_model, mj_data)
    self.base_mjx_data = mjx.put_data(self.cpu_model, mj_data)
    
    # randomize joint initial states (GPU)
    joint_ctrl_range = JOINT_INITIAL_CTRL_OFFSET_MIN + self.randomization_factor * (JOINT_INITIAL_CTRL_OFFSET_MAX - JOINT_INITIAL_CTRL_OFFSET_MIN)
    self.data_batch = jax.vmap(lambda rng: self.base_mjx_data.replace(ctrl=jax.random.uniform(rng, self.base_mjx_data.ctrl.shape, minval=-joint_ctrl_range, maxval=joint_ctrl_range)))(self.rng)

    #delays in actions and observations (10ms to 50ms)
    #round delays to be multiples of the timestep
    self.actual_timestep = self.timestep * self.physics_steps_per_control_step
    self.action_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.action_delay = round(self.action_delay / self.actual_timestep) * self.actual_timestep
    self.joint_angles_observation_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.joint_angles_observation_delay = round(self.joint_angles_observation_delay / self.actual_timestep) * self.actual_timestep
    self.local_ang_vel_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.local_ang_vel_delay = round(self.local_ang_vel_delay / self.actual_timestep) * self.actual_timestep
    self.torso_global_velocity_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.torso_global_velocity_delay = round(self.torso_global_velocity_delay / self.actual_timestep) * self.actual_timestep
    self.torso_local_accel_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.torso_local_accel_delay = round(self.torso_local_accel_delay / self.actual_timestep) * self.actual_timestep
    self.local_gravity_vector_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.local_gravity_vector_delay = round(self.local_gravity_vector_delay / self.actual_timestep) * self.actual_timestep
    self.pressure_values_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.pressure_values_delay = round(self.pressure_values_delay / self.actual_timestep) * self.actual_timestep
     
    #make buffers for observations and actions
    self.action_buffer = [self.data_batch.ctrl] * (int)(self.action_delay/self.actual_timestep)
    self.joint_angles_buffer = []
    self.local_ang_vel_buffer = []
    self.torso_global_velocity_buffer = []
    self.torso_local_accel_buffer = []
    self.local_gravity_vector_buffer = []
    self.pressure_values_buffer = []
    
    for i in range((int)(self.joint_angles_observation_delay/self.actual_timestep)):
      self.joint_angles_buffer.append(jp.array([[0]*len(JOINT_NAMES)]*self.num_envs))
    for i in range((int)(self.local_ang_vel_delay/self.actual_timestep)):
      self.local_ang_vel_buffer.append(jp.array([[0]*3]*self.num_envs))
    for i in range((int)(self.torso_global_velocity_delay/self.actual_timestep)):
      self.torso_global_velocity_buffer.append(jp.array([[0]*2]*self.num_envs))
    for i in range((int)(self.torso_local_accel_delay/self.actual_timestep)):
      self.torso_local_accel_buffer.append(jp.array([[0]*3]*self.num_envs))
    for i in range((int)(self.local_gravity_vector_delay/self.actual_timestep)):
      self.local_gravity_vector_buffer.append(jp.array([self.gravity_vector]*self.num_envs))
    for i in range((int)(self.pressure_values_delay/self.actual_timestep)):
      self.pressure_values_buffer.append(jp.array([[0]*len(PRESSURE_GEOM_NAMES)]*self.num_envs))
    
    # initialize environment properties
    self.lastAction = self.data_batch.ctrl
    self.action_change = jp.zeros(self.data_batch.ctrl.shape)

    if self.verbose: print("Done")
    
    return self._get_obs()

  def _get_rewards(self):
    if self.verbose: print("Computing rewards...             ", end='')
    
    torso_global_velocity = self.data_batch.cvel[:, self.torso_idx][:, 3:]
    torso_z_pos = self.data_batch.xpos[:, self.torso_idx, 2]
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    joint_torques = self.data_batch.qfrc_constraint[:, self.joint_torque_idx] + self.data_batch.qfrc_smooth[:, self.joint_torque_idx]
    self_collisions = self.isSelfColliding(self.non_robot_geom_ids, self.data_batch)
        
    rewards, areTerminal = self.reward_fn(torso_global_velocity, torso_z_pos, torso_quat, joint_torques, self.action_change, self_collisions)
    
    if self.verbose: print("Done")

    return np.array(rewards), np.array(areTerminal)
    
  def _get_obs(self):
    if self.verbose: print("Collecting observations...       ", end='')
    
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    torso_global_vel = self.data_batch.cvel[:, self.torso_idx]
    
    # joint positions     20          Joint positions in radians
    joint_angles = self.data_batch.qpos[:, self.joint_qpos_idx] + (self.randomization_factor * (JOINT_ANGLE_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.num_envs, len(self.joint_qpos_idx))))
    #normalize
    joint_angles = joint_angles / (jp.pi / 2)
    
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU (in torso reference frame)
    torso_global_ang_vel = torso_global_vel[:, 0:3]
    local_ang_vel = inverseRotateVectors(torso_quat, torso_global_ang_vel) + (self.randomization_factor * (GYRO_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 3)))
    
    # agent velocity      2           X and Y velocity of robot torso (global, NWU)
    torso_global_velocity = torso_global_vel[:, 3:] + (self.randomization_factor * VELOCIMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 3)))
    
    # linear acceleration 3           Linear acceleration from IMU (local to torso)
    torso_local_velocity = inverseRotateVectors(torso_quat, torso_global_vel[:, 3:])
    torso_local_accel = ((torso_local_velocity - self.previous_torso_local_velocity)/self.actual_timestep) + (self.randomization_factor * ACCELEROMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 3)))
    self.previous_torso_local_velocity = torso_local_velocity
    
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    noisy_torso_quat = torso_quat + (self.randomization_factor * (IMU_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 4)))
    local_gravity_vector = inverseRotateVectors(noisy_torso_quat, self.gravity_vector_batch)
    
    # foot pressure       8           Pressure values from foot sensors (N)
    pressure_values = self.getFootForces(self.pressure_sensor_ids, self.data_batch)
    #normalize
    pressure_values = jp.clip(pressure_values, 0.0, 5.0) / 5.0 # 500 grams ~ 5N

    # cycle observations through observation buffers
    self.joint_angles_buffer.append(joint_angles)
    self.local_ang_vel_buffer.append(local_ang_vel)
    self.torso_global_velocity_buffer.append(torso_global_velocity[:, 0:2])
    self.torso_local_accel_buffer.append(torso_local_accel)
    self.local_gravity_vector_buffer.append(local_gravity_vector)
    self.pressure_values_buffer.append(pressure_values)
    
    joint_angles = self.joint_angles_buffer.pop(0)
    local_ang_vel = self.local_ang_vel_buffer.pop(0)
    torso_global_velocity = self.torso_global_velocity_buffer.pop(0)
    torso_local_accel = self.torso_local_accel_buffer.pop(0)
    local_gravity_vector = self.local_gravity_vector_buffer.pop(0)
    pressure_values = self.pressure_values_buffer.pop(0)
    
    delayed_observations = jp.hstack((joint_angles, local_ang_vel, torso_global_velocity, torso_local_accel, local_gravity_vector, pressure_values))
        
    if self.verbose: print("Done")
    
    return np.array(delayed_observations)
  
  def step(self, action=None):
    if self.verbose: print("Stepping simulations...          ", end='')
    
    # cycle action through action buffer
    if action is None:
      action = self.data_batch.ctrl
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    action_to_take = jp.clip(jp.array(action_to_take), -1.0, 1.0)
    self.data_batch = self.data_batch.replace(ctrl=action_to_take)
    
    self.action_change = action_to_take - self.lastAction
    self.lastAction = action_to_take
    
    # apply forces to the robot to destabilise it
    xfrc_applied = applyExternalForces(self)
    self.data_batch = self.data_batch.replace(xfrc_applied=xfrc_applied)

    # step sims, update data batch
    self.data_batch = self.jax_step_minus_one(self.model, self.data_batch)
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    torso_global_vel = self.data_batch.cvel[:, self.torso_idx]
    self.previous_torso_local_velocity = inverseRotateVectors(torso_quat, torso_global_vel[:, 3:])
    self.data_batch = self.jax_single_step(self.model, self.data_batch)
    
    obs = self._get_obs()
    rewards, terminals = self._get_rewards()
    truncated = np.any(self.data_batch.time >= max_simulation_time)
    done = truncated or np.all(terminals)
    dones = np.full(terminals.shape, done)
    infos = [{}]*self.num_envs    
    
    # automatically reset environment if any instance terminates
    if done:
      for env_idx in range(self.num_envs):
        infos[env_idx]["terminal_observation"] = obs[env_idx]
        infos[env_idx]["TimeLimit.truncated"] = not terminals[env_idx]
      obs = self.reset()
          
    if self.verbose: print("Done")
    
    return obs, rewards, dones, infos
  
if __name__ == "__main__":
  sim_batch = GPUVecEnv(num_envs=256,
                                  xml_path=SIM_XML_PATH,
                                  reward_fn=standingRewardFn,
                                  randomization_factor=1,
                                  verbose=True)

  obs = sim_batch.reset()

  while True:
    actions = np.random.uniform(-1, 1, (sim_batch.num_envs, 16))
    actions = None
    obs, rewards, terminals, _ = sim_batch.step(actions)
    # print(rewards[0])
    if np.isnan(obs).any() or np.isnan(rewards).any() or np.isnan(terminals).any():
        print("ERROR: NaN value in observations/rewards/terminals.")