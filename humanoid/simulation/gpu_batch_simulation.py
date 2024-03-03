import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from .simulation_parameters import *
from .reward_functions import *
import gc
import random
from .gpu_batch_simulation_utils import *
from humanoid import SIM_XML_PATH

# STATE INFO FROM https://arxiv.org/pdf/2304.13653.pdf

# STATE
    # joint positions     5 · 20          Joint positions in radians (stacked last 5 timesteps)
    # linear acceleration 5 · 3           Linear acceleration from IMU (stacked)
    # angular velocity    5 · 3           Angular velocity (roll, pitch, yaw) from IMU (stacked)
    # foot pressure       5 · 8           Pressure values from foot sensors (stacked)
    # gravity             5 · 3           Gravity direction, derived from angular velocity using Madgwick filter (stacked)
    # agent velocity      5 · 2           X and Y velocity of robot torso (stacked)
    # previous action     5 · 20          Action filter state (stacked)

class GPUBatchSimulation:
  def __init__(self, count, xml_path, reward_fn, physics_steps_per_control_step=5, timestep=0.001, randomization_factor=0, verbose=False):
    if jax.default_backend() != 'gpu':
      print("ERROR: Failed to find GPU device.")
      exit()
    print("INFO: Running MuJoCo on GPU.")
    
    self.platform = "GPU"
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.count = count
    self.reward_fn = jax.jit(jax.vmap(lambda v, z, q, jt : reward_fn(v, z, q, jt)))
    self.physics_steps_per_control_step = physics_steps_per_control_step
    self.rng_key = jax.random.PRNGKey(42)
    self.rng = jax.random.split(self.rng_key, self.count)
    self.verbose = verbose
    
    # define jax step function
    def rollout(m,d):
      for _ in range(self.physics_steps_per_control_step):
        d = mjx.step(m,d)
      return d
    self.jax_step = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))
    
    #define contact force function
    self.getFootForces = jax.jit(jax.vmap(getFootForces, in_axes=(None, 0)))
    
    self.reset()
    
  def reset(self):
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
    
    if self.verbose: print("\nInitializing new simulations...")
    
    #load model from XML
    self.model = mujoco.MjModel.from_xml_path(self.xml_path)
    self.model.opt.timestep = self.timestep
    self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self.model.opt.iterations = 5
    self.model.opt.ls_iterations = 5

    #initialize instance parameters
    self.next_force_start_times = jp.zeros((self.count))
    self.next_force_durations = jp.zeros((self.count))
    self.next_force_magnitudes = jp.zeros((self.count))
    self.next_force_bodies = jp.zeros((self.count))
    self.next_force_directions = jp.zeros((self.count, 2))
    self.previous_torso_local_velocity = jp.zeros((self.count, 3))
    # save joint addresses
    self.joint_qpos_idx = []
    self.joint_torque_idx = []
    for joint in JOINT_NAMES:
      self.joint_torque_idx.append(self.model.jnt_dofadr[self.model.joint(joint).id])
      self.joint_qpos_idx.append(self.model.jnt_qposadr[self.model.joint(joint).id])
    self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
    self.joint_torque_idx = jp.array(self.joint_torque_idx)
    # save gravity vector
    self.gravity_vector = self.model.opt.gravity
    self.gravity_vector_batch = jp.array([self.gravity_vector]*self.count)
    # save torso body index
    self.torso_idx = self.model.body(TORSO_BODY_NAME).id
    # get pressure sensor geom ids
    self.pressure_sensor_ids = [self.model.geom(pressure_sensor_geom).id for pressure_sensor_geom in PRESSURE_GEOM_NAMES]
    
    # RANDOMIZATION
    # floor friction (0.5 to 1.0)
    self.model.geom('floor').friction = [coef * (1.0*(1.0-self.randomization_factor) + random.uniform(FLOOR_FRICTION_MIN_MULTIPLIER, FLOOR_FRICTION_MAX_MULTIPLIER)*self.randomization_factor) for coef in self.model.geom('floor').friction]    
    # vary the mass of all limbs randomly
    for i in range(self.model.nbody-1): self.model.body(i+1).mass[0] += random.uniform(-MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor, MAX_MASS_CHANGE_PER_LIMB*self.randomization_factor)
    # attach a random external mass (up to 0.1 kg) to a randomly chosen limb
    self.model.body(random.randint(1, self.model.nbody - 1)).mass[0] += random.uniform(0, MAX_EXTERNAL_MASS_ADDED*self.randomization_factor)
    # randomize IMU Z
    self.imu_z_offset = jax.random.uniform(key=self.rng_key, shape=(self.count,), minval=-IMU_Z_OFFSET_MAX, maxval=IMU_Z_OFFSET_MAX)
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
      kp = max(0, JOINT_P_GAIN + random.uniform(-JOINT_P_GAIN_MAX_CHANGE, JOINT_P_GAIN_MAX_CHANGE)*self.randomization_factor)
      self.model.actuator(joint).gainprm[0] = kp
      self.model.actuator(joint).biasprm[1] = -kp
      
    # create MJX model/data from CPU model/data
    self.cpu_model = self.model
    self.model = mjx.put_model(self.cpu_model)
    mj_data = mujoco.MjData(self.cpu_model)
    mujoco.mj_kinematics(self.cpu_model, mj_data)
    self.base_mjx_data = mjx.put_data(self.cpu_model, mj_data)
    
    # randomize joint initial states (GPU)
    self.data_batch = jax.vmap(lambda rng: self.base_mjx_data.replace(qpos=self.base_mjx_data.qpos + jax.random.uniform(rng, self.base_mjx_data.qpos.shape, minval=self.randomization_factor*-1.0*JOINT_INITIAL_STATE_OFFSET_MAX/180.0*jp.pi, maxval=self.randomization_factor*JOINT_INITIAL_STATE_OFFSET_MAX/180.0*jp.pi)))(self.rng)

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
      self.joint_angles_buffer.append(jp.array([[0]*len(JOINT_NAMES)]*self.count))
    for i in range((int)(self.local_ang_vel_delay/self.actual_timestep)):
      self.local_ang_vel_buffer.append(jp.array([[0]*3]*self.count))
    for i in range((int)(self.torso_global_velocity_delay/self.actual_timestep)):
      self.torso_global_velocity_buffer.append(jp.array([[0]*2]*self.count))
    for i in range((int)(self.torso_local_accel_delay/self.actual_timestep)):
      self.torso_local_accel_buffer.append(jp.array([[0]*3]*self.count))
    for i in range((int)(self.local_gravity_vector_delay/self.actual_timestep)):
      self.local_gravity_vector_buffer.append(jp.array([self.gravity_vector]*self.count))
    for i in range((int)(self.pressure_values_delay/self.actual_timestep)):
      self.pressure_values_buffer.append(jp.array([[0]*len(PRESSURE_GEOM_NAMES)]*self.count))
    
    # initialize environment properties
    self.observation_shape = self.getObs().shape
    self.action_shape = self.data_batch.ctrl.shape
    self.lastAction = self.data_batch.ctrl

    if self.verbose: print("Simulations initialized.")

  def computeReward(self):
    if self.verbose: print("Computing rewards...")
    
    torso_global_velocity = self.data_batch.cvel[:, self.torso_idx][:, 3:]
    torso_z_pos = self.data_batch.xpos[:, self.torso_idx, 2] + self.imu_z_offset
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    joint_torques = self.data_batch.qfrc_constraint[:, self.joint_torque_idx] + self.data_batch.qfrc_smooth[:, self.joint_torque_idx]
    
    rewards, areTerminal = self.reward_fn(torso_global_velocity, torso_z_pos, torso_quat, joint_torques)
    
    if self.verbose: print("Rewards computed.")

    return np.array(rewards), np.array(areTerminal)
    
  def getObs(self):
    if self.verbose: print("Collecting observations...")
    
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    torso_global_vel = self.data_batch.cvel[:, self.torso_idx]
    
    # joint positions     20          Joint positions in radians
    joint_angles = self.data_batch.qpos[:, self.joint_qpos_idx] + (self.randomization_factor * (JOINT_ANGLE_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.count, len(self.joint_qpos_idx))))
    
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU (in torso reference frame)
    torso_global_ang_vel = torso_global_vel[:, 0:3]
    local_ang_vel = inverseRotateVectors(torso_quat, torso_global_ang_vel) + (self.randomization_factor * GYRO_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    # agent velocity      2           X and Y velocity of robot torso (global, NWU)
    torso_global_velocity = torso_global_vel[:, 3:] + (self.randomization_factor * VELOCIMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    # linear acceleration 3           Linear acceleration from IMU (local to torso)
    torso_local_velocity = inverseRotateVectors(torso_quat, torso_global_velocity)
    torso_local_accel = ((torso_local_velocity - self.previous_torso_local_velocity)/self.actual_timestep) + (self.randomization_factor * ACCELEROMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    self.previous_torso_local_velocity = torso_local_velocity
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    noisy_torso_quat = torso_quat + (self.randomization_factor * (IMU_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.count, 4)))
    local_gravity_vector = inverseRotateVectors(noisy_torso_quat, self.gravity_vector_batch)
    # foot pressure       8           Pressure values from foot sensors (N)
    pressure_values = self.getFootForces(self.pressure_sensor_ids, self.data_batch)

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
        
    if self.verbose: print("Observations collected.")
    
    return np.array(delayed_observations)
  
  def step(self, action=None):
    if self.verbose: print("Stepping simulations...")
    
    # cycle action through action buffer
    if action is None:
      action = self.data_batch.ctrl
    # TODO: actions should be -1 to 1, we need to map each entry to the corresponding joint limits in radians
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    self.data_batch = self.data_batch.replace(ctrl=jp.array(action_to_take))
    self.lastAction = action_to_take
    
    # apply forces to the robot to destabilise it
    xfrc_applied = applyExternalForces(self)
    self.data_batch = self.data_batch.replace(xfrc_applied=xfrc_applied)

    # step sims, update data batch
    self.data_batch = self.jax_step(self.model, self.data_batch)
    
    if self.verbose: print("Simulations stepped.")
  
if __name__ == "__main__":
    sim_batch = GPUBatchSimulation(count=256,
                                   xml_path=SIM_XML_PATH,
                                   reward_fn=standingRewardFn,
                                   physics_steps_per_control_step=5,
                                   timestep=0.005,
                                   randomization_factor=1,
                                   verbose=True)


    while True:
      areTerminal = np.array([False])
      while not np.all(areTerminal):
        observations = sim_batch.getObs()
        actions = [[100]*16]*sim_batch.count
        # actions = None
        sim_batch.step(actions)
        rewards, areTerminal = sim_batch.computeReward()
        print(rewards[0])
      sim_batch.reset()