import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from simulation_parameters import *
from reward_functions import *
import gc
import random
import quaternion
import multiprocessing

class GPUBatchSimulation:
  def __init__(self, count, xml_path, reward_fn, physics_steps_per_control_step=5, timestep=0.001, randomization_factor=0, verbose=False):
    if jax.default_backend() != 'gpu':
      print("ERROR: failed to find GPU device.")
      exit()
      
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.count = count
    self.reward_fn = reward_fn
    self.physics_steps_per_control_step = physics_steps_per_control_step
    self.rng_key = jax.random.PRNGKey(42)
    self.rng = jax.random.split(self.rng_key, self.count)
    self.verbose = verbose
    
    self.reset()
    
  def reset(self):
    try: del self.model
    except: pass
    try: del self.renderer
    except: pass
    try: del self.cpu_model
    except: pass
    try: del self.cpu_data
    except: pass
    gc.collect()
    
    if self.verbose: print("Creating new simulations...")
    
    #load model from XML
    self.model = mujoco.MjModel.from_xml_path(self.xml_path)
    self.model.opt.timestep = self.timestep
    self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self.model.opt.iterations = 3
    self.model.opt.ls_iterations = 5
    self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
   
    #initialize instance parameters
    self.next_force_start_times = jp.zeros((self.count))
    self.next_force_durations = jp.zeros((self.count))
    self.next_force_magnitudes = jp.zeros((self.count))
    self.next_force_bodies = jp.zeros((self.count))
    self.next_force_directions = jp.zeros((self.count, 2))
    
    # RANDOMIZATION
    # floor friction (0.5 to 1.0)
    self.model.geom('floor').friction = [coef * (1.0*(1.0-self.randomization_factor) + random.uniform(FLOOR_FRICTION_MIN_MULTIPLIER, FLOOR_FRICTION_MAX_MULTIPLIER)*self.randomization_factor) for coef in self.model.geom('floor').friction]    
    #delays in actions and observations (10ms to 50ms)
    self.action_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    self.observation_delay = random.uniform(MIN_DELAY*self.randomization_factor, MAX_DELAY*self.randomization_factor)
    #round delays to be multiples of the timestep
    actual_timestep = self.timestep * self.physics_steps_per_control_step
    self.observation_delay = round(self.observation_delay / actual_timestep) * actual_timestep
    self.action_delay = round(self.action_delay / actual_timestep) * actual_timestep
    #make buffers for observations and actions
    self.observation_buffer = [None] * (int)(self.observation_delay/actual_timestep)
    self.action_buffer = [None] * (int)(self.action_delay/actual_timestep)
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

    self.cpu_model = self.model
    self.model = mjx.put_model(self.cpu_model)
    mjx_step = jax.jit(mjx.step)
    self.jax_step = jax.vmap(mjx_step, in_axes=(None, 0))
    mjx_data = mjx.put_data(self.cpu_model, mujoco.MjData(self.cpu_model))
    
    # randomize joint initial states (GPU)
    self.data_batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, mjx_data.qpos.shape, minval=-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi, maxval=JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi)))(self.rng)
    
    if self.verbose: print("Simulations initialized.")

  def computeReward(self): #TODO - improve speed
    
    if self.verbose: print("Computing rewards...")
    
    self.batch_cpu_data = mjx.get_data(self.cpu_model, self.data_batch)
    rewards = np.zeros((self.count))

    for i in range(self.count):
      rewards[i] = self.reward_fn(self.batch_cpu_data[i])
        
    if self.verbose: print("Rewards computed.")

    return rewards
    
  def getObs(self): #TODO - improve speed
    
    if self.verbose: print("Collecting observations...")
    
    self.batch_cpu_data = mjx.get_data(self.cpu_model, self.data_batch)
  
    observations = []
  
    for i in range(self.count):
      observation = []
    
      # joint positions     20          Joint positions in radians
      for joint in JOINT_SENSOR_NAMES:
        observation.append(self.batch_cpu_data[i].sensor(joint).data.copy()[0] + random.gauss(0, JOINT_ANGLE_NOISE_STDDEV/180.0*np.pi)) # LOCAL FRAME (PER-JOINT)
      # linear acceleration 3           Linear acceleration from IMU
      observation.extend([val + random.gauss(0, ACCELEROMETER_NOISE_STDDEV) for val in self.batch_cpu_data[i].sensor('accelerometer').data.copy()]) # LOCAL FRAME (IMU)
      # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU
      observation.extend([val + random.gauss(0, GYRO_NOISE_STDDEV) for val in self.batch_cpu_data[i].sensor('gyro').data.copy()]) # LOCAL FRAME (IMU)
      # foot pressure       8           Pressure values from foot sensors
      for pressure_sensor in PRESSURE_SENSOR_NAMES:
        observation.append(self.batch_cpu_data[i].sensor(pressure_sensor).data.copy()[0] + random.gauss(0, PRESSURE_SENSOR_NOISE_STDDEV))
      # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
      global_gravity_vector = self.model.opt.gravity
      world_IMU_quat_world = [val + random.gauss(0, IMU_NOISE_STDDEV/180.0*np.pi) for val in self.batch_cpu_data[i].sensor("IMU_quat").data.copy()]
      local_gravity_vector_IMU = quaternion.rotate_vectors(np.quaternion(*world_IMU_quat_world).inverse(), global_gravity_vector)
      observation.extend(local_gravity_vector_IMU) # LOCAL FRAME (IMU)
      # agent velocity      2           X and Y velocity of robot torso
      observation.extend([val + random.gauss(0, VELOCIMETER_NOISE_STDDEV) for val in self.batch_cpu_data[i].sensor('velocimeter').data.copy()]) # LOCAL FRAME (IMU)

      observations.append(observation)

    # cycle observation through observation buffer
    self.observation_buffer.append(np.array(observations))
    delayed_observations = self.observation_buffer.pop(0)
    
    if self.verbose: print("Observations collected.")
    
    return delayed_observations
  
  def step(self, action=None):
    
    if self.verbose: print("Stepping simulations...")
    
    # cycle action through action buffer
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    if action_to_take is not None:
      self.data_batch.replace(ctrl=jp.array(action_to_take))
    
    # apply forces to the robot to destabilise it
    
    # check which simulations need new random force times/durations/magnitudes/directions/target bodies
    should_update_force = self.data_batch.time > (self.next_force_start_times + self.next_force_durations)
    
    # for the simulations which need to be updated, randomly generate new values
    if jp.any(should_update_force):
      N = jp.sum(should_update_force)
      updated_next_force_start_times = self.data_batch.time[should_update_force] + jax.random.uniform(key=self.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_INTERVAL, maxval=MAX_EXTERNAL_FORCE_INTERVAL)
      updated_next_force_durations = jax.random.uniform(key=self.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_DURATION, maxval=MAX_EXTERNAL_FORCE_DURATION)
      updated_next_force_magnitudes = jax.random.uniform(key=self.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor, maxval=MAX_EXTERNAL_FORCE_MAGNITUDE*self.randomization_factor)
      updated_next_force_bodies = jax.random.randint(key=self.rng_key, shape=(N,), minval=1, maxval=len(self.data_batch.xfrc_applied) - 1)
      updated_next_force_directions = jax.random.ball(key=self.rng_key, d=2, shape=(N,))
      
      self.next_force_start_times.at[should_update_force].set(updated_next_force_start_times)
      self.next_force_durations.at[should_update_force].set(updated_next_force_durations)
      self.next_force_magnitudes.at[should_update_force].set(updated_next_force_magnitudes)
      self.next_force_bodies.at[should_update_force].set(updated_next_force_bodies)
      self.next_force_directions.at[should_update_force].set(updated_next_force_directions)

    # apply force values (times/durations/etc.) to corresponding simulations
    should_apply_force = jp.logical_and((self.data_batch.time > self.next_force_start_times), (self.data_batch.time < (self.next_force_start_times + self.next_force_durations)))
    
    if jp.any(should_apply_force):
      xfrc_applied = jp.zeros(self.data_batch.xfrc_applied.shape)
      applied_forces_x = self.next_force_directions[should_apply_force][0] * self.next_force_magnitudes[should_apply_force]
      applied_forces_y = self.next_force_directions[should_apply_force][1] * self.next_force_magnitudes[should_apply_force]
      xfrc_applied.at[should_apply_force][self.next_force_bodies[should_apply_force]][0].set(applied_forces_x)
      xfrc_applied.at[should_apply_force][self.next_force_bodies[should_apply_force]][1].set(applied_forces_y)
    
      self.data_batch.replace(xfrc_applied=xfrc_applied)
    
    # step simulation
    for s in range(self.physics_steps_per_control_step):
      self.data_batch = self.jax_step(self.model, self.data_batch)
      if self.verbose: print("{}%".format((s+1)/self.physics_steps_per_control_step), end='\r')
    
    if self.verbose: print("Simulations stepped.               ")

if __name__ == "__main__":
    sim_batch = GPUBatchSimulation(count=512, xml_path="assets/world.xml", reward_fn=standingRewardFn, physics_steps_per_control_step=5, timestep=0.005, randomization_factor=1, verbose=True)

    while True:
      while all(sim_batch.data_batch.time < 2):
        observations = sim_batch.getObs()
        actions = [[0]*4]*sim_batch.count
        sim_batch.step(actions)
        rewards = sim_batch.computeReward()
      sim_batch.reset()