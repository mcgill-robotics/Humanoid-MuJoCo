import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from simulation_parameters import *
from reward_functions import *
from jax.scipy.spatial.transform import Rotation
import gc
import random
import quaternion
import time

class GPUBatchSimulation:
  def __init__(self, count, xml_path, reward_fn, physics_steps_per_control_step=5, timestep=0.001, randomization_factor=0, verbose=False):
    if jax.default_backend() != 'gpu':
      print("ERROR: failed to find GPU device.")
      exit()
      
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.count = count
    self.reward_fn = jax.jit(jax.vmap(lambda v, z, q, jt : reward_fn(v, z, q, jt)))
    self.physics_steps_per_control_step = physics_steps_per_control_step
    self.rng_key = jax.random.PRNGKey(42)
    self.rng = jax.random.split(self.rng_key, self.count)
    self.verbose = verbose
    
    self.reset()
    
  def reset(self):    
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
    self.previous_torso_local_velocity = jp.zeros((self.count, 3))
    # save joint addresses in data.qpos
    self.joint_qpos_idx = []
    for joint in JOINT_NAMES:
      self.joint_qpos_idx.append(self.model.jnt_qposadr[self.model.joint(joint).id])
    self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
    # save gravity vector
    self.gravity_vector = self.model.opt.gravity
    # save torso body index
    self.torso_idx = self.model.body(TORSO_BODY_NAME).id
    #define vector rotation helper function
    self.__inverseRotateVectors = jax.jit(jax.vmap(lambda q, v : Rotation.from_quat(q).inv().apply(v)))
    
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
    self.model.body(random.randint(1, self.model.nbody - 1)).mass[0] += random.uniform(0, MAX_EXTERNAL_MASS_ADDED*self.randomization_factor)
    
    
    # TODO
    # randomize IMU X/Y/Z/Quat
    for i in range(len(self.model.site("IMU").pos)):
      self.model.site("IMU").pos[i] += random.uniform(-IMU_POS_OFFSET_MAX*self.randomization_factor, IMU_POS_OFFSET_MAX*self.randomization_factor)
    self.IMU_offset_quat = np.quaternion(*self.model.site("IMU").quat) * quaternion.from_euler_angles([random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*jp.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*jp.pi)*self.randomization_factor), random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*jp.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*jp.pi)*self.randomization_factor), random.uniform((-IMU_ORIENTATION_OFFSET_MAX/180.0*jp.pi)*self.randomization_factor, (IMU_ORIENTATION_OFFSET_MAX/180.0*jp.pi)*self.randomization_factor)])
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

    # create MJX model/data from CPU model/data
    self.cpu_model = self.model
    self.model = mjx.put_model(self.cpu_model)
    mjx_data = mjx.put_data(self.cpu_model, mujoco.MjData(self.cpu_model))
    
    # define step function (rollout)
    def rollout(m, d):
      for _ in range(self.physics_steps_per_control_step):
        d = mjx.step(m, d)
      return d
    jax_rollout = jax.jit(rollout)
    self.jax_step = jax.vmap(jax_rollout, in_axes=(None, 0))
    
    # randomize joint initial states (GPU)
    self.data_batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, mjx_data.qpos.shape, minval=-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*jp.pi, maxval=JOINT_INITIAL_STATE_OFFSET_MAX/180.0*jp.pi)))(self.rng)

    # step sim (to populate self.data)
    self.step()
    
    # clean up any unreferenced variables
    gc.collect()
    
    if self.verbose: print("Simulations initialized.")

  def computeReward(self):
    if self.verbose: print("Computing rewards...")
    
    torso_global_velocity = self.data_batch.cvel[:, self.torso_idx][:, 3:]
    torso_z_pos = self.data_batch.xpos[:, self.torso_idx, 2]
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    joint_torques = self.data_batch.qfrc_constraint + self.data_batch.qfrc_smooth
    
    rewards = self.reward_fn(torso_global_velocity, torso_z_pos, torso_quat, joint_torques)
    
    if self.verbose: print("Rewards computed.")

    return rewards
    
  def getObs(self):
    if self.verbose: print("Collecting observations...")
    
    torso_quat = self.data_batch.xquat[:, self.torso_idx]
    torso_global_vel = self.data_batch.cvel[:, self.torso_idx]
    
    # joint positions     20          Joint positions in radians
    joint_angles = self.data_batch.qpos[:, self.joint_qpos_idx] + ((JOINT_ANGLE_NOISE_STDDEV/180.0*jp.pi) * jax.random.normal(key=self.rng_key, shape=(self.count, len(self.joint_qpos_idx))))
    
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU
    torso_global_ang_vel = torso_global_vel[:, 0:3]
    local_ang_vel = self.__inverseRotateVectors(torso_quat, torso_global_ang_vel) + (GYRO_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    # agent velocity      2           X and Y velocity of robot torso
    torso_global_velocity = torso_global_vel[:, 3:] + (VELOCIMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    # linear acceleration 3           Linear acceleration from IMU
    torso_local_velocity = self.__inverseRotateVectors(torso_quat, torso_global_velocity)
    torso_local_accel = (torso_local_velocity - self.previous_torso_local_velocity) + (ACCELEROMETER_NOISE_STDDEV * jax.random.normal(key=self.rng_key, shape=(self.count, 3)))
    self.previous_torso_local_velocity = torso_local_velocity
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    noisy_torso_quat = torso_quat + ((IMU_NOISE_STDDEV/180.0*np.pi) * jax.random.normal(key=self.rng_key, shape=(self.count, 4)))
    local_gravity_vector = self.__inverseRotateVectors(noisy_torso_quat, jp.array([self.gravity_vector]*self.count))
    # foot pressure       8           Pressure values from foot sensors
    print(self.data_batch.contact)
    
    observations = jp.hstack((joint_angles, local_ang_vel, torso_global_velocity[:, 0:2], torso_local_accel, local_gravity_vector))
  
    # cycle observation through observation buffer
    self.observation_buffer.append(observations)
    delayed_observations = self.observation_buffer.pop(0)
    
    if self.verbose: print("Observations collected.")
    
    return delayed_observations
  
  def __applyExternalForces(self):
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
  
  def step(self, action=None):
    if self.verbose: print("Stepping simulations...")
    
    # cycle action through action buffer
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    if action_to_take is not None:
      self.data_batch.replace(ctrl=jp.array(action_to_take))
    
    # apply forces to the robot to destabilise it
    self.__applyExternalForces()

    # step sims, update data batch
    self.data_batch = self.jax_step(self.model, self.data_batch)
    
    if self.verbose: print("Simulations stepped.")

if __name__ == "__main__":
    sim_batch = GPUBatchSimulation(count=512,
                                   xml_path="assets/world.xml",
                                   reward_fn=standingRewardFn,
                                   physics_steps_per_control_step=5,
                                   timestep=0.005,
                                   randomization_factor=1,
                                   verbose=True)

    while True:
      while all(sim_batch.data_batch.time < 2):
        observations = sim_batch.getObs()
        actions = [[0]*4]*sim_batch.count
        sim_batch.step(actions)
        rewards = sim_batch.computeReward()
      sim_batch.reset()