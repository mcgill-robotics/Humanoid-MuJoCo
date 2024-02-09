import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import cv2
import random
import quaternion
from simulation_parameters import *
from reward_functions import *
import gc
import os

class CPUSimulation:
  def __init__(self, xml_path, reward_fn, physics_steps_per_control_step=5, timestep=0.001, randomization_factor=0):
    self.xml_path = xml_path
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.reward_fn = reward_fn
    self.physics_steps_per_control_step = physics_steps_per_control_step
    
    self.reset()
    
  def reset(self):
    try: del self.model
    except: pass
    try: del self.data
    except: pass
    try: del self.renderer
    except: pass
    
    gc.collect()
    #load model from XML
    self.model = mujoco.MjModel.from_xml_path(self.xml_path)
    if os.environ.get('RENDER_SIM', "True") == "True": self.renderer = mujoco.Renderer(self.model, 720, 1080)
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
    actual_timestep = self.physics_steps_per_control_step * self.timestep
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
      
    # create data from model
    self.cpu_model = None
    self.data = mujoco.MjData(self.model)
    mujoco.mj_kinematics(self.model, self.data)
    # randomize joint initial states (CPU)
    for i in range(len(self.data.qpos)):
      self.data.qpos[i] += random.uniform(-JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi, JOINT_INITIAL_STATE_OFFSET_MAX/180.0*np.pi)*self.randomization_factor
      
  def getObs(self):
    observations = []

    # joint positions     20          Joint positions in radians
    for joint in JOINT_SENSOR_NAMES:
      observations.append(self.data.sensor(joint).data.copy()[0] + random.gauss(0, JOINT_ANGLE_NOISE_STDDEV/180.0*np.pi)) # LOCAL FRAME (PER-JOINT)
    # linear acceleration 3           Linear acceleration from IMU
    observations.extend([val + random.gauss(0, ACCELEROMETER_NOISE_STDDEV) for val in self.data.sensor('accelerometer').data.copy()]) # LOCAL FRAME (IMU)
    # angular velocity    3           Angular velocity (roll, pitch, yaw) from IMU
    observations.extend([val + random.gauss(0, GYRO_NOISE_STDDEV) for val in self.data.sensor('gyro').data.copy()]) # LOCAL FRAME (IMU)
    # foot pressure       8           Pressure values from foot sensors
    for pressure_sensor in PRESSURE_SENSOR_NAMES:
      observations.append(self.data.sensor(pressure_sensor).data.copy()[0] + random.gauss(0, PRESSURE_SENSOR_NOISE_STDDEV))
    # gravity             3           Gravity direction, derived from angular velocity using Madgwick filter
    global_gravity_vector = self.model.opt.gravity
    world_IMU_quat_world = [val + random.gauss(0, IMU_NOISE_STDDEV/180.0*np.pi) for val in self.data.sensor("IMU_quat").data.copy()]
    local_gravity_vector_IMU = quaternion.rotate_vectors(np.quaternion(*world_IMU_quat_world).inverse(), global_gravity_vector)
    observations.extend(local_gravity_vector_IMU) # LOCAL FRAME (IMU)
    # agent velocity      2           X and Y velocity of robot torso
    observations.extend([val + random.gauss(0, VELOCIMETER_NOISE_STDDEV) for val in self.data.sensor('velocimeter').data.copy()]) # LOCAL FRAME (IMU)

    # cycle observation through observation buffer
    self.observation_buffer.append(observations)
    delayed_observation = self.observation_buffer.pop(0)
    
    return delayed_observation
    
  def step(self, action=None):
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
      self.next_force_body = random.randint(1, len(self.data.xfrc_applied) - 1)
      self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      self.data.xfrc_applied[self.next_force_body][0] = 0
      self.data.xfrc_applied[self.next_force_body][1] = 0
      
      while np.linalg.norm(self.next_force_direction) == 0: self.next_force_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
      self.next_force_direction = self.next_force_direction / np.linalg.norm(self.next_force_direction)
    if self.data.time > self.next_force_start_time and self.data.time < self.next_force_start_time + self.next_force_duration:
      self.data.xfrc_applied[self.next_force_body][0] = self.next_force_direction[0] * self.next_force_magnitude
      self.data.xfrc_applied[self.next_force_body][1] = self.next_force_direction[1] * self.next_force_magnitude
    
    # step simulation
    for _ in range(self.physics_steps_per_control_step):
      mujoco.mj_step(self.model, self.data)
    return self.reward_fn(self.data)
    
  def render(self, display=True):
    if not os.environ.get('RENDER_SIM', "True") == "True": return np.zeros((100,100))
    self.renderer.update_scene(self.data, camera="track", scene_option=self.scene_option)
    frame = self.renderer.render()
    if display:
      cv2.imshow("Sim View", frame)
      cv2.waitKey(1)
    return frame
    
if __name__ == "__main__":
    sim = CPUSimulation(xml_path="assets/world.xml", reward_fn=standingRewardFn, timestep=0.01, randomization_factor=1)
    
    while True:
      while sim.data.time < 2:
        observation = sim.getObs()
        action = [0]*4
        reward = sim.step(action)
        print(reward)
        sim.render()
      sim.reset()