import mujoco
import cv2
import numpy as np
import random

# CODE SNIPPETS FROM https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=HlRhFs_d3WLP

# MUJOCO REF FRAME: ENU

class Simulator:
  def __init__(self, xml_path, timestep=0.001, randomization_factor=0):
    self.model = mujoco.MjModel.from_xml_path(xml_path)
    self.data = mujoco.MjData(self.model)
    self.renderer = mujoco.Renderer(self.model, 720, 1080)
    mujoco.mj_kinematics(self.model, self.data)
    self.model.opt.timestep = timestep
    
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
    
    self.randomization_factor = randomization_factor
    self.timestep = timestep
    self.next_force_start_time = 0
    self.next_force_magnitude_x = 0
    self.next_force_magnitude_y = 0
    self.next_force_magnitude_z = 0
    self.next_force_duration = 0
    self.next_force_body = 0
    
  def resetSim(self):
    mujoco.mj_resetData(self.model, self.data)
    # RANDOMIZATION
    # floor friction (0.5 to 1.0)
    self.model.geom('floor').friction = [coef - (coef * random.uniform(0, 0.5*self.randomization_factor)) for coef in self.model.geom('floor').friction]
    self.model.geom('floor').friction = [0,0,0]
    # TODO doesnt update!!
    
    #noise in joint state measurements
    self.joint_angle_offset_deg = random.uniform(-2.9*self.randomization_factor, 2.9*self.randomization_factor)
    #noise in orientation measurements
    self.orientation_offset_x = random.uniform(-2*self.randomization_factor, 2*self.randomization_factor)
    self.orientation_offset_y = random.uniform(-2*self.randomization_factor, 2*self.randomization_factor)
    self.orientation_offset_z = random.uniform(-180, 180)
    #noise in position
    self.position_offset_x = random.uniform(-100, 100)
    self.position_offset_y = random.uniform(-100, 100)
    self.position_offset_z = random.uniform(-0.1*self.randomization_factor, 0.1*self.randomization_factor)
    #delays in actions and observations
    self.action_delay = random.uniform(0.01*self.randomization_factor, 0.05*self.randomization_factor)
    self.observation_delay = random.uniform(0.01*self.randomization_factor, 0.05*self.randomization_factor)
    #round delays to be multiples of the timestep
    self.observation_delay = round(self.observation_delay / self.timestep) * self.timestep
    self.action_delay = round(self.action_delay / self.timestep) * self.timestep
    #make buffers for observations and actions
    self.observation_buffer = [None] * (int)(self.observation_delay/self.timestep)
    self.action_buffer = [None] * (int)(self.action_delay/self.timestep)
    # vary the mass of all limbs randomly
    for i in range(self.model.nbody-1): self.model.body(i+1).mass[0] += random.uniform(-0.05*self.randomization_factor, 0.05*self.randomization_factor)
    # attach a random external mass (up to 0.5 kg) to a randomly chosen limb
    self.model.body(random.randint(0, self.model.nbody - 1)).mass[0] += random.uniform(0*self.randomization_factor, 0.1*self.randomization_factor)
    # randomize the joints (properties as well as states)! TODO
    
    self.data = mujoco.MjData(self.model)
      
  def getState(self):
    # RANDOMIZATION TODO
    # joint angular offsets (±2.9°)
    # vary the orientation (up to 2°) of IMU
    # vary the X/Y/Z position (up to 5 mm)
    # vary 
    # random time delays (10 ms to 50 ms) 
    action_time = self.data.time
    pass
    
  def takeAction(self, action):
    # cycle action through action buffer
    self.action_buffer.append(action)
    action_to_take = self.action_buffer.pop(0)
    if action_to_take is not None:
        self.data.ctrl = action_to_take
    
    # apply forces to the robot to destabilise it
    if self.data.time > self.next_force_start_time + self.next_force_duration:
      self.next_force_start_time = self.data.time + random.uniform(1, 3)
      self.next_force_duration = random.uniform(0.05*self.randomization_factor, 0.15*self.randomization_factor)
      self.next_force_magnitude_x = random.uniform(-7.5*self.randomization_factor, 7.5*self.randomization_factor)
      self.next_force_magnitude_y = random.uniform(-7.5*self.randomization_factor, 7.5*self.randomization_factor)
      self.data.xfrc_applied[self.next_force_body][0] = 0
      self.data.xfrc_applied[self.next_force_body][1] = 0
      self.next_force_body = random.randint(1, len(self.data.xfrc_applied) - 1)
    if self.data.time > self.next_force_start_time and self.data.time < self.next_force_start_time + self.next_force_duration:
      self.data.xfrc_applied[self.next_force_body][0] = self.next_force_magnitude_x
      self.data.xfrc_applied[self.next_force_body][1] = self.next_force_magnitude_y
    
    # step simulation
    mujoco.mj_step(self.model, self.data)
    
  def render(self):
    self.renderer.update_scene(self.data, camera="track", scene_option=self.scene_option)
    cv2.imshow("Sim View", self.renderer.render())
    cv2.waitKey(1)

    
if __name__ == "__main__":
    sim = Simulator("assets/flat.xml", timestep=0.005, randomization_factor=1)
    sim.resetSim()
    while sim.data.time < 5:
      sim.takeAction(None)
      sim.render()