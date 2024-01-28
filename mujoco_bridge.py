import mujoco
import cv2
import numpy as np

# CODE SNIPPETS FROM https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=HlRhFs_d3WLP

# MUJOCO REF FRAME: ENU

model = mujoco.MjModel.from_xml_path("assets/test.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 720, 1080)
mujoco.mj_kinematics(model, data)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
mujoco.mjv_defaultOption(scene_option)
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# tweak scales of contact visualization elements
model.vis.scale.contactwidth = 0.1
model.vis.scale.contactheight = 0.03
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3

#visualize joints
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

def get_geom_speed(model, data, geom_name):
  """Returns the speed of a geom."""
  geom_vel = np.zeros(6)
  geom_type = mujoco.mjtObj.mjOBJ_GEOM
  geom_id = data.geom(geom_name).id
  mujoco.mj_objectVelocity(model, data, geom_type, geom_id, geom_vel, 0)
  return np.linalg.norm(geom_vel)

def displayFrame():
    renderer.update_scene(data, scene_option=scene_option)
    cv2.imshow("Sim View", renderer.render())
    cv2.waitKey(1)

# set timestep
print(model.opt.timestep)
model.opt.timestep = 0.005 # lower timestep = better accuracy and better energy conservation

mujoco.mj_resetData(model, data)
data.ctrl = 20

while data.time < 5:
  mujoco.mj_step(model, data)
  displayFrame()