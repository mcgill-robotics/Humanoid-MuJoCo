import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import cv2

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_path("assets/gpu.xml")
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(jax.devices())

jit_step = jax.jit(mjx.step)
# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < 5:
  mjx_data = jit_step(mjx_model, mjx_data)
  renderer.update_scene(mjx.get_data(mj_model, mjx_data), scene_option=scene_option)
  cv2.imshow("Sim View", renderer.render())
  cv2.waitKey(1)