import mujoco
import numpy as np
from simulation.simulation_parameters import CONTROL_FREQUENCY
import cv2
import time
import os
import pathlib

from mujoco_mpc import agent as agent_lib

model_path = (
    pathlib.Path(__file__).parent
    / "mujoco_mpc/build/mjpc/tasks/humanoid_cap/stand/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))
renderer = mujoco.Renderer(model)
scene_option = mujoco.MjvOption()
mujoco.mjv_defaultOption(scene_option)
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
data = mujoco.MjData(model)
agent = agent_lib.Agent(task_id="Humanoid Cap Stand", model=model)

# rollout
mujoco.mj_resetData(model, data)
while True:
    # set planner state
    agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=data.mocap_pos,
        mocap_quat=data.mocap_quat,
        userdata=data.userdata,
    )

    # run planner for num_steps
    num_steps = 10
    for _ in range(num_steps):
        agent.planner_step()

    # set ctrl from agent policy
    data.ctrl = agent.get_action()

    # step
    mujoco.mj_step(model, data)

    # render
    renderer.update_scene(data, camera="track", scene_option=scene_option)
    frame = renderer.render()
    # time.sleep(1 / CONTROL_FREQUENCY)
    cv2.imshow("CPU Sim View", frame)
    cv2.waitKey(1)
