import numpy as np
from simulation.cpu_env import CPUEnv
from simulation import SIM_XML_PATH
from simulation.simulation_parameters import (
    TIMESTEP,
    CONTROL_FREQUENCY,
    JOINT_NAMES,
)
import time
import cv2
import mujoco
import pathlib
from mujoco_mpc import agent as agent_lib


# Function to render current state of mujoco (for debugging)
def render(renderer, mj_data):
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    renderer.update_scene(mj_data, camera="track", scene_option=scene_option)
    frame = renderer.render()
    # time.sleep(1 / CONTROL_FREQUENCY)
    cv2.imshow("CPU Sim View", frame)
    cv2.waitKey(1)


# function to calculate the distance the robot should be offset vertically so it is touching the ground
# This is a workaround since we do not have full state observations: the assumption being the robot will always be touching the ground
def find_ideal_distance_to_ground(
    mj_model, mj_data, joint_positions, torso_orientation, JOINT_QPOS_IDX
):
    height_offsets = np.linspace(-0.2, 0.2, 100)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
        mujoco.mj_forward(mj_model, mj_data)
        mj_data.qacc = 0
        mj_data.qpos[3:7] = torso_orientation
        mj_data.qpos[JOINT_QPOS_IDX] = joint_positions
        mj_data.qpos[2] += offset
        mujoco.mj_inverse(mj_model, mj_data)
        vertical_forces.append(mj_data.qfrc_inverse[2])

    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]
    HOVER_DISTANCE = 0.0
    return best_offset + HOVER_DISTANCE # add a small offset so that the robot does not assume it can always push off the ground, makes for more robust policy


def set_mujoco_state(state, mj_model, mj_data, JOINT_QPOS_IDX, JOINT_DOF_IDX):
    joint_positions, joint_velocities, torso_ang_vel, torso_orientation = state
    dist_to_ground = find_ideal_distance_to_ground(
        mj_model, mj_data, joint_positions, torso_orientation, JOINT_QPOS_IDX
    )
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
    mj_data.qacc = 0
    # update mujoco state with observations
    mj_data.qpos[3:7] = torso_orientation
    mj_data.qpos[JOINT_QPOS_IDX] = joint_positions
    mj_data.qpos[2] += dist_to_ground
    mj_data.qvel[JOINT_DOF_IDX] = joint_velocities
    mj_data.qvel[3:6] = torso_ang_vel
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data

def get_mujoco_setup():
    model_path = "/Humanoid-MPC/mujoco_mpc/build/mjpc/tasks/humanoid_cap/stand/task.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_model.opt.timestep = TIMESTEP
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, 720, 1080)
    JOINT_QPOS_IDX = []
    JOINT_DOF_IDX = []
    for joint in JOINT_NAMES:
        JOINT_DOF_IDX.append(mj_model.jnt_dofadr[mj_model.joint(joint).id])
        JOINT_QPOS_IDX.append(mj_model.jnt_qposadr[mj_model.joint(joint).id])
    agent = agent_lib.Agent(task_id="Humanoid Cap Stand", model=mj_model)
    
    return mj_model, mj_data, renderer, JOINT_QPOS_IDX, JOINT_DOF_IDX, agent

def compute_action(agent, mj_data, planning_horizon):
    agent.set_state(
            time=mj_data.time,
            qpos=mj_data.qpos,
            qvel=mj_data.qvel,
            act=mj_data.act,
            mocap_pos=mj_data.mocap_pos,
            mocap_quat=mj_data.mocap_quat,
            userdata=mj_data.userdata,
        )

    # run planner for planning_horizon
    for _ in range(planning_horizon):
        agent.planner_step()

    # get ctrl from agent policy
    torque_ctrl = agent.get_action()
    return torque_ctrl