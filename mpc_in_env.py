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
def render(mj_data):
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
    mj_model, mj_data, joint_positions, torso_orientation
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
    return best_offset + 0.01 # add a small offset so that the robot does not assume it can always push off the ground, makes for more robust policy


def set_mujoco_state(state, mj_model, mj_data):
    joint_positions, joint_velocities, torso_ang_vel, torso_orientation = state
    dist_to_ground = find_ideal_distance_to_ground(
        mj_model, mj_data, joint_positions, torso_orientation
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


if __name__ == "__main__":
    RENDER = False # make False to calculate control frequency
    PLANNER_HORIZON = 10
    # ----------- SETUP MUJOCO MPC -----------
    model_path = (
        pathlib.Path(__file__).parent
        / "mujoco_mpc/build/mjpc/tasks/humanoid_cap/stand/task.xml"
    )
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
    # ----------- SETUP ENVIRONMENT -----------
    env = CPUEnv(
        xml_path=SIM_XML_PATH,
        randomization_factor=0,
        enable_rendering=True,
    )
    done = False
    torque_ctrl = np.zeros(12)
    obs, _ = env.reset()
    # ----------- SIMULATION LOOP -----------
    while not done:
        start_time = time.time()
        joint_positions = obs[: len(JOINT_NAMES)]
        joint_velocities = obs[len(JOINT_NAMES) : 2 * len(JOINT_NAMES)]
        torso_ang_vel = obs[2 * len(JOINT_NAMES) : 3 + 2 * len(JOINT_NAMES)]
        torso_quat = env.torso_quat
        state = (
            joint_positions,  # radians
            joint_velocities,  # radians / s
            torso_ang_vel,  # local angular velocity, rad / s
            torso_quat,  # quaternion in WXYZ form of torso
        )
        mj_model, mj_data = set_mujoco_state(state, mj_model, mj_data)

        agent.set_state(
            time=mj_data.time,
            qpos=mj_data.qpos,
            qvel=mj_data.qvel,
            act=mj_data.act,
            mocap_pos=mj_data.mocap_pos,
            mocap_quat=mj_data.mocap_quat,
            userdata=mj_data.userdata,
        )

        # run planner for num_steps
        num_steps = PLANNER_HORIZON
        for _ in range(num_steps):
            agent.planner_step()

        # get ctrl from agent policy
        torque_ctrl = agent.get_action()
        end_time = time.time()
        obs, _, done, _, _ = env.step(torque_ctrl)
        if RENDER: env.render("human")
        else:
            control_time = end_time - start_time
            print(f"Control freq.: {1.0 / control_time}")
        # render(mj_data)

    renderer.close()
