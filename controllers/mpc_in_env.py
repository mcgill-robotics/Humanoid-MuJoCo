from mpc_utils import *

if __name__ == "__main__":
    RENDER = True  # make False to calculate control frequency
    PLANNER_HORIZON = 10
    # ----------- SETUP MUJOCO MPC -----------
    mj_model, mj_data, renderer, JOINT_QPOS_IDX, JOINT_DOF_IDX, agent = (
        get_mujoco_setup()
    )
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
        mj_model, mj_data = set_mujoco_state(
            state, mj_model, mj_data, JOINT_QPOS_IDX, JOINT_DOF_IDX
        )
        torque_ctrl = compute_action(agent, mj_data, PLANNER_HORIZON)

        torque_noise = np.random.normal(0, 0.15 * 1.5, torque_ctrl.shape)
        torque_ctrl += torque_noise

        end_time = time.time()
        obs, _, done, _, _ = env.step(torque_ctrl)
        control_time = end_time - start_time
        print(f"Control freq.: {1.0 / control_time}")
        if RENDER:
            env.render("human")
        # render(renderer, mj_data)

    renderer.close()
