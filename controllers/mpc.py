from mpc_utils import *

RENDER = False

mj_model, mj_data, renderer, JOINT_QPOS_IDX, JOINT_DOF_IDX, agent = get_mujoco_setup()

# rollout
mujoco.mj_resetData(mj_model, mj_data)
while True:
    start_time = time.time()

    # run planner for num_steps
    mj_data.ctrl = compute_action(agent, mj_data, 10)

    # step
    mujoco.mj_step(mj_model, mj_data)

    # render
    if RENDER:
        render(renderer, mj_data)
    else:
        end_time = time.time()
        control_time = end_time - start_time
        print(f"Control freq.: {1.0 / control_time}")
