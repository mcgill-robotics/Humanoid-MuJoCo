from mpc_utils import *
import zmq
import time
import json

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
print("Server running...")

mj_model, mj_data, renderer, JOINT_QPOS_IDX, JOINT_DOF_IDX, agent = get_mujoco_setup()

PLANNER_HORIZON = 10
DEBUG = True

PREDICTED_POSITION_TIMESTEP = 1 / 50  # 50 Hz

num_position_prediction_iterations = max(1, TIMESTEP // PREDICTED_POSITION_TIMESTEP)

while True:
    message = socket.recv_string()  # Receive a request
    start_time = time.time()
    decoded_json = json.loads(message)
    state = (
        decoded_json["joint_pos"],
        decoded_json["joint_vel"],
        decoded_json["ang_vel"],
        decoded_json["quat"],
    )
    state = [np.array(s) for s in state]

    mj_model, mj_data = set_mujoco_state(
        state, mj_model, mj_data, JOINT_QPOS_IDX, JOINT_DOF_IDX
    )
    torque_ctrl = compute_action(agent, mj_data, PLANNER_HORIZON)
    mj_model, mj_data = set_mujoco_state(
        state, mj_model, mj_data, JOINT_QPOS_IDX, JOINT_DOF_IDX
    )
    mj_data.ctrl = torque_ctrl
    for _ in num_position_prediction_iterations:
        mujoco.mj_step(mj_model, mj_data)
    joint_positions_after_control = mj_data.qpos[JOINT_QPOS_IDX]

    # render(renderer, mj_data)
    socket.send_string(
        json.dumps([float(x) for x in joint_positions_after_control])
    )  # Send the response back
    end_time = time.time()
    control_time = end_time - start_time
    print(f"Control freq.: {1.0 / control_time}")

    if DEBUG:
        # save image for debugging purposes
        frame = render(renderer, mj_data, display=False)
        cv2.imwrite("debug_pose.jpeg", frame)
