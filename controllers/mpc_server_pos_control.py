from mpc_utils import *
import zmq
import time
import json

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
print("Server running...")

mj_model, mj_data, renderer, JOINT_QPOS_IDX, JOINT_DOF_IDX, agent = get_mujoco_setup()

PLANNER_HORIZON = 2
DEBUG = True

TORQUE_TO_POSITION_MULTIPLIER = 0.2

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
    
    target_joint_positions = state[0] + TORQUE_TO_POSITION_MULTIPLIER * torque_ctrl

    # render(renderer, mj_data)
    socket.send_string(
        json.dumps([float(x) for x in target_joint_positions])
    )  # Send the response back
    end_time = time.time()
    control_time = end_time - start_time
    print(f"Control freq.: {1.0 / control_time}")
    if DEBUG:
        # save image for debugging purposes
        frame = render(renderer, mj_data, display=False)
        cv2.imwrite("debug_pose.jpeg", frame)
