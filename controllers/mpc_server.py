import socket
import json
import time
import numpy as np
import cv2
from mpc_utils import *

host = "localhost"
port = 5555

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print(f"Server running on {host}:{port}...")

mj_model, mj_data, renderer, JOINT_QPOS_IDX, JOINT_DOF_IDX, agent = get_mujoco_setup()

PLANNER_HORIZON = 2
TORQUE_TO_POSITION_MULTIPLIER = 0.3
POSITION_CONTROL = False
IGNORE_ANG_VEL = False
DEBUG = True

while True:
    client_socket, client_address = (
        server_socket.accept()
    )  # Wait for a client to connect
    print(f"Connection established!")

    while True:
        try:
            message = client_socket.recv(1024).decode()

            num_jsons_in_message = message.count("{")
            if num_jsons_in_message == 0:
                continue
            last_json_end_index = message.rfind("}") + 1
            last_json_start_index = message.rfind("{")
            message = message[last_json_start_index:last_json_end_index]

            if message:
                start_time = time.time()

                decoded_json = json.loads(message)
                state = (
                    decoded_json["joint_pos"],
                    decoded_json["joint_vel"],
                    decoded_json["ang_vel"],
                    decoded_json["quat"],
                )
                state = [np.array(s) for s in state]
                if IGNORE_ANG_VEL:
                    state[2] = np.zeros_like(state[2])

                mj_model, mj_data = set_mujoco_state(
                    state, mj_model, mj_data, JOINT_QPOS_IDX, JOINT_DOF_IDX
                )
                torque_ctrl = compute_action(agent, mj_data, PLANNER_HORIZON)
                if POSITION_CONTROL:
                    command = state[0] + TORQUE_TO_POSITION_MULTIPLIER * torque_ctrl
                else:
                    command = torque_ctrl

                response = json.dumps([float(x) for x in command])
                client_socket.sendall(response.encode())

                end_time = time.time()
                control_time = end_time - start_time
                print(f"Control freq.: {1.0 / control_time}")

                if DEBUG:
                    frame = render(renderer, mj_data, display=False)
                    cv2.imwrite("debug_pose.jpeg", frame)

        except socket.timeout:
            print("Timeout: No message received from client.")
            client_socket.close()
            break
