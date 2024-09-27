# server.py
import zmq
import time


def my_function(data):
    # Example processing function
    return f"Processed: {data}"


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Server running...")

while True:
    message = socket.recv_string()  # Receive a request
    print(f"Received request: {message}")

    # Call your function
    response = my_function(message)

    socket.send_string(response)  # Send the response back
