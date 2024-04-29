import asyncio
import websockets
import json
import pdb
import socket
from util import *
from Global_Constants import *
from data_handler import data_handler
import time
import numpy as np

# Assuming data_handler is a class you've defined elsewhere that handles sensor data
data_handler = data_handler(Total_sensor_num=Total_sensor_num)
async def websocket_server(websocket, path):
    try:
        async for message in websocket:
            # print(f"message: {message}")
            data = json.loads(message)
            # print(data)
            complex_name = data['sensorId'].split(" ")[-1]
            'transfer id to readable names'
            udp_address = Sensor_ID_to_Name(complex_name)
            data_type = data['type']
            sensor_data = data['data']

            if data_type == "acceleration":
                processed_data = SensorData(type='Acceleration', timestamp=-1, udp_address=udp_address, sensor_id=-1, data=[sensor_data['x'], sensor_data['y'], sensor_data['z']])
            elif data_type == "rotation":
                processed_data = SensorData(type='Rotation', timestamp=-1, udp_address=udp_address, sensor_id=-1, data=[sensor_data['w'], sensor_data['x'], sensor_data['y'], sensor_data['z']])
            else:
                assert False, "Unsupported data type"
            print(f"Received {data_type} data from sensor {udp_address}: {sensor_data}")
            data_handler.new_data_available(processed_data)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")

async def main():
    async with websockets.serve(websocket_server, "0.0.0.0", 8765):
        await asyncio.Future()  # 运行服务器直到被手动停止

asyncio.run(main())

# import pdb
# import socket
# from util import *
# from Global_Constants import *
# from data_handler import data_handler
# import time
# import numpy as np
#
# class SocketHandler:
#     def __init__(self, host: str, port: int):
#         self.host = host  # IP address where the server will listen
#         self.port = port  # Port on which the server is listening
#         self.back_port = port + 1
#         # Assuming data_handler is a class you've defined elsewhere that handles sensor data
#         self.data_handler = data_handler(Total_sensor_num=Total_sensor_num)
#
#         # # Create socket instances for both primary and secondary ports
#         # self.secondary_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
#     def start_server(self):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
#             server_socket.bind((self.host, self.port))  # Bind to the specified host and port
#             # self.secondary_socket.bind((self.host, self.back_port))
#             # self.secondary_socket.listen()
#             server_socket.listen()  # Listen for incoming connections
#             print(f"Server is running on {self.host}:{self.port}, {self.back_port}")
#
#
#             while True:
#                 client_socket, addr = server_socket.accept()  # Accept a new connection
#                 print(f"Connected to client {addr}")
#
#                 # second_client_socket, second_addr = self.secondary_socket.accept()  # Accept a new connection
#                 # print(f"Connected to client {second_addr}")
#
#                 with client_socket:
#                     self.handle_client(client_socket, client_socket)
#
#     def handle_client(self, client_socket, second_client_socket):
#         data_processed = 0
#         start_time = time.time()
#
#         while True:
#             data = client_socket.recv(1024)  # Receive data from the client
#             if not data:
#                 break  # No more data from client
#             processed_data = parse_sensor_data(data.decode())
#             if processed_data is None:
#                 print(f"Decode error: {data.decode()}")
#                 continue
#             print(processed_data)
#             # self.data_handler.send_data(second_client_socket, np.asarray([[1,2,3,4,5,6,7]]), np.asarray([0,0,0]))
#             self.data_handler.new_data_available(processed_data, second_client_socket)
#             data_processed += 1
#
#             current_time = time.time()
#             elapsed_time = current_time - start_time
#
#             if elapsed_time >= 1:
#                 print(f"Data processed per second: {data_processed // elapsed_time}")
#                 data_processed = 0
#                 start_time = current_time
#
# # class Socket_Handler(object):
# #     def __init__(self, host: str, port: int):
# #         self.host = host  # Server's IP address
# #         self.port = port  # Port on which server is listening
# #
# #         self.data_handler = data_handler(Total_sensor_num=Total_sensor_num)
# #
# #     def connect_to_server(self):
# #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
# #             sock.connect((self.host, self.port))
# #             print(f"Connected to {self.host}:{self.port}")
# #
# #             # Send a request to send data
# #             sock.sendall(b"SEND\n")
# #
# #             data_processed = 0
# #             start_time = time.time()
# #             # Receive the data from the server
# #             while True:
# #                 data = sock.recv(1024)
# #                 if not data:
# #                     break
# #                 processed_data = parse_sensor_data(data.decode())
# #                 # print(data.decode())
# #                 if processed_data is None:
# #                     print(f"decode error: {data.decode()}")
# #                     continue
# #
# #                 self.data_handler.new_data_available(processed_data)
# #                 # print("Received:", processed_data)
# #                 data_processed += 1
# #
# #                 current_time = time.time()
# #                 elapsed_time = current_time - start_time
# #
# #                 if elapsed_time >= 1:
# #                     # print(f"Data processed per second: {data_processed//(Total_sensor_num * 2)}")
# #                     data_processed = 0
# #                     start_time = current_time
#
# if __name__ == "__main__":
#     server = SocketHandler('0.0.0.0', 9901)
#     server.start_server()
# # if __name__ == "__main__":
# #     # server_host = 'localhost'  # Server's IP address
# #     # server_port = 9901  # Port on which server is listening
# #     server_host = '0.0.0.0'  # Server's IP address
# #     server_port = 9901  # Port on which server is listening
# #     SH = Socket_Handler(server_host, server_port)
# #     SH.connect_to_server()
