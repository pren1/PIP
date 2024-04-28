import pdb
import socket
from util import *
from Global_Constants import *
from data_handler import data_handler
import time

class Socket_Handler(object):
    def __init__(self, host: str, port: int):
        self.host = host  # Server's IP address
        self.port = port  # Port on which server is listening

        self.data_handler = data_handler(Total_sensor_num=Total_sensor_num)

    def connect_to_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")

            # Send a request to send data
            sock.sendall(b"SEND\n")

            data_processed = 0
            start_time = time.time()
            # Receive the data from the server
            while True:
                data = sock.recv(1024)
                if not data:
                    break
                processed_data = parse_sensor_data(data.decode())
                # print(data.decode())
                if processed_data is None:
                    print(f"decode error: {data.decode()}")
                    continue

                self.data_handler.new_data_available(processed_data)
                # print("Received:", processed_data)
                data_processed += 1

                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time >= 1:
                    # print(f"Data processed per second: {data_processed//(Total_sensor_num * 2)}")
                    data_processed = 0
                    start_time = current_time

if __name__ == "__main__":
    # server_host = 'localhost'  # Server's IP address
    # server_port = 9901  # Port on which server is listening
    server_host = '0.0.0.0'  # Server's IP address
    server_port = 9901  # Port on which server is listening
    SH = Socket_Handler(server_host, server_port)
    SH.connect_to_server()
