import socket
import pickle
import time

class ClientHandler:
    def __init__(self, server_ip: str, port: int):
        self.server_ip = server_ip  # IP address of the server to connect to
        self.port = port  # Port on which the server is listening

    def connect_to_server(self):
        # Establish connection to the server
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.port))
        print(f"Connected to server at {self.server_ip}:{self.port}")

    def receive_data_continuously(self):
        while True:
            try:
                self.receive_data(self.client_socket)
                # # Attempt to receive data from the server
                # data = self.client_socket.recv(4096)  # Adjust buffer size as necessary
                # if not data:
                #     print("No more data received from server. Connection may be closed.")
                #     break  # No more data from the server, likely the connection is closed
                # print(data.decode())
                # # Deserialize the data
                # # data = pickle.loads(data)
                # # print("Received data:", data)
            except Exception as e:
                print(e)

    def receive_data(self, server_socket):
        # Receive the size of the incoming data
        # data_size = 4096

        # Receive the actual data
        data_bytes = server_socket.recv(4096)

        # Decode bytes to string
        data_string = data_bytes.decode()
        print(data_string)

        # # Split the data string back into pose and trans components
        # pose_string, trans_string = data_string.split('|')
        #
        # # Convert string back to float lists
        # pose_list = list(map(float, pose_string.split()))
        # trans_list = list(map(float, trans_string.split()))
        #
        # print("Received pose:", pose_list)
        # print("Received trans:", trans_list)

# Usage
if __name__ == "__main__":
    client = ClientHandler('8.138.83.250', 9902)  # Replace 'server_ip_here' with the actual server IP
    client.connect_to_server()
    client.receive_data_continuously()