import pdb
from util import *
import pickle
import articulate as art
from articulate.utils.rbdl import *
from net import PIP
import pdb
# from test_data_processor import combine_all
from Online_Process import *
from SMPLVisualizer import SMPLVisualizer

class data_handler(object):
    def __init__(self, Total_sensor_num):
        'We multiply by 2 here because we want to wait until acceleration data and orientation data all available'
        self.size = Total_sensor_num * 2
        self.sensor_stack = {}
        self.address_dict = {} # Map address to a number
        self.sensor_id_counter = 0
        self.new_data_register = [False for i in range(self.size)]

        self.save_counter = 0
        self.total_acceleration = []
        self.total_orientation = []

        self.OP = OnlineProcess()
        self.net = PIP()
        self.SV = SMPLVisualizer()

    def new_data_available(self, input_data: SensorData):
        udp_address = input_data.udp_address
        data = input_data.data
        type = input_data.type

        # timestamp = input_data.timestamp
        if udp_address not in self.sensor_stack:
            self.sensor_stack[udp_address] = {
                'Rotation': [],
                'Acceleration': []
            }
            if len(self.sensor_stack) == self.size//2:
                print("All sensors connected")
                # pdb.set_trace()

        if f"{udp_address}_{type}" not in self.address_dict:
            # Just assign an address a number
            self.address_dict[f"{udp_address}_{type}"] = self.sensor_id_counter
            assert self.sensor_id_counter <= self.size
            self.sensor_id_counter += 1

        self.sensor_stack[udp_address][type].append(data)
        self.new_data_register[self.address_dict[f"{udp_address}_{type}"]] = True
        if sum(self.new_data_register) == self.size:
            # Time to ship your data!
            self.pack_data()
            'Set all bits to False'
            self.new_data_register = [False for i in range(self.size)]

    def pack_data(self):
        acceleration_pack = []
        rotation_matrix_pack = []

        for key in self.sensor_stack:
            acceleration_pack.append(self.sensor_stack[key]['Acceleration'][-1])
            rotation_matrix_pack.append(self.sensor_stack[key]['Rotation'][-1])

        Sign = 'Pos'
        if acceleration_pack[-1][2] < 0:
            Sign = 'Neg'
        print(f"\r {Sign}:", round(acceleration_pack[-1][2]), end="", flush=True)
        self.total_acceleration.append(acceleration_pack)
        self.total_orientation.append(rotation_matrix_pack)

        self.save_counter += 1
        if self.save_counter % 3 == 0:
            zeros_aM, eye_RMB = self.OP.new_data_available(acceleration_pack, rotation_matrix_pack)
            pose, trans =self. net.new_data_available(zeros_aM, eye_RMB, self.OP.init_pose)
            pose = art.math.rotation_matrix_to_axis_angle(pose).view(-1, 72)
            self.SV.visualize_smpl_with_tensors(pose, trans)

        # 'Save the data after every 10 seconds...'
        # if self.save_counter % 1000 == 0:
        #     self.save_data(self.save_counter)

        # We are going to call LSTM here...
        # print(f"Acceleration: {acceleration_pack}")
        # print(f"Rotation: {rotation_matrix_pack}")
        # pdb.set_trace()

    def save_data(self, counter):
        # 将self.total_acceleration存储为pkl文件
        self.save_data_to_pkl(self.total_acceleration, f'data/total_acceleration_{counter}.pkl')
        # 将self.total_orientation存储为pkl文件
        self.save_data_to_pkl(self.total_orientation, f'data/total_orientation_{counter}.pkl')

    def save_data_to_pkl(self, data, filename):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
