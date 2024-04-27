import pdb
import pandas as pd
import numpy as np
import time
import torch
import pickle
import pdb
from scipy.spatial.transform import Rotation
from test_data_processor import combine_all_offline
from Global_Constants import Total_sensor_num

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def q_to_rot(quaternion):
    # 使用scipy库中的Rotation类来处理四元数
    w,x,y,z = quaternion
    r = Rotation.from_quat([x,y,z,w])

    # 将四元数转换为旋转矩阵
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def process_acceleration_and_rotation_raw_data(acceleration_data, quaternion):
    rotation_matrix = []
    for single_quaternion in quaternion:
        rotation_matrix.append(q_to_rot(single_quaternion))
    rotation_matrix = torch.tensor(rotation_matrix)
    rotation_matrix = rotation_matrix.view(-1, 3, 3)

    acceleration_rotated = torch.tensor(acceleration_data, dtype=torch.float64)
    # acceleration_rotated = acceleration_rotated.view(-1, 3)
    acceleration_rotated = np.einsum('ijk,ik->ij', rotation_matrix, acceleration_rotated)
    acceleration_rotated = torch.tensor(acceleration_rotated, dtype=torch.float64)

    return acceleration_rotated, rotation_matrix


# def return_data():
#     root_path = "/Users/pren1/PycharmProjects/Socket_handler/data/"
#     acceleration_path = root_path + "total_acceleration_3000.pkl"
#     rotation_path = root_path + "total_orientation_3000.pkl"
#
#     acceleration_data = load_pickle(acceleration_path)
#     rotation_data = load_pickle(rotation_path)
#
#     test = process_acceleration_and_rotation_raw_data(acceleration_data[0], rotation_data[0])
#
#     rotation_matrix = []
#     for quat in rotation_data:
#         rotation_matrix.append(q_to_rot(quat[0]))
#     rotation_matrix = torch.tensor(rotation_matrix)
#     acceleration_rotated = torch.tensor(acceleration_data, dtype=torch.float64)
#     acceleration_rotated = acceleration_rotated.view(-1, 3)
#     acceleration_rotated = np.einsum('ijk,ik->ij', rotation_matrix, acceleration_rotated)
#     acceleration_rotated = torch.tensor(acceleration_rotated, dtype=torch.float64)
#     return acceleration_rotated, rotation_matrix

class OnlineProcess:
    def __init__(self):
        self.is_calibrated = False
        self.RMI = None
        self.RSB = None
        self.init_pose = self.create_init_pose()

    def create_init_pose(self):
        # Create an identity matrix of size [3, 3]
        eye_matrix = torch.eye(3)

        # Repeat this matrix 24 times and reshape to get the desired shape [1, 24, 3, 3]
        return eye_matrix.repeat(24, 1).view(1, 24, 3, 3)

    def tpose_calibration(self, rotation_matrix, target_index):
        RSI = rotation_matrix[target_index, :, :].view(3, 3).t()
        'This is suitable for our own sensor!'
        RMI = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=torch.float64).mm(RSI)
        # RMI = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64).mm(RSI)
        print(RMI)
        'Notice that here you should input the whole rotation matrix of all sensors'
        RIS = rotation_matrix
        RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3, dtype=torch.float64))  # = (R_MI R_IS)^T R_MB = R_SB
        # RSB = RMI.matmul(RIS).t()
        return RMI, RSB

    def new_data_available(self, acceleration_rotated, quaternion, index_pack):
        aI, RIS = process_acceleration_and_rotation_raw_data(acceleration_rotated, quaternion)
        assert aI.shape == (Total_sensor_num, 3), "Fatal error, check your acceleration"
        assert RIS.shape == (Total_sensor_num, 3, 3), "Fatal error, check your rotation matrix"
        assert len(index_pack) == Total_sensor_num, "Fatal error, check your index pack"
        if not self.is_calibrated:
            print("Doing Calibration")
            self.is_calibrated = True
            'Notice that here the rotation matrix should from the left hand sensor'
            target_index = index_pack.index(0)
            # target_index = 0
            self.RMI, self.RSB = self.tpose_calibration(RIS, target_index)
        RMB = self.RMI.matmul(RIS).matmul(self.RSB)
        aM = aI.mm(self.RMI.t())

        # Create a tensor of zeros with the desired shape [950, 6, 3]
        zeros_aM = torch.zeros(1, 6, 3)

        # Create a tensor filled with identity matrices
        eye_RMB = torch.eye(3).repeat(1, 6, 1, 1)

        'Important: Here we make sure each sensor is assigned to a correct place'
        'For example the sensor AVA will be assigned to the head position since its POI_index is 4'
        for index, POI_index in enumerate(index_pack):
            # Place the values of aM at the specified index along the second dimension
            zeros_aM[:, POI_index, :] = aM[index, :]
            eye_RMB[:, POI_index, :, :] = RMB[index, :, :]

        # POI_index = 0
        # # Place the values of aM at the specified index (1) along the second dimension
        # zeros_aM[:, POI_index, :] = aM
        #
        # # Replace the matrices at index 1 with RMB
        # eye_RMB[:, POI_index, :, :] = RMB

        return zeros_aM, eye_RMB


def combine_all():
    root_path = "/Users/pren1/PycharmProjects/Socket_handler/data/"
    acceleration_path = root_path + "total_acceleration_3000.pkl"
    rotation_path = root_path + "total_orientation_3000.pkl"

    acceleration_data = load_pickle(acceleration_path)
    rotation_data = load_pickle(rotation_path)

    OP = OnlineProcess()
    final_zeros_aM = []
    final_eye_RMB = []
    for i in range(len(acceleration_data)):
        cur_acceleration = acceleration_data[i]
        cur_rotation = rotation_data[i]

        zeros_aM, eye_RMB = OP.new_data_available(cur_acceleration, cur_rotation)
        final_zeros_aM.append(zeros_aM)
        final_eye_RMB.append(eye_RMB)

    final_zeros_aM = torch.cat(final_zeros_aM, dim=0)
    final_eye_RMB = torch.cat(final_eye_RMB, dim=0)
    return final_zeros_aM, final_eye_RMB, OP.init_pose

    # aI, RIS = return_data()
    # RMI, RSB = tpose_calibration(RIS)
    # RMB = RMI.matmul(RIS).matmul(RSB)
    # aM = aI.mm(RMI.t())
    #
    # sequence_length = aM.shape[0]
    # # Create a tensor of zeros with the desired shape [950, 6, 3]
    # zeros_aM = torch.zeros(sequence_length, 6, 3)
    #
    # POI_index = 0
    # # Place the values of aM at the specified index (1) along the second dimension
    # zeros_aM[:, POI_index, :] = aM
    #
    # # Create a tensor filled with identity matrices
    # eye_RMB = torch.eye(3).repeat(sequence_length, 6, 1, 1)
    #
    # # Replace the matrices at index 1 with RMB
    # eye_RMB[:, POI_index, :, :] = RMB
    #
    # # Create an identity matrix of size [3, 3]
    # eye_matrix = torch.eye(3)
    #
    # # Repeat this matrix 24 times and reshape to get the desired shape [1, 24, 3, 3]
    # init_pose = eye_matrix.repeat(24, 1).view(1, 24, 3, 3)

    # return zeros_aM, eye_RMB, init_pose

if __name__ == '__main__':
    zeros_aM, eye_RMB, init_pose = combine_all_offline()
    final_zeros_aM, final_eye_RMB, second_init_pose = combine_all()

    print(torch.equal(zeros_aM, final_zeros_aM))  # Output: True
    print(torch.equal(eye_RMB, final_eye_RMB))  # Output: True
    print(torch.equal(init_pose, second_init_pose))  # Output: True

    # pdb.set_trace()