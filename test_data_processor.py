import pdb
import pandas as pd
import numpy as np
import time
import torch

def return_data():
    # Replace 'your_file.csv' with the path to your CSV file
    # file_path = "data/test_data/trail_9_RoNIN_input.csv"
    file_path = "data/test_data/trail_12_RoNIN_input.csv"

    # Define column names based on the structure you provided
    column_names = ['gyro_x', 'gyro_y', 'gyro_z',
                    'acce_x', 'acce_y', 'acce_z',
                    'm11', 'm12', 'm13',
                    'm21', 'm22', 'm23',
                    'm31', 'm32', 'm33',
                    'raw_azimuth', 'timestamp']

    # Read the CSV file without headers and assign column names
    df = pd.read_csv(file_path, header=None, names=column_names)

    # Extract acceleration data into a list of [n, 3]
    acceleration = df[['acce_x', 'acce_y', 'acce_z']].values * 9.8
    # acceleration[:, 1], acceleration[:, 2] = acceleration[:, 2], acceleration[:, 1]
    # acceleration[:, 1] = -acceleration[:, 1] # Reverse y here...?
    # Extract rotation matrix into a list of [n, 3, 3]
    # First, extract the rotation matrix data into a 2D array
    rotation_matrix_data = df[['m11', 'm12', 'm13', 'm21', 'm22', 'm23', 'm31', 'm32', 'm33']].values

    # Now, reshape this array into the desired format [n, 3, 3]
    rotation_matrix = rotation_matrix_data.reshape(-1, 3, 3)
    # rotation_matrix[:, 0, 2] *= -1
    # rotation_matrix[:, 1, 2] *= -1
    # rotation_matrix[:, 2, 0] *= -1
    # rotation_matrix[:, 2, 1] *= -1
    # Step 1: Define the flipping matrix F
    # F = np.array([[1, 0, 0],
    #               [0, 0, 1],
    #               [0, 1, 0]])
    # Step 2: Perform the FR operation for each rotation matrix R in the array
    # rotation_matrix = np.dot(rotation_matrix,F)
    acceleration_rotated = np.einsum('ijk,ik->ij', rotation_matrix, acceleration) + [0,0,9.8]
    # pdb.set_trace()
    rotation_matrix = torch.tensor(rotation_matrix)
    acceleration_rotated = torch.tensor(acceleration_rotated)
    return acceleration_rotated, rotation_matrix


def tpose_calibration(rotation_matrix):
    RSI = rotation_matrix[0].view(3, 3).t()
    # RMI = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],dtype=torch.float64).mm(RSI)
    RMI = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64).mm(RSI)
    print(RMI)
    RIS = rotation_matrix[1]
    RSB = RMI.matmul(RIS).t()
    return RMI, RSB

def combine_all():
    aI, RIS = return_data()
    RMI, RSB = tpose_calibration(RIS)
    RMB = RMI.matmul(RIS).matmul(RSB)
    aM = aI.mm(RMI.t())

    sequence_length = aM.shape[0]
    # Create a tensor of zeros with the desired shape [950, 6, 3]
    zeros_aM = torch.zeros(sequence_length, 6, 3)

    # Place the values of aM at the specified index (1) along the second dimension
    zeros_aM[:, 1, :] = aM

    # Create a tensor filled with identity matrices
    eye_RMB = torch.eye(3).repeat(sequence_length, 6, 1, 1)

    # Replace the matrices at index 1 with RMB
    eye_RMB[:, 1, :, :] = RMB

    # Create an identity matrix of size [3, 3]
    eye_matrix = torch.eye(3)

    # Repeat this matrix 24 times and reshape to get the desired shape [1, 24, 3, 3]
    init_pose = eye_matrix.repeat(24, 1).view(1, 24, 3, 3)

    return zeros_aM, eye_RMB, init_pose

if __name__ == '__main__':
    zeros_aM, eye_RMB, init_pose = combine_all()