import torch
import os
from config import *
import pdb

data_dir = paths.totalcapture_dir
data_name = os.path.basename(data_dir)
result_dir = os.path.join(paths.result_dir, data_name, "Fake")
# First element: acc, shape: N * 6 * 3
# Second element: ori, shape: N * 6 * 3 * 3
# Third element: pose, shape: N * 24 * 3
# Fourth element: tran, shape: N * 3
# Fifth element: joint, shape: N * 24 * 3

# Difference between pose and joint
# Pose is angle
# Joint is position
# _, _, pose_t_all, tran_t_all = torch.load(os.path.join(data_dir, 'test.pt')).values()
_, _, pose_t_all, tran_t_all, _ = torch.load(os.path.join(data_dir, 'test.pt')).values()
reshaped_list = [tensor.view(-1, 72) for tensor in pose_t_all]
pdb.set_trace()