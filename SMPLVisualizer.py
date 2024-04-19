import os
import torch
from config import *
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import smplx
import time

class SMPLVisualizer():
    def __init__(self):
        # Assuming you have the SMPL model file path
        self.smpl_model_path = "models/"
        self.is_initialized = False
        self.smpl = None
        self.vis = None
        self.mesh = None

    def visualize_smpl_with_tensors(self, pose, trans):
        if not self.is_initialized:
            self.is_initialized = True
            # Initialize the SMPL model
            self.smpl = smplx.create(self.smpl_model_path, model_type='smpl', gender='male')

            # Setup the Open3D visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

            # Initial mesh setup
            output = self.smpl(body_pose=pose[3:].unsqueeze(0), global_orient=pose[:3].unsqueeze(0), transl=trans.unsqueeze(0))
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            faces = self.smpl.faces

            self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(faces)
            self.mesh.compute_vertex_normals()

            # Add the mesh to the visualizer
            self.vis.add_geometry(self.mesh)

        output = self.smpl(body_pose=pose[3:].unsqueeze(0), global_orient=pose[:3].unsqueeze(0), transl=trans.unsqueeze(0))
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        # Update the mesh vertices
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.compute_vertex_normals()

        # Update the visualizer
        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

        # You can add a small delay here if the animation is too fast
        # time.sleep(0.1)

    def post_vis_act(self):
        self.vis.run()  # Keep the window open until manually closed
        self.vis.destroy_window()
# Assuming your pose_t_all and tran_t_all are loaded as NumPy arrays
# _, _, pose_t_all_g, tran_t_all_g = torch.load(os.path.join(paths.dipimu_dir, 'test_case_4.pt')).values()

# SV = SMPLVisualizer()
# for i in range(len(pose_t_all_g[0])):
#     pose = pose_t_all_g[0][i]
#     trans = tran_t_all_g[0][i]
#     SV.visualize_smpl_with_tensors(pose, trans)
#
# SV.post_vis_act()


# _, _, pose_t_all_g, tran_t_all_g = torch.load(os.path.join(paths.dipimu_dir, 'test.pt')).values()
# _, _, pose_t_all, tran_t_all = torch.load(os.path.join(paths.dipimu_dir, 'test_p.pt')).values()
# Example usage
# Make sure to replace 'smpl_model_path' with the path to your SMPL model file
# and ensure pose_t_all and tran_t_all are loaded tensors with your pose and translation data
# visualize_smpl_with_tensors(smpl_model_path, pose_t_all_g[0], tran_t_all_g[0])