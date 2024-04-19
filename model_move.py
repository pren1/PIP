import os
import torch
from config import *
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import smplx
import time

# Assuming you have the SMPL model file path
smpl_model_path = "models/"

def visualize_smpl_with_tensors(smpl_model_path, pose_t_all, tran_t_all):
    # Initialize the SMPL model
    smpl = smplx.create(smpl_model_path, model_type='smpl', gender='male')

    # Setup the Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Initial mesh setup
    pose = pose_t_all[0]  # Initial pose
    trans = tran_t_all[0]  # Initial translation
    output = smpl(body_pose=pose[3:].unsqueeze(0), global_orient=pose[:3].unsqueeze(0), transl=trans.unsqueeze(0))
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = smpl.faces

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    for i in range(1, len(pose_t_all)):

        # Update pose and translation
        pose = pose_t_all[i]
        trans = tran_t_all[i]
        output = smpl(body_pose=pose[3:].unsqueeze(0), global_orient=pose[:3].unsqueeze(0), transl=trans.unsqueeze(0))
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        # Update the mesh vertices
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()

        # Update the visualizer
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # You can add a small delay here if the animation is too fast
        # time.sleep(0.1)

    vis.run()  # Keep the window open until manually closed
    vis.destroy_window()
# Assuming your pose_t_all and tran_t_all are loaded as NumPy arrays
_, _, pose_t_all_g, tran_t_all_g = torch.load(os.path.join(paths.dipimu_dir, 'test_case_4.pt')).values()
# _, _, pose_t_all_g, tran_t_all_g = torch.load(os.path.join(paths.dipimu_dir, 'test.pt')).values()
# _, _, pose_t_all, tran_t_all = torch.load(os.path.join(paths.dipimu_dir, 'test_p.pt')).values()
# Example usage
# Make sure to replace 'smpl_model_path' with the path to your SMPL model file
# and ensure pose_t_all and tran_t_all are loaded tensors with your pose and translation data
visualize_smpl_with_tensors(smpl_model_path, pose_t_all_g[0], tran_t_all_g[0])