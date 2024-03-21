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
    # # Setup the Open3D visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # for i in range(len(pose_t_all)):
    #     pose = pose_t_all[i]  # First frame pose
    #     trans = tran_t_all[i]  # First frame translation
    #     output = smpl(body_pose=pose[3:].unsqueeze(0), global_orient=pose[:3].unsqueeze(0), transl=trans.unsqueeze(0))
    #     # Get the output of the SMPL model
    #     # output = smpl(body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)
    #     # Get vertices and faces from the model output
    #     vertices = output.vertices.detach().cpu().numpy().squeeze()
    #     faces = smpl.faces
    #
    #     # Create an Open3D mesh
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #     mesh.triangles = o3d.utility.Vector3iVector(faces)
    #
    #     # Compute vertex normals for better lighting
    #     mesh.compute_vertex_normals()
    #
    #     # Visualize the mesh
    #     o3d.visualization.draw_geometries([mesh])


# Assuming your pose_t_all and tran_t_all are loaded as NumPy arrays
_, _, pose_t_all, tran_t_all = torch.load(os.path.join(paths.dipimu_dir, 'test.pt')).values()
# Example usage
# Make sure to replace 'smpl_model_path' with the path to your SMPL model file
# and ensure pose_t_all and tran_t_all are loaded tensors with your pose and translation data
visualize_smpl_with_tensors(smpl_model_path, pose_t_all[0], tran_t_all[0])

# def visualize_smpl_motion(pose_t_all, tran_t_all):
#     # Convert tensors to numpy arrays for compatibility with PyBullet
#     pose_t_all = pose_t_all[-1].cpu().numpy()
#     tran_t_all = tran_t_all[-1].cpu().numpy()
#     # Assuming PyBullet environment setup is similar
#     physicsClient = p.connect(p.GUI)  # Start PyBullet in GUI mode
#     p.setGravity(0, 0, -10)
#
#     # Load the plane URDF to act as the ground
#     plane_file = paths.plane_file
#     planeId = p.loadURDF(plane_file)
#
#     # Load the physics model URDF representing the SMPL model
#     physics_model_file = paths.physics_model_file
#     physics_model_id = p.loadURDF(physics_model_file, [0, 0, 1], useFixedBase=1)
#
#     num_frames = pose_t_all.shape[0]
#     num_joints = int(pose_t_all.shape[1] / 3)  # Assuming 72 parameters means 24 joints with 3 parameters each
#
#     for frame_idx in range(num_frames):
#         # Apply global translation for the current frame
#         translation = tran_t_all[frame_idx]
#         p.resetBasePositionAndOrientation(physics_model_id, posObj=translation, ornObj=[0, 0, 0, 1])
#
#         # Extract pose for the current frame
#         pose = pose_t_all[frame_idx].reshape((num_joints, 3))
#
#         # Iterate through each joint and apply the pose parameters
#         # NOTE: This assumes the URDF model joints correspond directly to the SMPL model's joints
#         for joint_idx in range(num_joints):
#             # Example of setting a joint target position, assuming the first parameter is sufficient for demonstration
#             # For a real application, you'd need to map these parameters to your URDF model's joint specifications
#             p.resetJointState(physics_model_id, joint_idx, targetValue=pose[joint_idx][0])
#
#         # Step the simulation and wait a bit to visualize the frame
#         p.stepSimulation()
#         time.sleep(1.0 / 60)  # Visualize at 60 FPS
#         # pdb.set_trace()
#
#     input("Press Enter to end the simulation...")
#     p.disconnect()
#
# # Assuming your pose_t_all and tran_t_all are loaded as NumPy arrays
# _, _, pose_t_all, tran_t_all = torch.load(os.path.join(paths.dipimu_dir, 'test.pt')).values()
# # Example usage
# visualize_smpl_motion(pose_t_all, tran_t_all)