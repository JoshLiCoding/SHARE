import sys
import trimesh
import numpy as np
import viser
import time
import open3d as o3d
import os

def rotate_points(points, x_angle=0, y_angle=0, z_angle=0):
    x_rad = np.deg2rad(x_angle)
    y_rad = np.deg2rad(y_angle)
    z_rad = np.deg2rad(z_angle)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_rad), -np.sin(x_rad)],
        [0, np.sin(x_rad),  np.cos(x_rad)]
    ])
    Ry = np.array([
        [np.cos(y_rad), 0, np.sin(y_rad)],
        [0, 1, 0],
        [-np.sin(y_rad), 0, np.cos(y_rad)]
    ])
    Rz = np.array([
        [np.cos(z_rad), -np.sin(z_rad), 0],
        [np.sin(z_rad),  np.cos(z_rad), 0],
        [0, 0, 1]
    ])
    rotation_matrix = Rz @ Ry @ Rx
    return points @ rotation_matrix.T

def visualize_scene_and_joints(skeleton):
    server = viser.ViserServer()
    scene_pc = trimesh.load('samples/body_scene_world.ply')
    scene_points = scene_pc.vertices
    scene_colors = scene_pc.visual.vertex_colors[:, :3]
    
    scene_pc2 = trimesh.load('share_scene.ply')
    scene_points2 = scene_pc2.vertices
    scene_colors2 = scene_pc2.visual.vertex_colors[:, :3]

    scene_pc3 = trimesh.load('aligned_pc2.ply')
    scene_points3 = scene_pc3.vertices
    scene_colors3 = scene_pc3.visual.vertex_colors[:, :3]

    mhm_scene = trimesh.load('mhm_scene.ply')

    subset_size = 1000000
    if len(scene_points) > subset_size:
        idx = np.random.choice(len(scene_points), subset_size, replace=False)
        scene_points = scene_points[idx]
        scene_colors = scene_colors[idx]
    if len(scene_points2) > subset_size:
        idx = np.random.choice(len(scene_points2), subset_size, replace=False)
        scene_points2 = scene_points2[idx]
        scene_colors2 = scene_colors2[idx]
    if len(scene_points3) > subset_size:
        idx = np.random.choice(len(scene_points3), subset_size, replace=False)
        scene_points3 = scene_points3[idx]
        scene_colors3 = scene_colors3[idx]


    T_opencv_to_viser = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0,  -1, 0],
    ])
    T_opengl_to_viser = np.array([
        [1,  0,  0],
        [0,  0,  -1],
        [0,  1, 0],
    ])
    scene_points = scene_points @ T_opencv_to_viser.T
    scene_points2 = scene_points2 @ T_opencv_to_viser.T
    scene_points3 = scene_points3 @ T_opencv_to_viser.T
    mhm_scene.vertices = mhm_scene.vertices @ T_opengl_to_viser.T
    server.add_point_cloud(
        points=scene_points,
        colors=scene_colors,
        point_size=0.01,
        name="gt"
    )
    server.add_point_cloud(
        points=scene_points2,
        colors=scene_colors2,
        point_size=0.01,
        name="aligned_1"
    )
    server.add_point_cloud(
        points=scene_points3,
        colors=scene_colors3,
        point_size=0.01,
        name="aligned2"
    )
    server.add_mesh_trimesh(
        mesh=mhm_scene,
        name="mhm_scene",
    )

    init_vert = np.load('init_pred_verts.npy')
    init_vert = init_vert @ T_opencv_to_viser.T
    init_faces = np.load('init_pred_faces.npy')
    for i in range(len(init_vert)):
        init_mesh = trimesh.Trimesh(vertices=init_vert[i], faces=init_faces)
        init_mesh.visual.vertex_colors = np.tile([0, 255, 0, 255], (init_vert[i].shape[0], 1))
        if i == 0 or i == 99:
            server.add_mesh_trimesh(
                mesh=init_mesh,
                name=f"init_pred_smpl_mesh_{i}",
            )

    vert = np.load('pred_verts.npy')
    vert = vert @ T_opencv_to_viser.T
    faces = np.load('pred_faces.npy')
    for i in range(len(vert)): 
        if i == 0 or i == 99:
            server.add_mesh_trimesh(
                mesh=trimesh.Trimesh(vertices=vert[i], faces=faces, visual=trimesh.visual.ColorVisuals(vertex_colors=np.tile([255, 0, 0, 255], (vert[i].shape[0], 1)))),
                name=f"pred_smpl_mesh_{i}",
            )
    vert2 = trimesh.load('samples/body_5.ply')
    vert2.vertices = vert2.vertices @ T_opencv_to_viser.T
    vert2.visual.vertex_colors = np.tile([255, 255, 255, 255], (vert2.vertices.shape[0], 1))  # RGBA, all red
    server.add_mesh_trimesh(
        mesh=vert2,
        name=f"gt_mesh_5",
    )
    vert3 = trimesh.load('samples/body_104.ply')
    vert3.vertices = vert3.vertices @ T_opencv_to_viser.T
    vert3.visual.vertex_colors = np.tile([255, 255, 255, 255], (vert3.vertices.shape[0], 1))  # RGBA, all red
    server.add_mesh_trimesh(
        mesh=vert3,
        name=f"gt_mesh_99",
    )

    mhm_vert = np.load('mhmocap_verts.npy')
    mhm_vert = mhm_vert @ T_opencv_to_viser.T
    mhm_faces = faces
    print(mhm_vert.shape, mhm_faces.shape)
    for i in range(len(mhm_vert)):
        if i == 0 or i == 99:
            mhm_mesh = trimesh.Trimesh(vertices=mhm_vert[i], faces=mhm_faces)
            mhm_mesh.visual.vertex_colors = np.tile([0, 0, 255, 255], (mhm_vert[i].shape[0], 1))  # RGBA, all red
            server.add_mesh_trimesh(
                mesh=mhm_mesh,
                name=f"mhm_{i}",
            )
    
    bev1 = np.load('0000__2_0.08.npz', allow_pickle=True)['results'].item()
    print(bev1['joints'].shape)
    bev_vert1 = bev1['verts'][0] + bev1['cam_trans'].squeeze()
    bev_vert1 = bev_vert1 @ T_opencv_to_viser.T
    bev_vert1_colors = np.tile([255, 255, 0, 255], (bev_vert1.shape[0], 1)) 
    server.add_mesh_trimesh(
        mesh=trimesh.Trimesh(vertices=bev_vert1, faces=faces, visual=trimesh.visual.ColorVisuals(vertex_colors=bev_vert1_colors)),
        name=f"bev_mesh_1",
    )
    bev2 = np.load('0099__2_0.08.npz', allow_pickle=True)['results'].item()
    bev_vert2 = bev2['verts'][0] + bev2['cam_trans'].squeeze()
    bev_vert2 = bev_vert2 @ T_opencv_to_viser.T
    bev_vert2_colors = np.tile([255, 255, 0, 255], (bev_vert2.shape[0], 1)) 
    server.add_mesh_trimesh(
        mesh=trimesh.Trimesh(vertices=bev_vert2, faces=faces, visual=trimesh.visual.ColorVisuals(vertex_colors=bev_vert2_colors)),
        name=f"bev_mesh_2",
    )

    # Add skeleton joints
    joints = np.load('pred_j3d.npy')[:, :, :]
    joints = joints @ T_opencv_to_viser.T
    server.add_point_cloud(
        points=joints.reshape(-1, 3),
        colors=(0, 255, 0),
        point_size=0.03,
        name="skeleton_joints"
    )

    # Add skeleton bones
    edges = []
    for i in range(len(joints)):
        for j, k in skeleton:
            edges.append([joints[i][j], joints[i][k]])
    edges = np.array(edges)
    server.add_line_segments(
        points=edges,
        colors=(0, 0, 255),
        name=f"skeleton_bones"
    )
    
    mhm_joints = np.load('mhmocap_joints.npy')
    mhm_joints = mhm_joints @ T_opencv_to_viser.T
    server.add_point_cloud(
        points=mhm_joints.reshape(-1, 3),
        colors=(255, 0, 0),
        point_size=0.03,
        name="mhm_skeleton_joints"
    )
    mhm_edges = []
    for i in range(len(mhm_joints)):
        for j, k in skeleton:
            mhm_edges.append([mhm_joints[i][j], mhm_joints[i][k]])
    mhm_edges = np.array(mhm_edges)
    server.add_line_segments(
        points=mhm_edges,
        colors=(255, 0, 0),
        name=f"mhm_skeleton_bones"
    )


    while True:
        time.sleep(1)

if __name__ == "__main__":
    kintree_path = "body_models/smpl/kintree_table.pkl"
    kintree = np.load(kintree_path, allow_pickle=True)
    skeleton = kintree.transpose((1, 0))[1:-2]

    visualize_scene_and_joints(skeleton)