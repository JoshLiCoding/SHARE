import sys
import trimesh
import numpy as np
import viser
import time
import open3d as o3d
import os

def visualize_scene_and_joints(folder):
    server = viser.ViserServer()
    scene_pc = trimesh.load(os.path.join(folder, 'share_scene.ply'))
    scene_points = scene_pc.vertices
    scene_colors = scene_pc.visual.vertex_colors[:, :3]

    scene_pc_1 = trimesh.load(os.path.join(folder, 'share_scene1.ply'))
    scene_pc_2 = trimesh.load(os.path.join(folder, 'share_scene2.ply'))
    scene_points_1 = scene_pc_1.vertices
    scene_points_2 = scene_pc_2.vertices
    scene_colors_1 = scene_pc_1.visual.vertex_colors[:, :3]
    scene_colors_2 = scene_pc_2.visual.vertex_colors[:, :3]

    subset_size = int(scene_points.shape[0] * 3 / 4)
    if len(scene_points) > subset_size:
        idx = np.random.choice(len(scene_points), subset_size, replace=False)
        scene_points = scene_points[idx]
        scene_colors = scene_colors[idx]
    if len(scene_points_1) > subset_size:
        idx = np.random.choice(len(scene_points_1), subset_size, replace=False)
        scene_points_1 = scene_points_1[idx]
        scene_colors_1 = scene_colors_1[idx]
    if len(scene_points_2) > subset_size:
        idx = np.random.choice(len(scene_points_2), subset_size, replace=False)
        scene_points_2 = scene_points_2[idx]
        scene_colors_2 = scene_colors_2[idx]

    T_opencv_to_viser = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0,  -1, 0],
    ])

    scene_points = scene_points @ T_opencv_to_viser.T
    server.add_point_cloud(
        points=scene_points,
        colors=scene_colors,
        point_size=0.015,
        name="share_scene"
    )

    scene_points_1 = scene_points_1 @ T_opencv_to_viser.T
    scene_points_2 = scene_points_2 @ T_opencv_to_viser.T
    server.add_point_cloud(
        points=scene_points_1,
        colors=scene_colors_1,
        point_size=0.015,
        name="share_scene1"
    )
    server.add_point_cloud(
        points=scene_points_2,
        colors=scene_colors_2,
        point_size=0.015,
        name="share_scene2"
    )

    scene_human1 = trimesh.load(os.path.join(folder, 'green_human1.ply'))
    scene_human2 = trimesh.load(os.path.join(folder, 'green_human2.ply'))
    human_points_1 = scene_human1.vertices
    human_points_2 = scene_human2.vertices
    human_colors_1 = scene_human1.visual.vertex_colors[:, :3]
    human_colors_2 = scene_human2.visual.vertex_colors[:, :3]
    human_points_1 = human_points_1 @ T_opencv_to_viser.T
    human_points_2 = human_points_2 @ T_opencv_to_viser.T
    
    server.add_point_cloud(
        points=human_points_1,
        colors=human_colors_1,
        point_size=0.015,
        name="green_human1"
    )
    server.add_point_cloud(
        points=human_points_2,
        colors=human_colors_2,
        point_size=0.015,
        name="green_human2"
    )
    
    faces = np.load('vis/faces.npy')

    share_vert = np.load(os.path.join(folder, 'share_vertices.npy'))
    share_vert = share_vert @ T_opencv_to_viser.T
    for i in range(len(share_vert)): 
        if i == 0 or i == 97 or i == len(share_vert) - 1:
            server.add_mesh_trimesh(
                mesh=trimesh.Trimesh(vertices=share_vert[i], faces=faces, visual=trimesh.visual.ColorVisuals(vertex_colors=np.tile([59, 255, 65, 255], (share_vert[i].shape[0], 1)))),
                name=f"share_vertices_{i}",
            )
    
    init_share_vert = np.load(os.path.join(folder, 'init_share_vertices.npy'))
    init_share_vert = init_share_vert @ T_opencv_to_viser.T
    for i in range(len(init_share_vert)): 
        if i == 0 or i == 97 or i == len(init_share_vert) - 1:
            server.add_mesh_trimesh(
                mesh=trimesh.Trimesh(vertices=init_share_vert[i], faces=faces, visual=trimesh.visual.ColorVisuals(vertex_colors=np.tile([255, 169, 77, 255], (init_share_vert[i].shape[0], 1)))),
                name=f"init_share_vertices_{i}",
            )
    

    while True:
        time.sleep(1)

# https://www.youtube.com/watch?v=likDxPBn0Ug
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder")
    args = parser.parse_args()

    visualize_scene_and_joints(args.folder)