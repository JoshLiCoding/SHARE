
import sys
import os
import numpy as np
import trimesh
import viser
import time
import random

T_opencv_to_viser = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0,  -1, 0],
])
T_opengl_to_opencv = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
])
subset_size = 1000000

def even_split_indices(num_splits, max_index=99):
    return np.round(np.linspace(0, max_index, num_splits)).astype(int)
num_splits = 3
indices = even_split_indices(num_splits)

# python vis/vis_batch.py Gym_010_lunge1_batch01 --mhmocap --bev --tram --share
# python vis/vis_batch.py ParkingLot2_009_impro1_batch07 --mhmocap --bev --tram --share
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize data.")
    parser.add_argument("folder_name", type=str, help="Name of the folder to visualize")
    parser.add_argument("--mhmocap", action="store_true", help="Load and visualize mhmocap scene, joints and verts")
    parser.add_argument("--bev", action="store_true", help="Load and visualize BEV joints and verts")
    parser.add_argument("--tram", action="store_true", help="Load and visualize tram joints and verts")
    parser.add_argument("--share", action="store_true", help="Load and visualize SHARE joints and verts")
    args = parser.parse_args()

    folder_name = args.folder_name
    use_mhmocap = args.mhmocap
    use_bev = args.bev
    use_tram = args.tram
    use_share = args.share

    batched_dir = os.path.join('eval/RICH/batched_smpl_motions', folder_name)
    verts_path = os.path.join(batched_dir, 'vertices.npy')
    joints_path = os.path.join(batched_dir, 'joints.npy')
    faces_path = 'vis/faces.npy'  # Use existing faces file in current dir

    if not (os.path.exists(verts_path) and os.path.exists(joints_path) and os.path.exists(faces_path)):
        print("Missing required npy files.")
        sys.exit(1)

    verts = np.load(verts_path)
    verts = verts @ T_opencv_to_viser.T
    joints = np.load(joints_path)
    joints = joints @ T_opencv_to_viser.T
    faces = np.load(faces_path)

    # Get scan name from folder_name (first word before '_')
    scan_name = folder_name.split('_')[0]
    if scan_name == 'LectureHall':
        scan_ply_path = os.path.join('../rich_toolkit/data/scan_calibration', scan_name, 'scan_yoga_scene_camcoord.ply')
    else:
        scan_ply_path = os.path.join('../rich_toolkit/data/scan_calibration', scan_name, 'scan_camcoord.ply')
    if not os.path.exists(scan_ply_path):
        print(f"Missing scan_camcoord.ply: {scan_ply_path}")
        sys.exit(1)
    scan_pc = trimesh.load(scan_ply_path)
    
    if len(scan_pc.vertices) > subset_size:
        idx = random.sample(range(len(scan_pc.vertices)), subset_size)
        scan_pc.vertices = scan_pc.vertices[idx]
        scan_pc.visual.vertex_colors = scan_pc.visual.vertex_colors[idx]

    scan_pc.vertices = scan_pc.vertices @ T_opencv_to_viser.T

    # Load kintree for skeleton
    kintree_path = "body_models/smpl/kintree_table.pkl"
    kintree = np.load(kintree_path, allow_pickle=True)
    skeleton = kintree.transpose((1, 0))[1:]

    # Visualize
    server = viser.ViserServer()
    # Add scan point cloud
    server.add_point_cloud(
        points=scan_pc.vertices,
        colors=scan_pc.visual.vertex_colors[:, :3],
        point_size=0.015,
        name="scan_camcoord"
    )
    # Add SMPL meshes
    for i in indices:
        mesh = trimesh.Trimesh(vertices=verts[i], faces=faces)
        mesh.visual.vertex_colors = np.tile([255, 255, 255, 255], (verts[i].shape[0], 1))
        server.add_mesh_trimesh(
            mesh=mesh,
            name=f"gt_mesh_{i}",
        )
    # Add joints
    # for i in indices:
    #     server.add_point_cloud(
    #         points=joints[i],
    #         colors=(0, 255, 0),
    #         point_size=0.03,
    #         name=f"gt_joints_{i}"
    #     )
    #     # Add bones
    #     edges = []
    #     for j, k in skeleton:
    #         edges.append([joints[i][j], joints[i][k]])
    #     edges = np.array(edges)
    #     server.add_line_segments(
    #         points=edges,
    #         colors=(0, 0, 255),
    #         name=f"gt_bones_{i}"
    #     )

    # Find corresponding cam# folder using camera number from sequence_camera_numbers.json
    import glob, json, xml.etree.ElementTree as ET
    base = folder_name.split('_batch')[0]
    batch = folder_name.split('_batch')[-1]
    cam_json_path = os.path.join('eval/sequence_camera_numbers.json')
    with open(cam_json_path, 'r') as f:
        cam_map = json.load(f)
    cam_num = cam_map.get(base)
    scan_root = '../rich_toolkit/data/scan_calibration'
    scan_name = base.split('_')[0]
    xml_path = os.path.join(scan_root, scan_name, 'calibration', f'{cam_num:03d}.xml')
    if not os.path.exists(xml_path):
        print(f"Missing calibration XML: {xml_path}")
        return
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cam_matrix_elem = root.find('CameraMatrix')
    cam_data = cam_matrix_elem.find('data').text.strip().split()
    cam_matrix = np.array([float(x) for x in cam_data]).reshape(3, 4)
    R = cam_matrix[:, :3]
    t = cam_matrix[:, 3]
    
    C = -np.linalg.inv(R) @ t
    C_vis = C @ T_opencv_to_viser.T
    server.add_point_cloud(
        points=np.array([C_vis]),
        colors=(255, 0, 0),
        point_size=0.03,
        name="camera"
    )

    def inverse_transform(pts):
        R_inv = np.linalg.inv(R)
        return (pts - t) @ R_inv.T

    # If mhmocap flag is set, load and visualize mhmocap joints and verts
    if use_mhmocap:
        frames_root = os.path.join('eval/RICH/mhm_results')
        frames_dir = os.path.join(frames_root, f"{base}_cam{cam_num}_batch{batch}", 'output')
        mhmocap_joints_path = os.path.join(frames_dir, 'mhmocap_joints.npy')
        mhmocap_verts_path = os.path.join(frames_dir, 'mhmocap_verts.npy')
        if not (os.path.exists(mhmocap_joints_path) and os.path.exists(mhmocap_verts_path)):
            print(f"Missing mhmocap npy files in {frames_dir}")
            return
        
        mhmocap_joints = np.load(mhmocap_joints_path)
        mhmocap_verts = np.load(mhmocap_verts_path)
        mhmocap_joints = inverse_transform(mhmocap_joints)
        mhmocap_verts = inverse_transform(mhmocap_verts)
        mhmocap_joints = mhmocap_joints @ T_opencv_to_viser.T
        mhmocap_verts = mhmocap_verts @ T_opencv_to_viser.T
        # Visualize mhmocap verts as mesh
        for i in indices:
            mesh = trimesh.Trimesh(vertices=mhmocap_verts[i], faces=faces)
            mesh.visual.vertex_colors = np.tile([153, 199, 255, 255], (mhmocap_verts[i].shape[0], 1))
            server.add_mesh_trimesh(
                mesh=mesh,
                name=f"mhmocap_mesh_{i}",
            )
        # Visualize mhmocap joints
        # for i in indices:
        #     server.add_point_cloud(
        #         points=mhmocap_joints[i],
        #         colors=(0, 255, 0),
        #         point_size=0.03,
        #         name=f"mhmocap_joints_{i}"
        #     )
        #     # Add bones for mhmocap
        #     edges = []
        #     for j, k in skeleton:
        #         edges.append([mhmocap_joints[i][j], mhmocap_joints[i][k]])
        #     edges = np.array(edges)
        #     server.add_line_segments(
        #         points=edges,
        #         colors=(0, 0, 255),
        #         name=f"mhmocap_bones_{i}"
        #     )
        # Also visualize and transform mhm_scene.ply
        # mhm_scene_path = os.path.join(frames_dir, 'mhm_scene.ply')
        # mhm_scene_pc = trimesh.load(mhm_scene_path)
        # if len(mhm_scene_pc.vertices) > subset_size:
        #     idx = random.sample(range(len(mhm_scene_pc.vertices)), subset_size)
        #     mhm_scene_pc.vertices = mhm_scene_pc.vertices[idx]
        #     mhm_scene_pc.visual.vertex_colors = mhm_scene_pc.visual.vertex_colors[idx]
        # mhm_scene_pc.vertices = mhm_scene_pc.vertices @ T_opengl_to_opencv.T
        # mhm_scene_pc.vertices = inverse_transform(mhm_scene_pc.vertices)
        # mhm_scene_pc.vertices = mhm_scene_pc.vertices @ T_opencv_to_viser.T
        # # Visualize
        # server.add_point_cloud(
        #     points=mhm_scene_pc.vertices,
        #     colors=mhm_scene_pc.visual.vertex_colors[:, :3],
        #     point_size=0.01,
        #     name="mhm_scene"
        # )
    if use_bev:
        frames_root = os.path.join('eval/RICH/bev_results')
        frames_dir = os.path.join(frames_root, f"{base}_cam{cam_num}_batch{batch}")
        bev_joints_path = os.path.join(frames_dir, 'bev_joints.npy')
        bev_verts_path = os.path.join(frames_dir, 'bev_vertices.npy')
        if not (os.path.exists(bev_joints_path) and os.path.exists(bev_verts_path)):
            print(f"Missing BEV npy files in {frames_dir}")
            return
        
        bev_joints = np.load(bev_joints_path)
        bev_verts = np.load(bev_verts_path)
        bev_joints = inverse_transform(bev_joints)
        bev_verts = inverse_transform(bev_verts)
        bev_joints = bev_joints @ T_opencv_to_viser.T
        bev_verts = bev_verts @ T_opencv_to_viser.T
        # Visualize BEV verts as mesh
        for i in indices:
            mesh = trimesh.Trimesh(vertices=bev_verts[i], faces=faces)
            mesh.visual.vertex_colors = np.tile([255, 115, 169, 255], (bev_verts[i].shape[0], 1))
            server.add_mesh_trimesh(
                mesh=mesh,
                name=f"bev_mesh_{i}",
            )
        # Visualize BEV joints
        # for i in indices:
        #     server.add_point_cloud(
        #         points=bev_joints[i],
        #         colors=(0, 255, 0),
        #         point_size=0.03,
        #         name=f"bev_joints_{i}"
        #     )
        #     # Add bones for BEV
        #     edges = []
        #     for j, k in skeleton:
        #         edges.append([bev_joints[i][j], bev_joints[i][k]])
        #     edges = np.array(edges)
        #     server.add_line_segments(
        #         points=edges,
        #         colors=(0, 0, 255),
        #         name=f"bev_bones_{i}"
        #     )
    if use_tram:
        frames_root = os.path.join('eval/RICH/tram_share_results')
        frames_dir = os.path.join(frames_root, f"{base}_cam{cam_num}_batch{batch}")
        tram_joints_path = os.path.join(frames_dir, 'joints.npy')
        tram_verts_path = os.path.join(frames_dir, 'vertices.npy')
        if not (os.path.exists(tram_joints_path) and os.path.exists(tram_verts_path)):
            print(f"Missing tram npy files in {frames_dir}")
            return
        
        tram_joints = np.load(tram_joints_path)
        tram_verts = np.load(tram_verts_path)
        tram_joints = inverse_transform(tram_joints)
        tram_verts = inverse_transform(tram_verts)
        tram_joints = tram_joints @ T_opencv_to_viser.T
        tram_verts = tram_verts @ T_opencv_to_viser.T
        # Visualize tram verts as mesh
        for i in indices:
            mesh = trimesh.Trimesh(vertices=tram_verts[i], faces=faces)
            mesh.visual.vertex_colors = np.tile([255, 169, 77, 255], (tram_verts[i].shape[0], 1))
            server.add_mesh_trimesh(
                mesh=mesh,
                name=f"tram_mesh_{i}",
            )
        # Visualize tram joints
        # for i in indices:
        #     server.add_point_cloud(
        #         points=tram_joints[i],
        #         colors=(0, 255, 0),
        #         point_size=0.03,
        #         name=f"tram_joints_{i}"
        #     )
        #     # Add bones for tram
        #     edges = []
        #     for j, k in skeleton:
        #         edges.append([tram_joints[i][j], tram_joints[i][k]])
        #     edges = np.array(edges)
        #     server.add_line_segments(
        #         points=edges,
        #         colors=(0, 0, 255),
        #         name=f"tram_bones_{i}"
        #     )
    if use_share:
        frame_root = os.path.join('eval/RICH/tram_share_results')
        frame_dir = os.path.join(frame_root, f"{base}_cam{cam_num}_batch{batch}")
        share_joints_path = os.path.join(frame_dir, 'share_joints.npy')
        share_verts_path = os.path.join(frame_dir, 'share_vertices.npy')
        if not (os.path.exists(share_joints_path) and os.path.exists(share_verts_path)):
            print(f"Missing SHARE npy files in {frame_dir}")
            return
        share_joints = np.load(share_joints_path)
        share_verts = np.load(share_verts_path)
        share_joints = inverse_transform(share_joints)
        share_verts = inverse_transform(share_verts)
        share_joints = share_joints @ T_opencv_to_viser.T
        share_verts = share_verts @ T_opencv_to_viser.T
        print(share_joints.shape)
        # Visualize SHARE verts as mesh
        for i in indices:
            mesh = trimesh.Trimesh(vertices=share_verts[i], faces=faces)
            mesh.visual.vertex_colors = np.tile([59, 255, 65, 255], (share_verts[i].shape[0], 1))
            server.add_mesh_trimesh(
                mesh=mesh,
                name=f"share_mesh_{i}",
            )
        # Visualize SHARE joints
        # for i in indices:
        #     server.add_point_cloud(
        #         points=share_joints[i],
        #         colors=(0, 255, 0),
        #         point_size=0.03,
        #         name=f"share_joints_{i}"
        #     )
        #     # Add bones for SHARE
        #     edges = []
        #     for j, k in skeleton:
        #         edges.append([share_joints[i][j], share_joints[i][k]])
        #     edges = np.array(edges)
        #     server.add_line_segments(
        #         points=edges,
        #         colors=(0, 0, 255),
        #         name=f"share_bones_{i}"
        #     )   
        share_scene_path = os.path.join(frame_dir, 'share_scene.ply')
        share_scene_pc = trimesh.load(share_scene_path)
        # if len(share_scene_pc.vertices) > subset_size:
        #     idx = random.sample(range(len(share_scene_pc.vertices)), subset_size)
        #     share_scene_pc.vertices = share_scene_pc.vertices[idx]
        #     share_scene_pc.visual.vertex_colors = share_scene_pc.visual.vertex_colors[idx]
        share_scene_pc.vertices = inverse_transform(share_scene_pc.vertices)
        share_scene_pc.vertices = share_scene_pc.vertices @ T_opencv_to_viser.T
        # Visualize
        # server.add_point_cloud(
        #     points=share_scene_pc.vertices,
        #     colors=share_scene_pc.visual.vertex_colors[:, :3],
        #     point_size=0.001,
        #     name="share_scene"
        # )
    # print(np.mean(np.linalg.norm(joints[:, 0]-bev_joints[:, 0], axis=1)))
    # print(np.mean(np.linalg.norm(verts-share_verts, axis=2)))
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()