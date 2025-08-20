import sys
import os
import numpy as np
import trimesh
import viser
import time
import random
import open3d as o3d
import imageio

subset_size = 1000000
indices = np.arange(0, 100)

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

    faces_path = 'vis/faces.npy'
    faces = np.load(faces_path)

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
    def inverse_transform(pts):
        R_inv = np.linalg.inv(R)
        return (pts - t) @ R_inv.T

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

    scan_min = scan_pc.vertices.min(axis=0)
    scan_max = scan_pc.vertices.max(axis=0)
    xyz_limits = [
        [scan_min[0], scan_max[0]],
        [scan_min[1], scan_max[1]],
        [scan_min[2], scan_max[2]],
    ]

    rendered_frames = []

    for i in indices:
        meshes = []
        batched_dir = os.path.join('eval/RICH/batched_smpl_motions', folder_name)
        verts_path = os.path.join(batched_dir, 'vertices.npy')
        joints_path = os.path.join(batched_dir, 'joints.npy')

        verts = np.load(verts_path)
        joints = np.load(joints_path)

        gt_mesh = (verts[i], faces, (255, 255, 255, 255))
        meshes.append(gt_mesh)

        if use_mhmocap:
            frames_root = os.path.join('eval/RICH/mhm_results')
            frames_dir = os.path.join(frames_root, f"{base}_cam{cam_num}_batch{batch}", 'output')
            mhmocap_joints_path = os.path.join(frames_dir, 'mhmocap_joints.npy')
            mhmocap_verts_path = os.path.join(frames_dir, 'mhmocap_verts.npy')
            
            mhmocap_joints = np.load(mhmocap_joints_path)
            mhmocap_verts = np.load(mhmocap_verts_path)
            mhmocap_joints = inverse_transform(mhmocap_joints)
            mhmocap_verts = inverse_transform(mhmocap_verts)
            
            mhmocap_mesh = (mhmocap_verts[i], faces, (153, 199, 255, 255))
            meshes.append(mhmocap_mesh)

        if use_bev:
            frames_root = os.path.join('eval/RICH/bev_results')
            frames_dir = os.path.join(frames_root, f"{base}_cam{cam_num}_batch{batch}")
            bev_joints_path = os.path.join(frames_dir, 'bev_joints.npy')
            bev_verts_path = os.path.join(frames_dir, 'bev_vertices.npy')
            
            bev_joints = np.load(bev_joints_path)
            bev_verts = np.load(bev_verts_path)
            bev_joints = inverse_transform(bev_joints)
            bev_verts = inverse_transform(bev_verts)

            bev_mesh = (bev_verts[i], faces, (255, 115, 169, 255))
            meshes.append(bev_mesh)

        if use_tram:
            frames_root = os.path.join('eval/RICH/tram_share_results')
            frames_dir = os.path.join(frames_root, f"{base}_cam{cam_num}_batch{batch}")
            tram_joints_path = os.path.join(frames_dir, 'joints.npy')
            tram_verts_path = os.path.join(frames_dir, 'vertices.npy')

            tram_joints = np.load(tram_joints_path)
            tram_verts = np.load(tram_verts_path)
            tram_joints = inverse_transform(tram_joints)
            tram_verts = inverse_transform(tram_verts)

            tram_mesh = (tram_verts[i], faces, (255, 169, 77, 255))
            meshes.append(tram_mesh)

        if use_share:
            frame_root = os.path.join('eval/RICH/tram_share_results')
            frame_dir = os.path.join(frame_root, f"{base}_cam{cam_num}_batch{batch}")
            share_joints_path = os.path.join(frame_dir, 'share_joints.npy')
            share_verts_path = os.path.join(frame_dir, 'share_vertices.npy')

            share_joints = np.load(share_joints_path)
            share_verts = np.load(share_verts_path)
            share_joints = inverse_transform(share_joints)
            share_verts = inverse_transform(share_verts)

            share_mesh = (share_verts[i], faces, (59, 255, 65, 255))
            meshes.append(share_mesh)
        img = render_mesh_offscreen(meshes, xyz_limits, scan_pc.vertices, scan_pc.visual.vertex_colors)
        rendered_frames.append(img)
    if rendered_frames:
        out_path = f"{folder_name}_mhmocap_rendered.mp4"
        save_video(rendered_frames, out_path)
        print(f"[✓] Saved video to {out_path}")
    else:
        print("[!] No frames were rendered.")

def render_mesh_offscreen(meshes, xyz_limits, scene_vertices, scene_colors, width=640, height=480):
    # Setup offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # white background

    for i, mesh in enumerate(meshes):
        vertices, faces, color = mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.paint_uniform_color(np.array(color[:3]) / 255.0)
        mesh_o3d.compute_vertex_normals()

        material = o3d.visualization.rendering.MaterialRecord()
        material.base_color = [c/255.0 for c in color[:3]] + [1.0]
        material.shader = "defaultLit"
        renderer.scene.add_geometry(f"mesh_{i}", mesh_o3d, material)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(scene_vertices)
    pc.colors = o3d.utility.Vector3dVector(scene_colors[:, :3] / 255.0)    
    renderer.scene.add_geometry("scene", pc, o3d.visualization.rendering.MaterialRecord())

    # Set camera bounds
    center = np.mean([xyz_limits[0], xyz_limits[1], xyz_limits[2]], axis=1)

    # Gym
    # eye = np.array([0.5, 0.0, 0])  # Camera at origin
    # up = [0, -1, 0]  # Adjust if needed for your coordinate system
    # renderer.setup_camera(60.0, center + [0, 1.5, 0], eye, up)

    # ParkingLot2
    eye = np.array([0, 0.0, 1])  # Camera at origin
    up = [0, -1, 0]  # Adjust if needed for your coordinate system
    renderer.setup_camera(60.0, [0, 0, 2], eye, up)

    # Render
    img = renderer.render_to_image()
    img_np = np.asarray(img)
    del renderer 
    return img_np

def save_video(image_list, output_path, fps=30):
    print(len(image_list))
    # imageio expects images in RGB format and shape (H, W, 3)
    imageio.mimsave(output_path, image_list, fps=fps, quality=8)


if __name__ == "__main__":
    main()