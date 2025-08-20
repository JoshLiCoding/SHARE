import sys
import trimesh
import numpy as np
import open3d as o3d
import os
import imageio

def render_mesh_offscreen(frame, total_frames, meshes, xyz_limits, scene_vertices, scene_colors, width=640, height=480):
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

    # Add scene point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(scene_vertices)
    if scene_colors is not None and len(scene_colors) == len(scene_vertices):
        pc.colors = o3d.utility.Vector3dVector(scene_colors[:, :3] / 255.0)
    else:
        pc.paint_uniform_color([0.5, 0.5, 0.5])
    renderer.scene.add_geometry("scene", pc, o3d.visualization.rendering.MaterialRecord())

    center = np.mean([xyz_limits[0], xyz_limits[1], xyz_limits[2]], axis=1)
    # Toyota
    # eye = [3, -2, 3]
    # up = [0, -1, -0.4]
    # renderer.setup_camera(60.0, center + [0, 1, 0], eye, up)
    pan_range = 6  # Adjust the range of the pan
    pan_offset = -pan_range / 2 + (frame / (total_frames - 1)) * pan_range  # Linear interpolation
    
    renderer.setup_camera(60.0, center, [pan_offset, 0, -4], [0, -1, 0])

    # Render
    img = renderer.render_to_image()
    img_np = np.asarray(img)
    del renderer
    return img_np

def save_video(image_list, output_path, fps=30): # 20
    print(f"Saving {len(image_list)} frames to video...")
    imageio.mimsave(output_path, image_list, fps=fps, quality=8)

T_opengl_to_opencv = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
])
def visualize_scene_and_joints(folder):
    # Load scene point cloud
    scene_pc = trimesh.load(os.path.join(folder, 'output', 'mhm_scene.ply'))
    scene_points = scene_pc.vertices
    scene_colors = scene_pc.visual.vertex_colors[:, :3]
    scene_points = scene_points @ T_opengl_to_opencv.T

    print(len(scene_points))

    # Load mesh data
    faces = np.load('vis/faces.npy')
    share_vert = np.load(os.path.join(folder, 'output', 'mhmocap_verts.npy'))

    # Calculate axis limits from scene
    scene_min = scene_points.min(axis=0)
    scene_max = scene_points.max(axis=0)
    xyz_limits = [
        [scene_min[0], scene_max[0]],
        [scene_min[1], scene_max[1]],
        [scene_min[2], scene_max[2]],
    ]

    # Generate frames
    rendered_frames = []

    num_meshes = len(share_vert)
    for i in range(num_meshes):
        # Use uniform color instead of gradient
        color = [255, 255, 255, 255]
        
        # Create mesh list for this frame
        meshes = [(share_vert[i], faces, color)]
        
        # Render frame
        img = render_mesh_offscreen(
            frame=i,
            total_frames=num_meshes,
            meshes=meshes,
            xyz_limits=xyz_limits,
            scene_vertices=scene_points,
            scene_colors=scene_colors
        )
        rendered_frames.append(img)

    # Save video
    if rendered_frames:
        print(len(rendered_frames))
        output_path = f"{os.path.basename(folder)}_MHMocap.mp4"
        save_video(rendered_frames, output_path)
        print(f"Video saved to {output_path}")
    else:
        print("No frames were rendered.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder")
    args = parser.parse_args()

    visualize_scene_and_joints(args.folder)