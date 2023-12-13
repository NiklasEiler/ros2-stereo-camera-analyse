import open3d as o3d

def visualize_ply_file(ply_filename):
    # Read the PLY file
    point_cloud = o3d.io.read_point_cloud(ply_filename)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    ply_file_path = "cloud.ply"  # Replace with the actual path to your PLY file
    visualize_ply_file(ply_file_path)