import open3d as o3d
import numpy as np

def visualize_pcd(pcd: list):
    """
    Visualize it in Open3D interface.
    pcd: list. Must be passed in a list format, if wished multiple
    point clouds can be passe.  
    """
    o3d.visualization.draw_geometries(pcd,
                                    zoom=0.49,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

def numpy_to_o3d(np_array: np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd

def get_pass_through_filter_boundaries(point_cloud_points: np.array,
                                       # 0.08cm offset
                                       z_offset_ground_removal = 0.08):
    """
    Based on the input point cloud, get min and max of each dimension (XYZ) and adds
    one small offsets in the Z-axis so that ground plane removal works and outputs
    a dictionary so that it can be used on a pass-through filter.

    Args:
    - point_cloud_points: numpy array containing the points.
    - z_offset_ground_removal: float. Default value is 0.08cm because that's where the sensor is located.
    """

    x_max, y_max, z_max = point_cloud_points.max(axis=0)
    x_min, y_min, z_min = point_cloud_points.min(axis=0)
    

    filter_boundaries = {
        "x": [x_min, x_max],
        "y": [y_min, y_max],
        "z": [z_min + z_offset_ground_removal, z_max]
    }

    return filter_boundaries

def pass_through_filter(boundaries: dict,
                        points: o3d.geometry.PointCloud.points) -> o3d.geometry.PointCloud:
    """
    Removes one
    Args:
        boundaries (dict)
    Returns
        o3dpc (open3d.geometry.PointCloud): filtered open3d point cloud
    """
    #points = np.asarray(pcd.points)
    #print(f"Input Point Cloud with size of: {len(points)}")
    x_range = np.logical_and(points[:,0] >= boundaries["x"][0] ,points[:,0] <= boundaries["x"][1])
    y_range = np.logical_and(points[:,1] >= boundaries["y"][0] ,points[:,1] <= boundaries["y"][1])
    z_range = np.logical_and(points[:,2] >= boundaries["z"][0] ,points[:,2] <= boundaries["z"][1])
    pass_through_filter = np.logical_and(x_range,np.logical_and(y_range,z_range))
    pcd.points = o3d.utility.Vector3dVector(points[pass_through_filter])
    return pcd

def preprocess_point_cloud_offline(pcd: o3d.geometry.PointCloud,
                           pcd_file_path: str = '',
                           pcd_file_name: str = 'output_point_cloud.pcd',
                           write_to_file: bool = False,
                           verbose:bool = False) -> o3d.geometry.PointCloud:
    """
    Preprocesses the point cloud by using a pass-through filter to remove the ground plane.
    This function has the sole purpose to be used for offline point clouds due to debugging functionalities that
    aren't worth to be incorporated to an online scenario/function.
    """
    pcd_points = np.asarray(pcd.points)
    filter_boundaries = get_pass_through_filter_boundaries(pcd_points)
    filtered_pc = pass_through_filter(filter_boundaries, pcd_points)
    if verbose:
        print(f"Point cloud has {filtered_pc} points.")
    if write_to_file:
        o3d.io.write_point_cloud(f"{pcd_file_path}preprocessed_{pcd_file_name}", pcd)
    return filtered_pc

if __name__ == '__main__':
    path = '../test_data/chair/'
    file_name = 'first_snapshot.pcd'
    file_location = f"{path}{file_name}"
    pcd = o3d.io.read_point_cloud(file_location)

    preprocessed_pc = preprocess_point_cloud_offline(pcd, path, file_name, write_to_file=True, verbose=True)
    visualize_pcd([preprocessed_pc])