import open3d as o3d
import numpy as np

def visualize_pcd(pcd: list):
    """
    Visualize it in Open3D interface.
    pcd: list. Must be passed in a list format, if wished multiple
    point clouds can be passe.  
    """
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    geometries = [pcd]
    for geometry in pcd:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    #opt.background_color = np.asarray([0.5, 0.5, 0.5])
    #viewer.destroy_window()
    """o3d.visualization.draw_geometries(pcd,
                                    zoom=0.49,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])"""
    
    viewer.run()
    viewer.destroy_window()
# TODO: Use a better method for downsampling to 2048 fixed size (input that PointNet takes in)
# uniform_down_sample?
def voxelgrid_downsampling(pcd):
    #print("Downsample the point cloud with a voxel of 0.05")
    voxelized_pcd = pcd.voxel_down_sample(voxel_size=0.05)
    return voxelized_pcd

def numpy_to_o3d(np_array: np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd

def create_point_cloud_from_bbox_vertices(pcd: o3d.geometry.PointCloud):
    """
    Draws bounding box out of the input point cloud. 
    """
    #pcd_points = np.asarray(pcd.points)
    #pc_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_points)
    
    bb = pcd.get_axis_aligned_bounding_box()
    return bb

def get_pass_through_filter_boundaries(point_cloud_points: np.array,
                                       x_offset: float = 0.0,
                                       y_offset: float = 0.0,
                                       # 8 cm offset
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
        "x": [x_min, x_max - x_offset],
        "y": [y_min, y_max - y_offset],
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

def filter_pc_by_radius(pcd: o3d.geometry.PointCloud,
                        sphere_radius: float = 1.0):
    print(f"Using radius filtering of: {sphere_radius}")
    raw_pcd, inlier_pc_indexes = pcd.remove_radius_outlier(nb_points=16, radius=sphere_radius)
    pcd = raw_pcd.select_by_index(inlier_pc_indexes)
    return pcd

def preprocess_point_cloud_offline(pcd: o3d.geometry.PointCloud,
                           pcd_file_path: str = '',
                           pcd_file_name: str = 'output_point_cloud.pcd',
                           downsampling: bool = False,
                           radius_filtering: bool = False,
                           write_to_file: bool = False,
                           verbose:bool = False) -> o3d.geometry.PointCloud:
    """
    Preprocesses the point cloud by using a pass-through filter to remove the ground plane.
    This function has the sole purpose to be used for offline point clouds due to debugging functionalities that
    aren't worth to be incorporated to an online scenario/function.
    """
    pcd_points = np.asarray(pcd.points)
    # Used an offset of 0.3m so that objects too far away are disregarded
    filter_boundaries = get_pass_through_filter_boundaries(pcd_points, x_offset=0.3 ,y_offset=1.0)
    pcd = pass_through_filter(filter_boundaries, pcd_points)
    if downsampling:
        pcd = voxelgrid_downsampling(pcd)
    if radius_filtering:
        sphere_radius = 0.1
        pcd = filter_pc_by_radius(pcd, sphere_radius)
    if verbose:
        print(f"Point cloud has {pcd}.")

    if write_to_file:
        o3d.io.write_point_cloud(f"{pcd_file_path}preprocessed_{pcd_file_name}", pcd)
    return pcd


if __name__ == '__main__':
    path = '../test_data/chair/'
    file_name = 'first_snapshot.pcd'
    file_location = f"{path}{file_name}"
    pcd = o3d.io.read_point_cloud(file_location)
    preprocessed_pc = preprocess_point_cloud_offline(pcd, path, file_name, radius_filtering=False, write_to_file=True, verbose=True)
    bounding_box = create_point_cloud_from_bbox_vertices(preprocessed_pc)
    visualize_pcd([preprocessed_pc, bounding_box])