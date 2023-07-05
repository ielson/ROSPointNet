import open3d as o3d
import numpy as np
import os
import pathlib

def read_point_cloud_files(test_data_path: list) -> list:
    """
    Traverse to wished directory and return list of path to files.
    """
    from os import walk
    list_of_files_path = []
    try:
        for (dirpath, dirnames, filenames) in walk(test_data_path):
            #if "wall_table_chair" in dirpath:
            for file_name in filenames:
                # Skipping all files that are not a .pcd
                if not file_name.endswith(".pcd"):
                    pass
                else:
                    print(f"Reading {file_name}")
                    list_of_files_path.append(os.path.join(dirpath, file_name))
        if not list_of_files_path:
            print("There aren't any .pcd files in the directory.")
            return False
        else:
            print(f"Got the following files from the path {test_data_path}: {list_of_files_path}")
            return list_of_files_path
    except ValueError as e:
        print(f"{e} due to wrong or non-existing file name.")

def concatenate_multiple_pcds(list_of_pcd,
                              output_name: str):
    """
    Concatenates multiple PCDs file into a single one.
    """
    
    # Creates output filename based on the folder of given point cloud
    file_name = list_of_pcd[0].split('/')[3]
    output_file = f"{file_name}.pcd"

    # TODO: Inherit variable to make it dynamic
    absolute_path = list_of_pcd[0].partition('47')[0]

    output_file_name = os.path.join(
                        # Root folder path
                        # TODO: Inherit variable to make it dynamic
                        absolute_path,
                        output_file)

    # Stores converted pcds to temp. list to be concatenated later
    pcd_t = []

    try:
        for pcd_file in list_of_pcd:
            print(pcd_file)
            if pcd_file == output_file_name:
                continue
            print(f"Reading {pcd_file}")
            pcd = o3d.io.read_point_cloud(pcd_file)
            pcd_load = np.asarray(pcd.points)
            print(f"Shape of pc: {len(pcd_load)}")
            pcd_t.append(pcd_load)
        final_pcd = np.concatenate(pcd_t)
        print(f"Shape of Final PCD: {final_pcd.shape}")
        concatenated_pcd = o3d.geometry.PointCloud()
        concatenated_pcd.points = o3d.utility.Vector3dVector(final_pcd)
    except IndexError:
        print("Could not read pcd files, maybe the list was empty.")

    # TODO: I think this could be reworked in the beginning of the function
    # Takes absolute path
    path = pathlib.Path(absolute_path)
    absolute_path = path.parent
    output_file_name = os.path.join(
                        # Root folder path
                        absolute_path,
                        # TODO: Inherit variable to make it dynamic
                        'concatenated_pc.pcd')
    print(output_file_name)
    
    o3d.io.write_point_cloud(output_file_name, concatenated_pcd)
    print(f"Wrote merged PCD to {output_file_name}")
    return output_file_name

def read_and_merge_pcd(test_data_path: str,
                       output_name: str
                       ):
    # Adapting for local testing (i.e. reading the PCL as a file, not from ROS)
    list_of_files_path = read_point_cloud_files(test_data_path)
    # TODO: Use try/exception
    if list_of_files_path:
        concatenated_pcd_path = concatenate_multiple_pcds(list_of_files_path, output_name)
        print(f"Reading: {concatenated_pcd_path}")
        pcd = o3d.io.read_point_cloud(concatenated_pcd_path)
        return pcd
    else:
        print("Could not get files.")

def visualize_pcd(pcd: list):
    """
    Visualize it in Open3D interface.
    pcd: list. Must be passed in a list format, if wished multiple
    point clouds can be passe.  
    """
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in pcd:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    #opt.background_color = np.asarray([0.5, 0.5, 0.5])
    
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
    # TODO: Add logging debug
    print(f"Filters are set to x_offset {x_offset}, y_offset {y_offset}, z_offset_ground_removal {z_offset_ground_removal}")

    x_max, y_max, z_max = point_cloud_points.max(axis=0)
    x_min, y_min, z_min = point_cloud_points.min(axis=0)

    print(f"y_min: {y_min}, y_max {y_max}")

    filter_boundaries = {
        "x": [x_min, x_max - x_offset],
        "y": [y_min, y_max - y_offset],
        "z": [z_min + z_offset_ground_removal, z_max]
    }

    print(f"Filter boundaries: {filter_boundaries}")

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

    # Draw


    #points = np.asarray(pcd.points)
    #print(f"Input Point Cloud with size of: {len(points)}")
    x_range = np.logical_and(points[:,0] >= boundaries["x"][0] ,points[:,0] <= boundaries["x"][1])
    y_range = np.logical_and(points[:,1] >= boundaries["y"][0] ,points[:,1] <= boundaries["y"][1])
    z_range = np.logical_and(points[:,2] >= boundaries["z"][0] ,points[:,2] <= boundaries["z"][1])
    pass_through_filter = np.logical_and(x_range,np.logical_and(y_range,z_range))
    pcd.points = o3d.utility.Vector3dVector(points[pass_through_filter])
    return pcd

def draw_guild_lines(self,boundaries, density = 0.01):
    new_col = []
    new_pos = []
    x_start,x_end = boundaries["x"]
    y_start,y_end = boundaries["y"]
    z_start,z_end = boundaries["z"]

    x_points,y_points,z_points = np.asarray(np.arange(x_start,x_end,density)),np.asarray(np.arange(y_start,y_end,density)),np.asarray(np.arange(z_start,z_end,density))
    
    y_starts,y_ends = np.asarray(np.full((len(x_points)),y_start)),np.asarray(np.full((len(x_points)),y_end))
    z_starts,z_ends = np.asarray(np.full((len(x_points)),z_start)),np.asarray(np.full((len(x_points)),z_end))
    lines_x = np.concatenate((np.vstack((x_points,y_starts,z_starts)).T,np.vstack((x_points,y_ends,z_starts)).T,np.vstack((x_points,y_starts,z_ends)).T,np.vstack((x_points,y_ends,z_ends)).T))


    x_starts,x_ends = np.asarray(np.full((len(y_points)),x_start)),np.asarray(np.full((len(y_points)),x_end))
    z_starts,z_ends = np.asarray(np.full((len(y_points)),z_start)),np.asarray(np.full((len(y_points)),z_end))
    lines_y = np.concatenate((np.vstack((x_starts,y_points,z_starts)).T,np.vstack((x_ends,y_points,z_starts)).T,np.vstack((x_starts,y_points,z_ends)).T,np.vstack((x_ends,y_points,z_ends)).T))


    x_starts,x_ends = np.asarray(np.full((len(z_points)),x_start)),np.asarray(np.full((len(z_points)),x_end))
    y_starts,y_ends = np.asarray(np.full((len(z_points)),y_start)),np.asarray(np.full((len(z_points)),y_end))
    lines_z = np.concatenate((np.vstack((x_starts,y_starts,z_points)).T,np.vstack((x_ends,y_starts,z_points)).T,np.vstack((x_starts,y_ends,z_points)).T,np.vstack((x_ends,y_ends,z_points)).T))

    if (self.is_rgb_4byte):
        lines_x_color =  np.full((len(lines_x)),self.rgb2float(255,0,0))#blue for x
        lines_y_color =  np.full((len(lines_y)),self.rgb2float(0,255,0))#green for y
        lines_z_color =  np.full((len(lines_z)),self.rgb2float(0,0,255))#red for z
        return np.concatenate((lines_x,lines_y,lines_z)),np.asmatrix(np.concatenate((lines_x_color,lines_y_color,lines_z_color))).T
    else:
        lines_x_color = np.zeros((len(lines_x),3))
        lines_y_color = np.zeros((len(lines_y),3))
        lines_z_color = np.zeros((len(lines_z),3))

        lines_x_color[:,0] = 1.0 #red for x
        lines_y_color[:,1] = 1.0 #green for y
        lines_z_color[:,2] = 1.0 #blue for z
        return np.concatenate((lines_x,lines_y,lines_z)),np.asmatrix(np.concatenate((lines_x_color,lines_y_color,lines_z_color))) 

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
    filter_boundaries = get_pass_through_filter_boundaries(pcd_points, x_offset=0.3)
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

    # Path used for the pcd concatenation function
    path = '../test_data/table/angle2/'

    # File name used for the preprocess function
    file_name = 'concatenated_pc.pcd'
    file_location = f"{path}{file_name}"

    # Change this flag to True to take a bunch of pcd files generated from ROS and concatenate them into one
    MERGE_PCD = True

    if MERGE_PCD:
        pcd = read_and_merge_pcd(path, file_name)
    else:
        pcd = o3d.io.read_point_cloud(file_location)
        preprocessed_pc = preprocess_point_cloud_offline(pcd, path, file_name, radius_filtering=False, write_to_file=True, verbose=True)
        bounding_box = create_point_cloud_from_bbox_vertices(preprocessed_pc)

    visualize_pcd([pcd])

    """new_data = np.concatenate((new_pos, new_col),axis = 1)
    guild_points = o3d.geometry.PointCloud()
    guild_points.points = o3d.utility.Vector3dVector(new_pos)
    guild_points.colors = o3d.utility.Vector3dVector(new_col)"""