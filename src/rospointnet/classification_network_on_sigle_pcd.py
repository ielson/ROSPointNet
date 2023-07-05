
from pointnet_classification import set_network_input_parameters, locate_and_parse_dataset, test_trained_model, create_model
from pointnet_classification import set_dataset_directories, open_a_mesh_from_dataset
from pyntcloud import PyntCloud
import tensorflow as tf
from os import path
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
    for geometry in pcd:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    #opt.background_color = np.asarray([0.5, 0.5, 0.5])
    
    viewer.run()
    viewer.destroy_window()

def pcd_to_numpy_array(pcd: o3d.geometry.PointCloud):
    return np.asarray(pcd.points)

def get_cropping_bounding_box_from_pc(pcd: o3d.geometry.PointCloud):
    bb = pcd.get_axis_aligned_bounding_box()
    bb_limits = bb.get_box_points()
    bb_limits = np.asarray(bb_limits)
    # Adds offset to the X component so that the bounding box excludes the points
    bb_limits[bb_limits == 0.0] = 0.05
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bb_limits))    
    bb.color = [0.0, 0.0, 0.1]
    return bb

def crop_pc_from_bounding_box(pcd: o3d.geometry.PointCloud,
                              bounding_box: o3d.geometry.AxisAlignedBoundingBox):
    return pcd.crop(bounding_box)

def numpy_to_o3d(np_array: np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd

# TODO: Implement logic for checking if upsampling is needed
def check_if_pc_needs_upsampling():
    """
    """
    pass

def downsample_input_pc_dimensions(pcd: o3d.geometry.PointCloud,
                               num_points: int) -> o3d.geometry.PointCloud:
    """
    Uses farthest to downsample pcl to specific dimension.
    """
    
    pc = np.asarray(pcd.points)
    pc_size = pc.shape[0]

    if pc_size < num_points:
        print("Point cloud has fewer points than what the classification network expects, upsampling is needed.")
        raise NotImplementedError
    else:
        return pcd.farthest_point_down_sample(num_points)

def open_up_pcd_file(file_path: str,
                     num_points: int,
                     adjust_pc_shape: bool = True,
                     visualize: bool = False) -> np.array:
    """
    Opens up pcd file using Open3D library.
    Then extract pcd_points only
    """
    pcd = o3d.io.read_point_cloud(file_path)
    
    #pcd = remove_sensor_noise_from_simulation(pcd)

    bb = get_cropping_bounding_box_from_pc(pcd)

    pcd = crop_pc_from_bounding_box(pcd, bb)

    if adjust_pc_shape:
        pcd = downsample_input_pc_dimensions(pcd, num_points)
        print(f"Input point cloud with shape of : {pcd}")

    if visualize:
        visualize_pcd([pcd])
    
    points = pcd_to_numpy_array(pcd)
    return points

def convert_pc_to_tensor(
                            input_pc: np.array

):
    """
    
    """
    
    print(f"Input point cloud has size of {input_pc.shape}")

    # <class 'tensorflow.python.framework.ops.EagerTensor'> shape=(2048, 3), dtype=float64)
    pc_as_tensor = tf.convert_to_tensor(input_pc)

    # Adds new dimension over the first axis so that it complies with Network (None, N, 3)
    # Current: (1, 2048, 3)
    pc_as_tensor = tf.expand_dims(pc_as_tensor, axis=0)
    print(pc_as_tensor.shape.as_list())

    return pc_as_tensor

def get_one_point_cloud_from_dataset(test_dataset, classmap):
    """
    Gets one model from test dataset only.
    """
    
    # Slices model
    data = test_dataset.take(1)

    points, labels = list(data)[0]
    points = points[:8, ...]
    labels = labels[:8, ...]

if __name__ == "__main__":

    # Assuming model has to be created because we just saved the trained weights
    # Loads weights and instantiates model
    WEIGHTS_PATH = '../../pointnet_network_config/weights/modelnet10_weights.h5'
    NUM_POINTS, NUM_CLASSES, BATCH_SIZE = set_network_input_parameters()
    model = create_model(NUM_POINTS, NUM_CLASSES)
    model.load_weights(WEIGHTS_PATH)

    # -------------- Use this to test the Modelnet10 dataset -------------------
    # Loads dataset
    using_modelnet10 = True
    if using_modelnet10:
        train_set, test_set, CLASS_MAP = locate_and_parse_dataset(NUM_POINTS= 2048,
                                NUM_CLASSES = 10,
                                BATCH_SIZE = 32)

        get_one_point_cloud_from_dataset(test_set, CLASS_MAP)



    # ------------- Use this when using single pcd files ----------------

    POINT_CLOUD_FILE_ROOT_PATH = '../../test_data/'
    PC_FILE = 'airplane.pcd'
    pc_full_path = path.join(POINT_CLOUD_FILE_ROOT_PATH, PC_FILE)

    # Testing for a single point cloud on the trained net
    input_point_cloud = open_up_pcd_file(pc_full_path, NUM_POINTS, adjust_pc_shape=True, visualize=True)

    converted_pc = convert_pc_to_tensor(input_point_cloud)

    pred = model.predict(converted_pc)

    print('pred before argmax: ' + str(pred))
    pred = tf.math.argmax(pred, -1)
    print(f' Pred after argmax {pred}')

    """
    # print("pred: {:}".format(classmap[pred.numpy()]))
    # CLASS_MAP[preds[i].numpy()]
    # formato: tf.Tensor: shape=(1,), dtype=int64, numpy=array([0]
    pred.numpy()[0]
    """
