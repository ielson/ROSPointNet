import numpy as np
import rospy
from sensor_msgs.msg import PointCloud
import open3d as o3d

point_cloud_data = []

def numpy_to_o3d(np_array: np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd

def callback(pc_msg):
    """
    Converts sensor_msgs/PointCloud.msg to numpy array.
    http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud.html
    """

    xyz = []
    for data in pc_msg.points:
        print(f"Appending message to the list {data.x}, {data.y}, {data.z}")
        point_cloud_data.append(np.array([data.x, data.y, data.z]))
        #xyz.append([data.x, data.y, data.z])

    #point_cloud_data.append(np.array(xyz))

def save_to_point_cloud_file(file_name: str = 'pcd_from_ros'):
    """
    Hook function called whenever the node is killed.
    Saves numpy array to pcd file by taking the global variabe point_cloud_data.
    """
    print(f"Saving PCD with {pcd} to a file...")
    pcd = numpy_to_o3d(point_cloud_data)
    o3d.io.write_point_cloud(f"{file_name}.pcd", pcd)

def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/scan", PointCloud, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
    # Try exception to check if the list is empty, otherwise do not save a file
    rospy.on_shutdown(save_to_point_cloud_file)