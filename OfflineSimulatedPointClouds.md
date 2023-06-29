## Setup

- C++ 17
- Ubuntu 20.04
- ROS Noetic

## Installation

Clone the following repositories in order to run a simulated environment in Gazebo and the simulated Lidar:

```shell
$ git clone git@github.com:Livox-SDK/livox_laser_simulation.git
$ git clone git@github.com:ericksuzart/lar_gazebo.git
$ git clone git@github.com:ielson/ROSPointNet.git
$ git clone https://github.com/Livox-SDK/livox_ros_driver.git
$ git clone git@github.com:pal-robotics-forks/point_cloud_converter.git
```

After cloning all required packages, **if using ROS Noetic, make sure to go to and change in /livox_laser_simulation/CMakeLists.txt in Line 5** from

```## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
```

To:
```
add_compile_options(-std=c++17)
```

And update your catkin workspace:

```shell 
$ cd ~/catkin_ws
$ catkin_make
```

Now visualize the environment by launching the launch file:

`$ roslaunch rospointnet laboratory_environment_with_lidar.launch`

Place screenshot.

## Registering the simulation and saving a snapshot

Assuming the simulation is working properly like in the command in the previous section, record the simulation by using rosbag:

`$ rosbag record -a`

Now play the recorded rosbag:

` $ rosbag play `

Go to where your bag file is and convert:

`$ rosrun pcl_ros bag_to_pcd BAG_FILE_NAME.bag /converted_point_cloud_2 .`

Expected output:

```
Saving recorded sensor_msgs::PointCloud2 messages on topic /converted_point_cloud_2 to .
Got 10000 data points in frame laser_livox on topic /converted_point_cloud_2 with the following fields: x y z
Data saved to ./36.530000000.pcd
Got 10000 data points in frame laser_livox on topic /converted_point_cloud_2 with the following fields: x y z
Data saved to ./36.630000000.pcd
```

