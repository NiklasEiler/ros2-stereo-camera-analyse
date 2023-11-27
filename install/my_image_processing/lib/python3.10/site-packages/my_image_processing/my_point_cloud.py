import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/stereo/points2',  # Replace with the actual topic name
            self.pointcloud_callback,
            10
        )
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Extract x, y, z coordinates
        x = []
        y = []
        z = []
        for point in points:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])

        # Plot the point cloud
        self.ax.clear()
        self.ax.scatter(x, y, z)
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        plt.pause(0.1)  # Pause to allow for real-time visualization

def main(args=None):
    rclpy.init(args=args)
    pointcloud_subscriber = PointCloudSubscriber()
    rclpy.spin(pointcloud_subscriber)
    pointcloud_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()