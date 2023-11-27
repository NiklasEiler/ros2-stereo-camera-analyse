#!/usr/bin/env python3

import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import cv2

class ImageSubscriber:
    def __init__(self):
        self.node = rclpy.create_node('image_subscriber')
        self.subscription = self.node.create_subscription(
            Image,
            '/stereo/left/image_rect',  # topic of image stream
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # This function will be called whenever a new image is received
        # Process the image data as needed
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.node.get_logger().info('Image2 recieved')
            # Add your image processing code here

            # Display the image (for example)
            if 1:
                cv2.imshow('Image', cv_image)
                cv2.waitKey(1)

        except Exception as e:
            self.node.get_logger().info("Error processing image:", str(e))



def main():
    rclpy.init()
    subscriber = ImageSubscriber()
    rclpy.spin(subscriber.node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()