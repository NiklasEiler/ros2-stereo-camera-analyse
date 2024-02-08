#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pandas as pd
from std_msgs.msg import String
import my_image_processing.bildverarbeitung as bild 

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_node')
        self.safe_right=-1
        self.safe_left=-1
        self.cnt_img_r_s=1
        self.cnt_img_l_s=1
        self.imageright = self.create_subscription(
            Image,
            '/stereo/left/image_rect',  
            self.imageright_callback,
            10
        )
        
        self.imageleft = self.create_subscription(
            Image,
            '/stereo/left/image_rect',  
            self.imageleft_callback,
            10
        )
        

        self.subscription = self.create_subscription(
            String,
            '/gui/speichern',  
            self.button_callback,
            10  
        )
        self.bridge = CvBridge()

    def imageright_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.get_logger().info('Image recieved r' )
            cv2.imwrite('tmp.png',cv_image)
            #cv2.imshow('r_img',cv_image)
            if self.safe_right==1:
                path= "/home/stereocamera/Documents/stereocamera_mesurment/r/"
                cv2.imwrite(path + 'm_r_' + str(self.cnt_img_r_s) + '.png',cv_image)
                self.safe_right=-1
                self.cnt_img_r_s+=1

            #img= bild.Bildverarbeitung(cv_image)
                
        except:
            self.get_logger().info("Error processing right image")
    
    def button_callback(self, msg):
            self.safe_right=1
            self.safe_left=1
            print('save')

    def imageleft_callback(self, msg):
        try:
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.get_logger().info('Image recieved l' )
            
            if self.safe_left==1:
                path= "/home/stereocamera/Documents/stereocamera_mesurment/l/"
                cv2.imwrite(path + 'm_l_' + str(self.cnt_img_l_s) + '.png',cv_image)
                self.safe_left=-1
                self.cnt_img_l_s+=1

            #img= bild.Bildverarbeitung(cv_image)
                
        except:
            self.get_logger().info("Error processing left image")
'''

'''


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    print('start image')
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()