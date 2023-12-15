#!/usr/bin/env python3

import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import my_image_processing.bildverarbeitung as bild 
import my_image_processing.pointcloud2  as pc2
import os
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d



 
#gloable Parameter
key='#'



class ImageSubscriber:
    def __init__(self):
        self.n=1
        self.right_idx=1
        self.left_idx=1
        self.node = rclpy.create_node('image_subscriber')
        #self.imageright = self.node.create_subscription(
        #    Image,
        #    '/stereo/right/image_rect',  # topic of image stream
        #    self.imageright_callback,
        #    10
        #)

        self.imageleft = self.node.create_subscription(
            Image,
            '/stereo/left/image_rect',  # topic of image stream
            self.imageleft_callback,
            10
        )

        #self.pcd_subscriber = self.node.create_subscription(
        #    PointCloud2,    
        #    '/stereo/points2',                      
        #    self.pointcloud_callback,     
        #    10                          
        #)

        #self.pcd_subscriber = self.node.create_subscription(
        #    Image,    
        #    '/stereo/depth',                      
        #    self.stereo_callback,     
        #    10                          
        #)

        self.bridge = CvBridge()

    def imageright_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            #self.node.get_logger().info('Image recieved' )
            # Add your image processing code here

            
            if 1:
                img= bild.Bildverarbeitung(cv_image)
                #img.binar()
                img.regionofattraction()
                img.aussenkontur()
                img.sobel_img()
                #img.Hough_Circles()
                
                #cv2.imshow('Image', img.regionofattraction_img )
                cv2.imshow('Image', img.imgregion)
                if cv2.waitKey(1)== 49:                          
                    print('save right')
                    img.speichern('~/Pictures', 'right', self.right_idx)
                    self.right_idx= self.right_idx + 1
        except:
            self.node.get_logger().info("Error processing image")

    def imageleft_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            #self.node.get_logger().info('Image recieved' )
            # Add your image processing code here

            
            if 1:
                img= bild.Bildverarbeitung(cv_image)
                img.regionofattraction()
                img.aussenkontur()
                #img.Hough_Circles()
                
                cv2.imshow('Image', img.countur_img)
                if cv2.waitKey(1)== 50:                          
                    print('save left')
                    img.speichern('~/Pictures', 'left', self.left_idx)
                    self.left_idx= self.left_idx + 1
        except:
            self.node.get_logger().info("Error processing image")

    def pointcloud_callback(self, msg):
        try:
            xyz = np.array([[0,0,0]])
            gen = pc2.read_points(msg, skip_nans=True)
            int_data = list(gen)
            idx=0
            for x in int_data:
                # you can get back the float value by the inverse operations
                xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
                #rgb = np.append(rgb,[[r,g,b]], axis = 0)
                idx=idx+1

            out_pcd = o3d.geometry.PointCloud()    
            out_pcd.points = o3d.utility.Vector3dVector(xyz)
            #out_pcd.colors = o3d.utility.Vector3dVector(rgb)
            o3d.io.write_point_cloud("cloud.ply",out_pcd)
        except:
            print('##Error')
        #self.get_logger().info('Received PointCloud2 message')
        
        # Extract data from the PointCloud2 message
        #cloud = read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        #cloud= np.array(list(cloud))
        #print(type(np.array(list(cloud))))
        # Print the first few points
        #for point in cloud:
            #print('Point: x=%f, y=%f, z=%f' % point[:3])
        #if self.n==10:
        #    np.savetxt('test.csv', cloud)
        #self.n=self.n+1

    def stereo_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            #cv_image = self.bridge.imgmsg_to_cv2(msg, encoding="32FC1")
            #self.node.get_logger().info('Image recieved' )
            # Add your image processing code here
            x=1
            
            #if 1:
                #img= bild.Bildverarbeitung(cv_image)
                #img.regionofattraction()
                #img.aussenkontur()
                #img.Hough_Circles()
                
                #cv2.imshow('Image', img.countur_img)
                #if cv2.waitKey(1)== 49:                          
                    #print('save')
                    #img.speichern('~/Pictures')

        except:
            self.node.get_logger().info("Error processing stereo")

def main():
    rclpy.init()
    subscriber = ImageSubscriber()
    rclpy.spin(subscriber.node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()