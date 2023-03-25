"""! @brief Defines the PanopticROS Class.
The package subscribes to the camera images and publishes instances of object detection.
"""
##
# @file panoptic_node.py
#
# @brief Defines the PanopticROS classes.
#
# @section description_panoptic_ros Description
# Defines the class to connect panoptic with the ROS 2 Environment.
# - PanopticROS
#
# @section libraries_pcdet_ros Libraries/Modules
# - ROS 2 Humble (https://docs.ros.org/en/humble/index.html)
# - realtime_panoptic (https://github.com/TRI-ML/realtime_panoptic)
#
# @section author_panoptic_ros Author(s)
# - Created by Shrijal Pradhan on 23/03/2023.
#
# Copyright (c) 2023 Shrijal Pradhan.  All rights reserved.

# Imports
import rclpy
from rclpy.node import Node

import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection.image_list import ImageList

from realtime_panoptic.models.rt_pano_net import RTPanoNet
from realtime_panoptic.config import cfg
import realtime_panoptic.data.panoptic_transform as P
from realtime_panoptic.utils.visualization import visualize_segmentation_image

import datasets.cityscapes

class PanopticROS(Node):
    def __init__(self):
        super().__init__('rt_panoptic')

def main(args=None):
    rclpy.init(args=args)
    panoptic_node = PanopticROS()
    rclpy.spin(panoptic_node)
    panoptic_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()