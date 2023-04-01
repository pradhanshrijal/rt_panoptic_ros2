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
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import ObjectHypothesisWithPose

import copy
import argparse
import torch
import numpy as np
from PIL import Image as PIL_Image
from torchvision.models.detection.image_list import ImageList

from realtime_panoptic.config import cfg
import realtime_panoptic.data.panoptic_transform as P
from realtime_panoptic.utils.visualization import random_color

from datasets.dataset_selection import DatasetSelection

class PanopticROS(Node):
    def __init__(self):
        super().__init__('rt_panoptic')
        self.__initBaseParams__()
        self.__initObjects__()

    def __imageCB__(self, image_msg):
        img = self.__bridge__.imgmsg_to_cv2(image_msg, "bgr8")
        panoptic_result, segmentation_result, det_msg = self.__runDetectionInstance__(img)

        det_msg.header.frame_id = image_msg.header.frame_id
        det_msg.header.stamp = self.get_clock().now().to_msg()
        pan_pub_msg = self.__bridge__.cv2_to_imgmsg(panoptic_result, "passthrough")
        pan_pub_msg.header.frame_id = image_msg.header.frame_id
        pan_pub_msg.header.stamp = self.get_clock().now().to_msg()
        seg_pub_msg = self.__bridge__.cv2_to_imgmsg(segmentation_result, "passthrough")
        seg_pub_msg.header.frame_id = image_msg.header.frame_id
        seg_pub_msg.header.stamp = self.get_clock().now().to_msg()

        self.__pub_pan_image__.publish(pan_pub_msg)
        self.__pub_seg_image__.publish(seg_pub_msg)
        self.__pub_det__.publish(det_msg)
    
    def __getBBox__(self, box):
        TO_REMOVE = 1
        out_box = BoundingBox2D()
        box_temp = box.convert("xywh")
        xmin, ymin, w, h = box.bbox.split(1, dim=-1)
        xcenter = xmin + int((w - TO_REMOVE).clamp(min=0) / 2)
        ycenter = ymin + int((h - TO_REMOVE).clamp(min=0) / 2)
        out_box.center.position.x = xcenter
        out_box.center.position.y = ycenter
        out_box.center.theta = 0
        out_box.size_x = w
        out_box.size_y = h
        return out_box
    
    def __getInstanceImage__(predictions, original_image, label_id_to_names, fade_weight=0.8, score_thr=None):
        # TODO
        """Log a single detection result for visualization.

        Inspired From: https://github.com/TRI-ML/realtime_panoptic/blob/master/realtime_panoptic/utils/visualization.py
        
        Overlays predicted classes on top of raw RGB image.

        Parameters:
        -----------
        predictions: torch.cuda.LongTensor
            Per-pixel predicted class ID's for a single input image
            Shape: (H, W)

        original_image: np.array
            HxWx3 original image. or None

        label_id_to_names: list
            list of class names for instance labels

        fade_weight: float, default: 0.8
            Visualization is fade_weight * original_image + (1 - fade_weight) * predictions

        Returns:
        -------
        visualized_image: np.array
            Visualized image with detection results.
        """

        # Load raw image using provided dataset and index
        # ``images_numpy`` has shape (H, W, 3)
        # ``images_numpy`` has shape (H, W,3)
        output_msg = Detection2DArray()
        if not isinstance(original_image, np.ndarray):
            original_image = np.array(original_image)
        original_image_height, original_image_width,_ = original_image.shape

        # Color per-pixel predictions
        predictions_numpy = predictions.cpu().numpy().astype('uint8')
        colored_predictions_numpy = colormap[predictions_numpy.flatten()]
        colored_predictions_numpy = colored_predictions_numpy.reshape(original_image_height, original_image_width, 3)

        # overlay_boxes
        visualized_image = copy.copy(np.array(original_image))

        labels = predictions.get_field("labels").to("cpu")
        boxes = predictions.bbox

        dtype = labels.dtype
        palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1]).to(dtype)
        colors = labels[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        masks = None
        if predictions.has_field("mask"):
            masks = predictions.get_field("mask")
        else:
            masks = [None] * len(boxes)
        # overlay_class_names_and_score
        if predictions.has_field("scores"):
            scores = predictions.get_field("scores").tolist()
        else:
            scores = [1.0] * len(boxes)
        # predicted label starts from 1 as 0 is reserved for background.
        label_names = [label_id_to_names[i-1] for i in labels.tolist()]

        text_template = "{}: {:.2f}"

        for box, color, score, mask, label in zip(boxes, colors, scores, masks, label_names):
            if score < score_thr:
                continue
            box = box.to(torch.int64)
            color = random_color(color)
            color = tuple(map(int, color))

            instance = Detection2D()
            instance.bbox = self.__getBBox__(box)
            
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = label
            hypothesis.hypothesis.score = score

            instance.results.append(hypothesis)
            output_msg.detections.append(instance)

            if mask is not None:
                thresh = (mask > score_thr).cpu().numpy().astype('uint8')
                visualized_image, color = draw_mask(visualized_image, thresh)

            x, y = box[:2]
            s = text_template.format(label, score)
            cv2.putText(visualized_image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            visualized_image = cv2.rectangle(visualized_image, tuple(top_left), tuple(bottom_right), tuple(color), 1)
        return visualized_image, colored_predictions_numpy, output_msg

    def __runDetectionInstance__(self, input):
        data = {'image': input}

        # pre-processing
        normalize_transform = P.Normalize(mean=self.__config_params__.input.pixel_mean,
                                            std=self.__config_params__.input.pixel_std,
                                            to_bgr255=self.__config_params__.input.to_bgr255)
        
        transform = P.Compose([
            P.ToTensor(),
            normalize_transform,
        ])

        data = transform(data)
        with torch.no_grad():
            input_image_list = ImageList([data['image'].to(self.__device__)], 
                                            image_sizes=[input.size[::-1]])
            panoptic_result, _ = self.__net__.forward(input_image_list)
            instance_detection = [o.to('cpu') for o in panoptic_result["instance_segmentation_result"]]
            if(self.__allow_score_thresholding__):
                instace_image, segmentation_image, instances = self.__getInstanceImage__(instance_detection[0], input, self.__dataset__.label_map, score_thr=self.__thr_arr__)
            else:
                instace_image, segmentation_image, instancecs = self.__getInstanceImage__(instance_detection[0], input, self.__dataset__.label_map, score_thr=self.__dataset__.score_thr)
        return instance_image, segmentation_image, instances

    def __initParams__(self):
        self.__bridge__ = CvBridge()
        try:
            self.__dataset__ = DatasetSelection(self.__config_params__)
        except RuntimeError as err:
            self.get_logger().error('%s' % err)
            break
    
    def __readModelConfig__(self):
        self.__config_params__ = cfg.merge_from_file(self.__config_file__)
        torch.cuda.set_device(self.__device_id__)
        torch.backends.cudnn.benchmark = False
        self.__device__ = torch.device('cuda:'+ str(self.__device_id__) if torch.cuda.is_available() else "cpu")
        if(self.__allow_memory_fractioning__):
            torch.cuda.set_per_process_memory_fraction(self.__device_memory_fraction__, device=self.__device_id__)
        
        self.__net__ = self.__dataset__.getNetwork()
        self.__net__ = self.__net__.to(self.__device__).eval()
        self.__net__.load_state_dict(torch.load(self.__model_file__))
    
    def __initBaseParams__(self):
        self.declare_parameter("config_file", rclpy.Parameter.Type.STRING)
        self.declare_parameter("package_folder_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("model_file", rclpy.Parameter.Type.STRING)
        self.declare_parameter("allow_memory_fractioning", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("allow_score_thresholding", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("device_id", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("device_memory_fraction", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("threshold_array", rclpy.Parameter.Type.DOUBLE_ARRAY)

        self.__config_file__ = self.get_parameter("config_file").value
        self.__package_folder_path__ = self.get_parameter("package_folder_path").value
        self.__model_file__ = self.get_parameter("model_file").value
        self.__allow_memory_fractioning__ = self.get_parameter("allow_memory_fractioning").value
        self.__allow_score_thresholding__ = self.get_parameter("allow_score_thresholding").value
        self.__device_id__ = self.get_parameter("device_id").value
        self.__device_memory_fraction__ = self.get_parameter("device_memory_fraction").value
        self.__thr_arr__ = self.get_parameter("threshold_array").value

        self.__config_file__ = self.__package_folder_path__ + "/" + self.__config_file__
        self.__model_file__ = self.__package_folder_path__ + "/" + self.__model_file__
        self.__logger__ = common_utils.create_logger()
        self.__initParams__()
        self.__readModelConfig__()
    
    def __initObjects__(self):
        self.__sub_image__ = self.create_subscription(Image,
                                                        "input",
                                                        self.__imageCB__,
                                                        10)
        self.__pub_pan_image__ = self.create_publisher(Image,
                                                    "output_image",
                                                    10)
        self.__pub_seg_image__ = self.create_publisher(Image,
                                                    "seg_image",
                                                    10)
        self.__pub_det__ = self.create_publisher(Detection2DArray,
                                                    "output_det",
                                                    10) 

def main(args=None):
    rclpy.init(args=args)
    panoptic_node = PanopticROS()
    rclpy.spin(panoptic_node)
    panoptic_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()