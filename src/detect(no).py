#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np
from rostopic import get_topic_type
import random
from concurrent.futures import ThreadPoolExecutor

class Yolov8Node:

    def __init__(self):
        rospy.init_node('yolov8_node')

        # Parameters
        self.model_path = rospy.get_param("~model", "/home/teammiracle/ROS/yolo_ws/src/yolov8_ros/src/best8.pt")
        self.device = rospy.get_param("~device", "cuda:0")
        self.threshold = rospy.get_param("~threshold", 0.5)
        self.view_image = rospy.get_param("~view_image", True)
        self.publish_image = rospy.get_param("~publish_image", False)

        # Load YOLOv8 model using SAHI's AutoDetectionModel
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=self.model_path,
            confidence_threshold=self.threshold,
            device=self.device
        )

        # Initialize OpenCV bridge and publishers
        self.cv_bridge = CvBridge()
        self._pub = rospy.Publisher("yolo_detections", BoundingBoxes, queue_size=10)

        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking=True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(input_image_topic, CompressedImage, self.image_cb, queue_size=1)
        else:
            self.image_sub = rospy.Subscriber(input_image_topic, Image, self.image_cb, queue_size=1)

        if self.publish_image:
            self.image_pub = rospy.Publisher("annotated_image", Image, queue_size=10)

        # Create a color map for bounding boxes
        hex_colors = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB')
        self.colors = self.hex_to_bgr(hex_colors)
        random.shuffle(self.colors)

        # Thread pool for multithreading
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers for your CPU

    def hex_to_bgr(self, hex_colors):
        """Convert hex colors to BGR for OpenCV."""
        bgr_colors = []
        for hex_color in hex_colors:
            hex_color = hex_color.lstrip('#')
            bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
            bgr_colors.append(bgr_color)
        return bgr_colors

    def image_cb(self, msg):
        """Callback function to process image messages and run YOLOv8 predictions."""
        if self.compressed_input:
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        else:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Run multithreaded sliced prediction
        future = self.executor.submit(self.run_sliced_inference, cv_image, msg)
        future.add_done_callback(self.publish_results)

    def run_sliced_inference(self, cv_image, msg):
        """Run sliced inference in parallel."""
        result = get_sliced_prediction(
            image=cv_image,
            detection_model=self.detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        return (result, cv_image, msg)

    def publish_results(self, future):
        """Publish bounding boxes and annotate images when multithreaded inference is complete."""
        result, cv_image, msg = future.result()

        detections_msg = BoundingBoxes()
        detections_msg.header = msg.header

        # Loop through predictions and visualize them
        for object_prediction in result.object_prediction_list:
            box = object_prediction.bbox
            bbox_msg = BoundingBox()
            bbox_msg.xmin, bbox_msg.ymin = int(box.minx), int(box.miny)
            bbox_msg.xmax, bbox_msg.ymax = int(box.maxx), int(box.maxy)
            bbox_msg.Class = object_prediction.category.name
            bbox_msg.probability = object_prediction.score.value

            # Draw the bounding box
            color = self.colors[object_prediction.category.id % len(self.colors)]
            cv2.rectangle(cv_image, (bbox_msg.xmin, bbox_msg.ymin), (bbox_msg.xmax, bbox_msg.ymax), color, 2)
            label = f"{bbox_msg.Class} {bbox_msg.probability:.2f}"

            # Draw text label
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(cv_image, label, (bbox_msg.xmin, bbox_msg.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            detections_msg.bounding_boxes.append(bbox_msg)

        # Publish detections
        self._pub.publish(detections_msg)

        # Publish annotated image
        if self.publish_image:
            annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
            annotated_image_msg.header = msg.header
            self.image_pub.publish(annotated_image_msg)

        # Optionally display the image
        if self.view_image:
            cv_image_resized = cv2.resize(cv_image, (1280, 720))
            cv2.imshow("Detection", cv_image_resized)
            cv2.waitKey(1)

def main():
    node = Yolov8Node()
    rospy.spin()

if __name__ == '__main__':
    main()

