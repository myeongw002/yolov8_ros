#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO, NAS
from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
import cv2
import numpy as np
from rostopic import get_topic_type
import random

class Yolov8Node:

    def __init__(self):
        rospy.init_node('yolov8_node')

        # params
        self.model_type = rospy.get_param("~model_type", "YOLO")
        self.model = rospy.get_param("~model", "/home/teammiracle/ROS/yolo_ws/src/yolov8_ros/src/best5.pt")
        self.device = rospy.get_param("~device", "cuda:0")
        self.threshold = rospy.get_param("~threshold", 0.5)
        self.view_image = rospy.get_param("~view_image", True)
        self.publish_image = rospy.get_param("~publish_image", False)

        self.type_to_model = {
            "YOLO": YOLO,
            "NAS": NAS
        }

        self.yolo = self.type_to_model[self.model_type](self.model)
        self.yolo.fuse()

        self.cv_bridge = CvBridge()

        self._pub = rospy.Publisher("yolo_detections", BoundingBoxes, queue_size=10)

        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking=True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.image_cb, queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.image_cb, queue_size=1
            )

        if self.publish_image:
            self.image_pub = rospy.Publisher("annotated_image", Image, queue_size=10)

        # Create a color map for classes using provided hex colors
        hex_colors = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                      '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.colors = self.hex_to_bgr(hex_colors)
        random.shuffle(self.colors)  # Shuffle the colors to assign them randomly

    def hex_to_bgr(self, hex_colors):
        bgr_colors = []
        for hex_color in hex_colors:
            hex_color = hex_color.lstrip('#')
            bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # Convert to BGR
            bgr_colors.append(bgr_color)
        return bgr_colors

    def parse_hypothesis(self, results):
        hypothesis_list = []

        if results.boxes:
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf)
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results):
        boxes_list = []

        if results.boxes:
            for box_data in results.boxes:
                msg = BoundingBox()

                # get boxes values
                box = box_data.xyxy[0]
                msg.xmin = int(box[0])
                msg.ymin = int(box[1])
                msg.xmax = int(box[2])
                msg.ymax = int(box[3])

                # append msg
                boxes_list.append(msg)

        return boxes_list

    def image_cb(self, msg):
        # convert image + predict
        if self.compressed_input:
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        else:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Check if the image is in Bayer format and debayer it
        if cv_image is not None and len(cv_image.shape) == 2:  # Bayer images are single-channel
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2BGR)  # Adjust the Bayer pattern as needed

        cv_image = cv_image.copy()  # Make a writable copy of the image
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            device=self.device
        )
        results = results[0].cpu()
        hypothesis = self.parse_hypothesis(results) if results.boxes else []
        boxes = self.parse_boxes(results) if results.boxes else []

        # create detection msgs
        detections_msg = BoundingBoxes()

        for i in range(len(hypothesis)):
            aux_msg = BoundingBox()

            if results.boxes and hypothesis and boxes:
                aux_msg.Class = hypothesis[i]["class_name"]
                aux_msg.probability = hypothesis[i]["score"]
                aux_msg.xmin = boxes[i].xmin
                aux_msg.ymin = boxes[i].ymin
                aux_msg.xmax = boxes[i].xmax
                aux_msg.ymax = boxes[i].ymax

                # Get color for the class
                color = self.colors[hypothesis[i]["class_id"] % len(self.colors)]

                # Draw bounding box on the image
                cv2.rectangle(cv_image, (aux_msg.xmin, aux_msg.ymin), (aux_msg.xmax, aux_msg.ymax), color, 2)
                label = f"{aux_msg.Class} {aux_msg.probability:.2f}"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                text_x = aux_msg.xmin
                text_y = aux_msg.ymin - 10 if aux_msg.ymin - 10 > 10 else aux_msg.ymin + 10

                # Draw background rectangle for text
                cv2.rectangle(cv_image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), color, cv2.FILLED)

                # Calculate brightness of the color
                brightness = np.mean(color)

                # Choose text color based on brightness
                text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

                # Draw text on top of the rectangle
                cv2.putText(cv_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

            detections_msg.bounding_boxes.append(aux_msg)

        # publish detections
        detections_msg.header = msg.header
        self._pub.publish(detections_msg)

        # Publish annotated image
        if self.publish_image:
            annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
            annotated_image_msg.header = msg.header
            self.image_pub.publish(annotated_image_msg)

        # Display image
        if self.view_image:
            cv_image_resized = cv2.resize(cv_image, (1280, 720))  # Resize for better visualization
            cv2.imshow("Detection", cv_image_resized)
            cv2.waitKey(1)

        del results
        del cv_image

def main():
    node = Yolov8Node()
    rospy.spin()

if __name__ == '__main__':
    main()