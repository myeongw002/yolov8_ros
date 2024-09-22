#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rostopic import get_topic_type



def publish_image():
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('image_topic', Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(10)  # 10 Hz

    # 이미지 파일 경로
    image_path = '/home/teammiracle/ROS/yolo_ws/src/yolov8_ros/src/000098_jpg.rf.nWiuZeqkH4gfqqcWHymm.jpg'
    cv_image = cv2.imread(image_path)

    if cv_image is None:
        rospy.logerr("Could not read image from path: {}".format(image_path))
        return

    while not rospy.is_shutdown():
        image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        image_pub.publish(image_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass