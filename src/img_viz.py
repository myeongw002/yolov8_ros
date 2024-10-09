#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np




def callback(data):
    # Convert the compressed image to a numpy array
    np_arr = np.fromstring(data.data, np.uint8)
    # Decode the image
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Display the image
    cv2.imshow('Compressed Image', image_np)
    cv2.waitKey(1)

def main():
    rospy.init_node('image_listener', anonymous=True)
    rospy.Subscriber("/pylon_camera_node/image_raw/compressed", CompressedImage, callback)
    rospy.spin()

if __name__ == '__main__':
    main()