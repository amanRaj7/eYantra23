#!/usr/bin/env python3
'''
# * Team Id : #3114
# * Author List : Aman Raj, Pratyush Roshan Mallik, Gaurav Kumar Sharma, Chandan Priyadarshi
# * Filename: = led_detection.py
# * Theme: Luminosity Drone
# * Functions: NONE
# * Global Variables: NONE
'''

# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import rospy 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('opencv_example', anonymous=True)

# load the image, 
bridge = CvBridge()
cv_image = None


def run_code(cv_image):
    # print('in loop')
    image = cv_image
    gray = image
    # print(image._type)
    # Convert it to grayscale, and blur it
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to reveal light regions in the blurred image
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
    kernel = np.ones((1, 1), np.uint8)
    dilation = cv2.dilate(cv2.erode(thresh, kernel, iterations=2), kernel, iterations=2)
    cv2.imshow('output', dilation)
    # Perform connected component analysis on the thresholded image
    labels = measure.label(dilation, background=0)
    mask = np.zeros(dilation.shape, dtype="uint8")

    # Loop over the unique components
    for i in np.unique(labels):
        if i == 0:
            continue

        labelMask = np.zeros(dilation.shape, dtype="uint8")
        labelMask[labels != i] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > 300:
            mask = cv2.add(mask, labelMask)

    # Find contours in the mask and sort them from left to right
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Initialize lists to store centroid coordinates and area
    centroids = []
    areas = []

    # Loop over the contours
    for i, contour in enumerate(contours):
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Find the centroid of the contour
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Draw the bright spot on the image
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
        cv2.putText(image, f'#{i + 1}', (cX - 15, cY - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Append centroid coordinates and area to the respective lists
        centroids.append((cX, cY))
        areas.append(area)
    print(centroids, areas)

    cv2.imshow('output1', image)
    cv2.waitKey(3)

def image_callback(img_msg):
    # print('hi')
    global cv_image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "mono8")
        # print('decoded')
        run_code(cv_image)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

# Subscribe to the image topic and wait for data
sub_image = rospy.Subscriber("/swift/camera_rgb/image_raw", Image, image_callback)
rospy.spin()
