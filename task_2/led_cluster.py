#!/usr/bin/env python3

from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2 as cv
import rospy 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from swift_msgs.msg import swift_msgs
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64


rospy.init_node('drone_control')

bridge = CvBridge()
cv_image = None
global centroids
centroids = []
#centroid= []
new_setpoint = None

def run_code(cv_image):
    image = cv_image
    #print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 130, 255, cv.THRESH_BINARY)[1]

    thresh = cv.erode(thresh, None, iterations=1)
    thresh = cv.dilate(thresh, None, iterations=2)
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv.countNonZero(labelMask)
        #print(numPixels)

        if numPixels > 30 and numPixels < 300: 
            mask = cv.add(mask, labelMask)

    contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda c: cv.minEnclosingCircle(c)[0])

    centroid = []
    area = []

    for i, contour in enumerate(contours):
        areas = cv.contourArea(contour)
        M = cv.moments(contour)
        c_X = round(M["m10"] / M["m00"],2)
        c_Y = round(M["m01"] / M["m00"],2)
        cX = int(c_X)
        cY = int(c_Y)

        cv.drawContours(image, [contour], -1, (0, 0, 255), 2)
        image = cv.putText(image, f'{i + 1}', (cX -30, cY - 25), cv.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 255), 1)

        centroid.append((c_X, c_Y))
        area.append(areas)

        a = len(centroid)
        
    global centroids
    centroids = centroid
    # if len(centroid) > 0:
    #     print(centroid)

    cv.namedWindow('output', cv.WINDOW_NORMAL)
    cv.imshow('output', image)
    
    cv.waitKey(3)

def image_callback(img_msg):
    global cv_image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        run_code(cv_image)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

sub_image = rospy.Subscriber("/swift/camera_rgb/image_raw", Image, image_callback)
#sub_image = rospy.Subscriber("/whycon/image_out", Image, image_callback)   

def next_setpoint(coordinate):
    x,y = coordinate
    x_n = round((x-640)/80, 2)
    y_n = round((y - 640)/80 ,2)
    z_n = 20
    new_setpoint = [x_n,y_n,z_n]
    print(new_setpoint)
    return new_setpoint
    #pid_control(new_setpoint)
    

def pid_control(setpoint):
    rospy.init_node('drone_control')

    drone_position = [0.0, 0.0, 0.0]

    cmd = swift_msgs()
    cmd.rcRoll = 1500
    cmd.rcPitch = 1500
    cmd.rcYaw = 1500
    cmd.rcThrottle = 1500
    cmd.rcAUX1 = 1500
    cmd.rcAUX2 = 1500
    cmd.rcAUX3 = 1500
    cmd.rcAUX4 = 1500
    if setpoint == [0, 0, 8]:
        Kp = [30.78, 34.5, 104.8]
        Ki = [0.0008, 0, 0.0427]
        Kd = [600.5, 579, 1583]
    else:
        Kp = [30.78, 34.5, 63.2]
        Ki = [0.0008, 0, 0.0826]
        Kd = [600.5, 579, 655]

    prev_error = [0.0, 0.0, 0.0]
    sum_error = [0.0, 0.0, 0.0]

    min_values = [1350, 1350, 1000]
    max_values = [1700, 1700, 2000]

    derivative = [0.0, 0.0, 0.0]
    integral = [0.0, 0.0, 0.0]

    command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)

    def disarm():
        cmd.rcRoll = 1500
        cmd.rcYaw = 1500
        cmd.rcPitch = 1500
        cmd.rcThrottle = 1000
        cmd.rcAUX4 = 1100
        command_pub.publish(cmd)
        rospy.sleep(1)

    def arm():
        disarm()
        cmd.rcRoll = 1500
        cmd.rcYaw = 1500
        cmd.rcPitch = 1500
        cmd.rcThrottle = 1000
        cmd.rcAUX4 = 1500
        command_pub.publish(cmd)
        rospy.sleep(1)

    def whycon_callback(msg):
        global drone_position_set
        drone_position_set = drone_position
        drone_position[0] = msg.poses[0].position.x
        drone_position[1] = msg.poses[0].position.y
        drone_position[2] = msg.poses[0].position.z

    def calculate_pid():
        nonlocal prev_error
                
        error =     [drone_position[i] - setpoint[i] for i in range(3)]

        integral = [sum_error[i] * Ki[i] for i in range(3)]

        derivative = [error[i] - prev_error[i] for i in range(3)]


        sum_error[0] += error[0]
        sum_error[1] += error[1]
        sum_error[2] += error[2]

        cmd.rcRoll = int(1500 - ((Kp[0] * error[0]) + (integral[0]) + (Kd[0] * derivative[0])))
        cmd.rcPitch = int(1500 + ((Kp[1] * error[1]) + (integral[1]) + (Kd[1] * derivative[1])))
        cmd.rcThrottle = int(1550 + ((Kp[2] * error[2]) + (integral[2]) + (Kd[2] * derivative[2])))

        cmd.rcRoll = max(min(cmd.rcRoll, max_values[0]), min_values[0])
        cmd.rcPitch = max(min(cmd.rcPitch, max_values[1]), min_values[1])
        cmd.rcThrottle = max(min(cmd.rcThrottle, max_values[2]), min_values[2])

        prev_error = error

        command_pub.publish(cmd)

        global loop_flag
        global loop_setpoint
        global count
        global station
        if all(abs(e) < 0.1 for e in error) and (loop_flag) and setpoint==[0, 0, 8]:
            #print(centroids[0])
            count += 1
            loop_setpoint = next_setpoint(centroids[0])
            loop_flag = False
        if all(abs(e) < 0.2 for e in error) and setpoint!=[0, 0, 8] and setpoint!=[11, 11, 37]:
            #add transmission code here
            station = True
        if all(abs(e) < 0.2 for e in error) and setpoint==[11, 11, 37]:
            disarm()
            

    arm()
    rospy.Subscriber('whycon/poses', PoseArray, whycon_callback)
    
    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        
        global flag_set
        if loop_setpoint!=None and flag_set:
                setpoint = loop_setpoint
                flag_set = False
        if station:
            setpoint = [11, 11, 30]

        calculate_pid()
        r.sleep()
global flag_set
global loop_flag 
global loop_setpoint
global count
global station
station = False
count = 0
loop_setpoint = None
loop_flag = True
flag_set  = True
if __name__ == '__main__':
    # Example usage of the pid_control function with a custom setpoint:
    # custom_setpoint = [-3.77, 4.43, 25]
    custom_setpoint = [0, 0, 8]

    pid_control(custom_setpoint)
    pid_control(next_setpoint(centroids[0]))
    # if flag:
    #     pid_control(next_setpoint(centroids[0]))

# Keep the ROS node running
#rospy.spin()
        
