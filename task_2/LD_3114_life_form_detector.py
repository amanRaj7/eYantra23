#!/usr/bin/env python3

'''
# * Team Id : #3114
# * Author List : Aman Raj, Pratyush Roshan Mallik, Gaurav Kumar Sharma, Chandan Priyadarshi
# * Filename: = life_form_detector.py
# * Theme: Luminosity Drone
# * Functions: disarm, arm, imu_callback, whycon_callback, image_callback, pid, image_pid, nav_pid, land_pid, pub_bio
# * Global Variables: end
'''

# Importing the required libraries

# from swift_msgs.msg import swift_msgs
from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import os
from imutils import contours
from skimage import measure
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from luminosity_drone.msg import Biolocation
import math
import tf
from sensor_msgs.msg import Imu



class swift():
	"""docstring for swift"""
	def __init__(self):
		
		rospy.init_node('drone_control')	# initializing ros node with name drone_control
		# initializing the drone position and orientation	
		self.drone_position = [0.0,0.0,0.0]	
		self.drone_quaternion = [0.0, 0.0, 0.0, 0.0]
		self.drone_euler = [0.0, 0.0, 0.0]

		self.setpoint = [-9.0, -10.0, 28.0]	# starting at top left		

		self.reached_start = False	# boolean whether we have reached top left or not
		self.image_on_screen = False # boolean whether the alien is on screen or not
		self.recalculate_approx = False # boolean whether to recalculate the approximate position of the alien or not
		self.land_flag = False # boolean whether to land the drone or not
		self.drone_left_to_right = True # boolean whether the drone is moving from left to right or not
		self.drone_up_to_down = False #	boolean whether the drone is moving from top to bottom or not
		self.deactivate = False # boolean whether to deactivate the drone or not
		self.cent = [] # list to store the centroid of the alien

		self.cmd = swift_msgs()
		
		# initializing the rotor values
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX3 = 1500
		self.cmd.rcAUX4 = 1500
		
		# initializing the PID values
		self.Kp = [ 30.78  , 34.5 , 63.2]
		self.Ki = [ 0.0008 , 0    ,0.0826]
		self.Kd = [ 600.5  , 579  , 655]

		# second PID values for some uses
		# self.Kp = [ 35.37  , 20.06 , 120.1]
		# self.Ki = [ 0.0009 , 0.0005    ,0.0147]
		# self.Kd = [ 808  , 153  , 829]

		# initializing the error values
		self.prev_error = [0.0, 0.0, 0.0]
		self.sum_error =  [0.0, 0.0, 0.0]

		# initializing the min and max values of the rotor
		self.min_values = [1400,1400,1400]
		self.max_values = [1550,1550,1600]

		# initializing the derivative and integral values
		self.derivative = [0.0, 0.0, 0.0]
		self.integral = [0.0, 0.0, 0.0]

		# Publishing /drone_command 
		# / Subscribe to /whycon/poses, /ardrone/bottom/image_raw, /pid_tuning_altitude, 
		# /pid_tuning_roll, /pid_tuning_pitch, /pid_tuning_yaw

		self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
		rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
		rospy.Subscriber('/swift/camera_rgb/image_raw', Image, self.image_callback)
		rospy.Subscriber('swift/imu', Imu, self.imu_callback)
		self.bio_pub = rospy.Publisher('astrobiolocation', Biolocation, queue_size = 1)
		self.bio = Biolocation()
		self.arm() 

	# function to disarm the drone
	def disarm(self):
		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1100
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)

	# function to arm the drone
	def arm(self):

		self.disarm()

		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1500
		self.command_pub.publish(self.cmd)	# Publishing /drone_command
		rospy.sleep(1)

	# function to get the current orientation of the drone
	def imu_callback(self,msg):
		
		self.drone_quaternion[0] = msg.orientation.x
		self.drone_quaternion[1] = msg.orientation.y
		self.drone_quaternion[2] = msg.orientation.z
		self.drone_quaternion[3] = msg.orientation.w
        # converting the current orientations from quaternion to euler angles 
		(self.drone_euler[0], self.drone_euler[1], self.drone_euler[2]) = tf.transformations.euler_from_quaternion([self.drone_quaternion[0], self.drone_quaternion[1], self.drone_quaternion[2], self.drone_quaternion[3]])
												
		#convert radians to degrees
		self.drone_euler[0] = self.drone_euler[0]*180/np.pi
		self.drone_euler[1] = self.drone_euler[1]*180/np.pi
		self.drone_euler[2] = self.drone_euler[2]*180/np.pi

	# function to get the current position of the drone
	def whycon_callback(self,msg):
		self.drone_position[0] = msg.poses[0].position.x
		self.drone_position[1] = msg.poses[0].position.y
		self.drone_position[2] = msg.poses[0].position.z

	# function to detect the alien and calculate the approximate position of the alien
	def image_callback(self, img_msg):
		try:
			cvimage = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
			image = cvimage
			gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			blurred = cv.GaussianBlur(gray, (5, 5), 0)
			thresh = cv.threshold(blurred, 130, 255, cv.THRESH_BINARY)[1]
			kernel = np.ones((3, 3), np.uint8)
			thresh = cv.erode(thresh, kernel, iterations=3)
			thresh = cv.dilate(thresh, kernel, iterations=3)
			labels = measure.label(thresh, background=0)

			# for debugging
			#cv.imshow('output', thresh)
			# cv.waitKey(3)

			mask = np.zeros(thresh.shape, dtype="uint8")

			for label in np.unique(labels):
				if label == 0:
					continue

			labelMask = np.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			numPixels = cv.countNonZero(labelMask)

			# debugging
			# print(numPixels)

			if numPixels > 30 and numPixels < 800: 
				mask = cv.add(mask, labelMask)

			contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			contours = sorted(contours, key=lambda c: cv.minEnclosingCircle(c)[0])

			centroid = []

			for i, contour in enumerate(contours):
				M = cv.moments(contour)
				c_X = round(M["m10"] / M["m00"],2)
				c_Y = round(M["m01"] / M["m00"],2)
				centroid.append((c_X, c_Y))
				
			if len(centroid)==0:
				return
			else:
				x_values, y_values = zip(*centroid)
				self.cent.append(round(sum(x_values)/len(centroid), 2))
				self.cent.append(round(sum(y_values)/len(centroid), 2))
				self.cent.append(35)
				self.image_on_screen = True
				self.recalculate_approx = True

		except CvBridgeError as e:
			print("CvBridge Error: {0}".format(e))
	
	# function to calculate the PID 
	def pid(self):
		self.integral = [self.sum_error[0] * self.Ki[0],
						 self.sum_error[1] * self.Ki[1],
					  	 self.sum_error[2] * self.Ki[2]
						 ]

		self.derivative = [self.error[0] - self.prev_error[0],
					  	   self.error[1] - self.prev_error[1],
						   self.error[2] - self.prev_error[2]]
		
		self.sum_error[0] += self.error[0]
		self.sum_error[1] += self.error[1]
		self.sum_error[2] += self.error[2]			

		self.cmd.rcRoll     = int(1500 - ((self.Kp[0] * self.error[0]) + (self.integral[0]) + (self.Kd[0] * self.derivative[0])))
		self.cmd.rcPitch    = int(1500 + ((self.Kp[1] * self.error[1]) + (self.integral[1]) + (self.Kd[1] * self.derivative[1])))
		self.cmd.rcThrottle = int(1550 + ((self.Kp[2] * self.error[2]) + (self.integral[2]) + (self.Kd[2] * self.derivative[2])))

		if self.cmd.rcRoll > self.max_values[0]:
				self.cmd.rcRoll = self.max_values[0]

		elif self.cmd.rcRoll < self.min_values[0]:
				self.cmd.rcRoll = self.min_values[0]

		if self.cmd.rcPitch > self.max_values[1]:
				self.cmd.rcPitch = self.max_values[1]

		elif self.cmd.rcPitch < self.min_values[1]:
				self.cmd.rcPitch = self.min_values[1]

		if self.cmd.rcThrottle > self.max_values[2]:
			self.cmd.rcThrottle = self.max_values[2]

		elif self.cmd.rcThrottle < self.min_values[2]:
			self.cmd.rcThrottle = self.min_values[2]

		self.prev_error = self.error

		self.command_pub.publish(self.cmd)

		if all(abs(e) < 0.5 for e in self.error):
			self.reached_start = True
	
	# function to calculate the approximate position of the alien
	def image_pid(self):
		os.system("clear")
		print("IMAGE PID")	
		cent_last_index = len(self.cent) - 3
		print(self.cent[cent_last_index], self.cent[cent_last_index + 1])
		print("cent length", cent_last_index + 3)
		
		if self.recalculate_approx:
			# yaw correction
			rotation_angle = self.drone_euler[2] - 90

			prev_x = (self.cent[cent_last_index]-250)
			prev_y = (self.cent[cent_last_index + 1]-250)
			
			# rotation matrix
			rotated_x = (prev_x * math.cos(math.radians(rotation_angle))) - (prev_y * math.sin(math.radians(rotation_angle)))
			rotated_y = (prev_x * math.sin(math.radians(rotation_angle))) + (prev_y * math.cos(math.radians(rotation_angle)))

			self.alien_approx_x = self.drone_position[0] + (math.tan(0.1) * self.drone_position[2] * (rotated_x / 500))
			self.alien_approx_y = self.drone_position[1] + (math.tan(0.1) * self.drone_position[2] * (rotated_y / 500))

			print("RELATIVE POS")
			print(self.cent[cent_last_index], self.cent[cent_last_index + 1])
			print(rotated_x, rotated_y)
			print(math.tan(0.2) * self.drone_position[2] * (rotated_x / 500), (math.tan(0.2) * self.drone_position[2] * (rotated_y / 500)))

			self.recalculate_approx = False

		roll_error = self.drone_position[0] - self.alien_approx_x
		pitch_error = self.drone_position[1] - self.alien_approx_y
		alt_error = self.drone_position[2]- 28

		self.error = [roll_error, pitch_error, alt_error]


		if all(abs(e) < 0.2 for e in self.error):
			print(self.drone_position[0], self.drone_position[1], 25)
			print('done')
			self.pub_bio(self.alien_approx_x, self.alien_approx_y, self.drone_position[2], 'alien_b')
			self.land_flag = True
		print(self.cent[0])

	# function to navigate the drone in a rectangular path
	def nav_pid(self):
		# os.system("clear")

		print("NAV PID")
		left_right_flag = (int(self.drone_left_to_right) * 2) - 1
		print("left_right_flag", left_right_flag)

		# move to next setpoint if under error range
		if self.reached_start:
			error = 0.1 if self.drone_up_to_down else 0.5
			print("error", error)
			if(abs(self.setpoint[0] - self.drone_position[0]) < error):	
					self.setpoint[0] = self.setpoint[0] + ((left_right_flag) * 6)
					self.drone_up_to_down = False
			if(self.setpoint[0] > 10 or self.setpoint[0] < -10): # out of bounds
				self.drone_up_to_down = True
				self.drone_left_to_right = False if self.drone_left_to_right else True
				self.setpoint[0] = left_right_flag * 9
				self.setpoint[1] += 2.5


		print("curr setpoint : ", self.setpoint[0], self.setpoint[1])

		roll_error = self.drone_position[0] - self.setpoint[0]
		pitch_error = self.drone_position[1] - self.setpoint[1]
		alt_error = self.drone_position[2] - self.setpoint[2]
		self.error = [roll_error, pitch_error, alt_error]

	# function to land the drone
	def land_pid(self):
		roll_error = self.drone_position[0] - 10.8
		pitch_error = self.drone_position[1] - 10.8
		alt_error = self.drone_position[2] - 28
		self.error = [roll_error, pitch_error, alt_error]
		
		# deactivate the drone if it is within the error range
		if all(abs(e) < 0.1 for e in self.error):
			self.disarm()
			print('deactivate')
			self.deactivate = True

	# function to publish the bio location of the alien	
	def pub_bio(self, x, y, z, org):
		self.bio.organism_type = org
		self.bio.whycon_x = x
		self.bio.whycon_y = y
		self.bio.whycon_z = z
		self.bio_pub.publish(self.bio)


if __name__ == '__main__':

	swift_drone = swift()
	r = rospy.Rate(30) #specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz
	while not rospy.is_shutdown() and not (swift_drone.deactivate):

		if(not swift_drone.image_on_screen) and (not swift_drone.land_flag):
			swift_drone.nav_pid()
			swift_drone.pid()
		elif(swift_drone.image_on_screen) and (not swift_drone.land_flag):
			swift_drone.image_pid()
			swift_drone.pid()
		else:
			swift_drone.land_pid()
			if (swift_drone.deactivate):
				break
			swift_drone.pid()

		r.sleep()
