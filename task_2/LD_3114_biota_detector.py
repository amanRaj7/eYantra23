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

		self.alien = 0
		self.reached_start = False	# boolean whether we have reached top left or not
		self.image_on_screen = False # boolean whether the alien is on screen or not
		self.recalculate_approx = False # boolean whether to recalculate the approximate position of the alien or not
		self.land_flag = False # boolean whether to land the drone or not
		self.drone_left_to_right = True # boolean whether the drone is moving from left to right or not
		self.drone_up_to_down = False #	boolean whether the drone is moving from top to bottom or not
		self.deactivate = False # boolean whether to deactivate the drone or not
		self.cent = [] # list to store the centroid of the alien
		self.stop_image = True # boolean whether to stop the image detection or not
		self.cmd = swift_msgs()

		self.visited_set = []   # set to store the visited coordinates
		

		self.curr_setpoint = [0.0, 0.0, 0.0] # current setpoint of the drone

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
		# / Subscribe to /whycon/poses, imu , etc

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
			blurred = cv.GaussianBlur(gray, (11, 11), 0)

			# threshold the image to reveal light regions in the blurred image
			thresh = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)[1]

			# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
			thresh = cv.erode(thresh, None, iterations=2)
			thresh = cv.dilate(thresh, None, iterations=4)

			# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
			labels = measure.label(thresh, background=0)
			mask = np.zeros(thresh.shape, dtype="uint8")

			# loop over the unique components
			for label in np.unique(labels):
				# if this is the background label, ignore it
				if label == 0:
					continue

				# otherwise, construct the label mask and count the number of pixels 
				labelMask = np.zeros(thresh.shape, dtype="uint8")
				labelMask[labels == label] = 255
				numPixels = cv.countNonZero(labelMask)
				#print(numPixels)

				# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
				if numPixels > 300: 
					mask = cv.add(mask, labelMask)

			# find the contours in the mask, then sort them from left to right
			contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

			# loop over the contours
			contours = sorted(contours, key=lambda c: cv.minEnclosingCircle(c)[0])

			# Initialize lists to store centroid coordinates and area
			centroid = []

			# Loop over the contours
			for i, contour in enumerate(contours):
				# Calculate the area of the contour
				areas = cv.contourArea(contour)
				M = cv.moments(contour)
				c_X = round(M["m10"] / M["m00"],10)
				c_Y = round(M["m01"] / M["m00"],10)
				cX = int(c_X)
				cY = int(c_Y)

				# Append centroid coordinates and area to the respective lists
				centroid.append((c_X, c_Y))

			# print("alien no ",len(centroid))
			self.alien = len(centroid)
			if len(centroid)==0:
				return
			else:
				x_values, y_values = zip(*centroid)
				self.cent.append(round(sum(x_values)/len(centroid), 2))
				self.cent.append(round(sum(y_values)/len(centroid), 2))
				self.cent.append(35)
				if self.stop_image:
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

		self.command_pub.publish(self.cmd)  # Publishing /drone_command

		if all(abs(e) < 0.5 for e in self.error):
			self.reached_start = True
	
	# function to calculate the approximate position of the alien
	def image_pid(self):
		prev_x = (self.cent[-3] - 250)
		prev_y = (self.cent[-2] - 250)
		if self.recalculate_approx:
			rotation_angle = self.drone_euler[2] - 90
			rotated_x = (prev_x * math.cos(math.radians(rotation_angle))) - (prev_y * math.sin(math.radians(rotation_angle)))
			rotated_y = (prev_x * math.sin(math.radians(rotation_angle))) + (prev_y * math.cos(math.radians(rotation_angle)))

			self.alien_approx_x = self.drone_position[0] + (math.tan(0.1) * self.drone_position[2] * (rotated_x / 500))
			self.alien_approx_y = self.drone_position[1] + (math.tan(0.1) * self.drone_position[2] * (rotated_y / 500))

			print("RELATIVE POS")
			self.recalculate_approx = False

		roll_error = self.drone_position[0] - self.alien_approx_x
		pitch_error = self.drone_position[1] - self.alien_approx_y
		alt_error = self.drone_position[2] - 28
		self.error = [roll_error, pitch_error, alt_error]

		# publish when drone is within error range of calculated position
		if all(abs(e) < 0.2 for e in self.error):
			alreadyFound = False
			for coord in self.visited_set:
				if math.dist(coord, [self.alien_approx_x, self.alien_approx_y]) < 3.0:
					alreadyFound = True
			if not alreadyFound:
				self.visited_set.append([self.alien_approx_x, self.alien_approx_y])
				print(self.alien_approx_x, self.alien_approx_y, 28)
				print('done')
				if self.alien == 2:
					alien_type = 'alien_a'
				elif self.alien == 3:
					alien_type = 'alien_b'
				elif self.alien == 4:
					alien_type = 'alien_c'
				self.pub_bio(self.alien_approx_x, self.alien_approx_y, self.drone_position[2], alien_type)
			# Additional logic for continuing the search
			swift_drone.next()
			self.image_on_screen = False  # Reset the flag to continue searching
			self.stop_image = False

		# publish when alien is almost in center of frame
		elif abs(prev_x) < 4 and abs(prev_y) < 4:
			alreadyFound = False
			for coord in self.visited_set:
				if math.dist(coord, [self.alien_approx_x, self.alien_approx_y]) < 3.0:
					alreadyFound = True
			if not alreadyFound:
				self.visited_set.append([self.alien_approx_x, self.alien_approx_y])
				print(self.alien_approx_x, self.alien_approx_y, 28)
				print('done')
				if self.alien == 2:
					alien_type = 'alien_a'
				elif self.alien == 3:
					alien_type = 'alien_b'
				elif self.alien == 4:
					alien_type = 'alien_c'
				self.pub_bio(self.alien_approx_x, self.alien_approx_y, self.drone_position[2], alien_type)
			# Additional logic for continuing the search
			swift_drone.next()
			self.image_on_screen = False  # Reset the flag to continue searching
			self.stop_image = False
			
		# reached end now land drone
		if(self.setpoint[1] > 9):
			self.land_flag = True

	# function to navigate the drone in a rectangular path
	def nav_pid(self):
		swift_drone.next()
		self.curr_setpoint = self.setpoint

		roll_error = self.drone_position[0] - self.setpoint[0]
		pitch_error = self.drone_position[1] - self.setpoint[1]
		alt_error = self.drone_position[2] - self.setpoint[2]
		self.error = [roll_error, pitch_error, alt_error]


	# function to move to the next setpoint
	def next(self): 
		left_right_flag = (int(self.drone_left_to_right) * 2) - 1
	
		if(self.setpoint[1] > 9):  # reached end now land drone
			self.land_flag = True

		# move to next setpoint if under error range
		if self.reached_start:
			error = 0.1 if self.drone_up_to_down else 0.5

			if(abs(self.setpoint[0] - self.drone_position[0]) < error):	 # go to next setpoint
					self.setpoint[0] = self.setpoint[0] + ((left_right_flag) * 6)
					self.drone_up_to_down = False
					self.stop_image = True

			if(self.setpoint[0] > 10 or self.setpoint[0] < -10): # out of bounds
				self.drone_up_to_down = True
				self.drone_left_to_right = False if self.drone_left_to_right else True
				self.setpoint[0] = left_right_flag * 9
				self.setpoint[1] += 3


	# function to land the drone
	def land_pid(self):
		roll_error = self.drone_position[0] - 10.9
		pitch_error = self.drone_position[1] - 10.8
		alt_error = self.drone_position[2] - 28
		self.error = [roll_error, pitch_error, alt_error]
		
		# deactivate the drone if it is within the error range
		if abs(roll_error)<0.1 and abs(pitch_error)<0.1 and abs(alt_error)<0.2:
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
		# navigate normally when alien is not on screen
		if(not swift_drone.image_on_screen) and (not swift_drone.land_flag):
			swift_drone.nav_pid()
			swift_drone.pid()
		# navigate and detect alien when alien is on screen
		elif(swift_drone.image_on_screen) and (not swift_drone.land_flag):
			swift_drone.image_pid()
			swift_drone.pid()
		# land the drone when the drone reaches the end
		else:
			swift_drone.land_pid()
			if (swift_drone.deactivate):
				break
			swift_drone.pid()

		r.sleep()
