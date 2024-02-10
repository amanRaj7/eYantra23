#!/usr/bin/env python3

""" Importing the required libraries """
# Import - 1 (trash)
"""
Date : 10-2-2024
Author : Unkown
Reason : As we import led_detection.py, I hope that we need to install the required libraries for the same.
"""
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2 as cv
import os
import argparse

# Import - 2 (standard imports)
"""
Date : 10-02-2024
Author : Unkown
Reason : we felt it required to import the required libraries for receive image and doing maths.
"""
import math
import copy
import time
import cv2 as cv
from cv_bridge import CvBridge
import threading

# Import - 3 (third-party imports)
"""
Date : 10-02-2024
Author : Unkown
Reason : Given by e-Yantra
"""
import scipy.signal
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PidTune
from swift_msgs.msg import PIDError, RCMessage
from swift_msgs.srv import CommandBool
from sensor_msgs.msg import Image # cross check
from loc_msg.msg import Biolocation
import tf_transformations
# Import - 4 (local imports (defined by us))
"""
Date : 10-02-2024
Author : Unkown
Reason : For simplicity, we use the local imports.
"""
import json
import led_detection as led

# Constant Variable
"""
Last Update Date: 10-02-2024
Author : Unkown
Reason : define the pitch's parameter value as it cann't travel very in x - axis
Proposal : I think to reduce the the range of roll value min_roll = 1485 and max_roll = 1500 as system become slow and steady
"""
MIN_ROLL = 1400
BASE_ROLL = 1490
MAX_ROLL = 1600
SUM_ERROR_ROLL_LIMIT = 30
"""
Last Update Date: 10-02-2024
Author : Unkown
Reason : define the pitch's parameter value as it cann't travel very in y - axis
Proposal : I think to reduce the the range of roll value min_pitch = 1485 and max_pitch = 1500 as system become slow and steady
"""
MIN_PITCH = 1400
BASE_PITCH = 1480
MAX_PITCH = 1600
SUM_ERROR_PITCH_LIMIT = 30
"""
Last Update Date: 10-02-2024
Author : Unkown
Reason : define the throttle's parameter value as it cann't travel very in z - axis
Proposal : Look Nice no update required
"""
MIN_THROTTLE = 1420
BASE_THROTTLE = 1460
MAX_THROTTLE = 1520
SUM_ERROR_THROTTLE_LIMIT = 10000

# LANDING MACROS
"""
Last Update Date: 10-02-2024
Author : Unkown
Reason : Safe Landing
Proposal : Reduce landing_decrement
"""
LANDING_TRIGGER_ITERATION = 500
LANDING_DECREMENTER = 1

# Global Variable 
"""
Last Update Date: None
Author : Unknown
Reason : Unkown
Proposal : Recheck it
"""
DRONE_WHYCON_POSE = [[], [], []]


class DroneController():
	""" Controller Class """
	def __init__(self,node):
		"""
		Last Update Date : 10-02-2024
		Author : team-LD_3114
		Reason : Intialize the Class variable
		Update(reverse date wise) : led, led_count, buzzer, buzzer_count, bridge, image, all error types,  landing_variables, rest import variables
		Garbage (Recheck and remove it) : new_speed, last_whycon_pose_received_at, changeit, drone_quaternion, drone_euler, landing_iteration_count
		"""

		# new update
		self.led = True
		self.led_count = 0
		self.buzzer = True
		self.buzzer_count = 0
		self.new_speed = [BASE_ROLL, BASE_PITCH, BASE_THROTTLE]
		self.last_whycon_pose_received_at = 0
		
		# iteration count
		self.changeit = 0
		self.it = 0 # pid loop count

		# loading Arena for map transversal
		self.set_points = [[0,0,28]]         # adding dummy value 0   
		with open('arena_mapper.json', 'r') as file:
			json_data = json.load(file)
		for key in json_data:
			self.set_points.append(json_data[key])
		self.set_points.append([-10.7, -10.2, 29]) # adding dummy value 1
		self.set_points.append([-10.7, -10.2, 31]) # adding dummy value 2
		self.set_point_current_index = 0 

		# Import variable
		self.bridge = CvBridge()
		self.image = None
		self.node = node
		self.rc_message = RCMessage()
		self.drone_whycon_pose_array = PoseArray()
		self.commandbool = CommandBool.Request()
		service_endpoint = "/swift/cmd/arming"
		self.arming_service_client = self.node.create_client(CommandBool,service_endpoint)
		# Error for roll, pitch and throttle
		self.error = [0, 0, 0]         
		self.drone_position = [0,0,0]
		self.integral_error = [0, 0, 0]
		self.derivative_error = [0, 0, 0]
		self.previous_error = [0, 0, 0]
		self.sum_error = [0, 0, 0]
		self.drone_quaternion = [0,0,0,0] # NEW and UNKOWN
		self.drone_euler = [0,0,0] # NEW and UNKOWN
		"""
		PID VALUE
		Last Changed Date: 30-01-2024
		Proposal : Need some changes
		"""
		self.Kp = [ 500 * 0.01  , 500 * 0.01  ,  615 * 0.01  ]
		self.Ki = [ 190 * 0.0001  , 230 * 0.0001  , 230* 0.0001  ]
		self.Kd = [ 1000 * 0.4   , 1000 * 0.4   , 1840 * 0.1    ]


		# ALIEN DETECTION CODE
		self.alien_detect = False				# boolean is alien on screen or not
		self.alien_centroid = [0,0]				# location of aliens
		self.alien_count = 0					# number of aliens 
		self.alien_visited_set = []				# alien visited set
		self.alien_published = False			# alien pulished flag
		self.drone_camera_pixel_x = 640 	# range of camera pixels	
		self.drone_camera_pixel_y = 480		# range of camera pixels
		self.switch_to_alien_flag = False # switch flag help to change between nav_pid to image_pid
		self.switch_to_normal_flag = False # switch flag help to change between nav_pid to image_pid

		# LANDING VARIABLES
		self.land_iteration_count = 0
		self.landing_decrement_sum = 0
		self.is_landing = False

		# Create subscriber for WhyCon 
		self.whycon_sub = node.create_subscription(PoseArray,"/whycon/poses",self.whycon_poses_callback,1)
		
		# Similarly create subscribers for pid_tuning_altitude, pid_tuning_roll, pid_tuning_pitch and any other subscriber if required
		self.pid_alt = node.create_subscription(PidTune,"/pid_tuning_altitude",self.pid_tune_throttle_callback,1)
		self.pid_roll = node.create_subscription(PidTune,"/pid_tuning_roll",self.pid_tune_roll_callback,1)
		self.pid_pitch = node.create_subscription(PidTune,"/pid_tuning_pitch",self.pid_tune_pitch_callback,1)
		self.image = node.create_subscription(Image,"/video_frames",self.led_detection_callback,1)

		# Create publisher for sending commands to drone 
		self.rc_pub = node.create_publisher(RCMessage, "/swift/rc_command",1)

		# Create publisher for publishing errors for plotting in plotjuggler 
		self.pid_error_pub = node.create_publisher(PIDError, "/luminosity_drone/pid_error",1) 
		self.error_pub = node.create_publisher(PIDError, "/luminosity_drone/error",1)        
		self.filter_pub = node.create_publisher(PIDError, "/luminosity_drone/filter",1)
		self.alien_pub = node.create_publisher(Biolocation,"/astrobiolocation",1)

	def whycon_poses_callback(self, msg):
		"""
		Last update Date : 10-02-2024
		Author : team-LD_3114
		Changes : adding orienations, eular (not working)
		proposal : remove quaternion, and euler or rectify it 
		"""
		self.last_whycon_pose_received_at = self.node.get_clock().now().seconds_nanoseconds()[0] 
		self.drone_whycon_pose_array = msg 
		self.drone_quaternion[0] = msg._poses[0].orientation.x
		self.drone_quaternion[1] = msg._poses[0].orientation.y
		self.drone_quaternion[2] = msg._poses[0].orientation.z
		self.drone_quaternion[3] = msg._poses[0].orientation.w
		# converting the current orientations from quaternion to euler angles 
		(self.drone_euler[0], self.drone_euler[1], self.drone_euler[2]) = tf_transformations.euler_from_quaternion([self.drone_quaternion[0], self.drone_quaternion[1], self.drone_quaternion[2], self.drone_quaternion[3]])
		self.drone_euler[0] = self.drone_euler[0]*180/np.pi
		self.drone_euler[1] = self.drone_euler[1]*180/np.pi
		self.drone_euler[2] = self.drone_euler[2]*180/np.pi   
		
	def led_detection_callback(self, msg):
		"""
		Last update Date: 09-02-2024
		Author : team-LD_3114
		Changes : (No changes)
		Proposal : (Working) but we can remove cv.imshow and cv.waitKey
		"""
		
		self.image = self.bridge.imgmsg_to_cv2(msg) # detect the image and assign self.image type cv_image
		cv.imshow("image", self.image) # show image
		cv.waitKey(1) # wait for 1 milliseconds

		thresh = led.image_processing(self.image) # change the image in binary form
		mask = led.image_masking(thresh) # little bit image change by adding and removing noise, increase contract and brightnes
		contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # find the contours in the mask
		contours = sorted(contours, key=lambda c: cv.minEnclosingCircle(c)[0]) # Sort the contours from left to right
		if len(contours)!=0:
			# find a centeroid and types of alien (number of leds in cluster)
			self.alien_detect = True # help to change the order of code exectution by stoping or starting new function
			led_clusters, clusters = led.centroid_calculation(mask) # cluster of leds, and its ceteroids
			self.alien_num = len(led_clusters[0]) # number of leds in cluster
			self.alien_centroid[0] = clusters[0][0] # centeroid x 
			self.alien_centroid[1] = clusters[0][1] # centeroid y
		else:
			self.alien_detect = False # disable the function like image_nav, others

	"""
	Last Update Date : Unkown
	Author : e-Yantra
	Reason : tune by slider
	Proposal : we might disable this function
	"""
	def pid_tune_throttle_callback(self, alt):
		self.Kp[2] = alt.kp * 0.01
		self.Ki[2] = alt.ki * 0.0001
		self.Kd[2] = alt.kd * 0.1

	def pid_tune_roll_callback(self, roll):
		self.Kp[0] = roll.kp * 0.01
		self.Ki[0] = roll.ki * 0.0001
		self.Kd[0] = roll.kd * 0.1

	def pid_tune_pitch_callback(self, pitch):
		self.Kp[1] = pitch.kp * 0.01
		self.Ki[1] = pitch.ki * 0.0001
		self.Kd[1] = pitch.kd * 0.1

	def butter_position(self, roll, pitch, throttle):
		"""
		Last Update Date : 28-01-2024
		Author : team-LD_3114
		Reason : Reduce the noise in received whycon poses
		Proposal : we can remove as it not look like it working correctly or in best shape
		"""
		# apply the low pass filter 
		alpha = 1
		beta = 0.9
		self.drone_position[0] = alpha*roll+ (1-alpha)*self.drone_position[0]
		self.drone_position[1] = alpha*pitch+ (1-alpha)*self.drone_position[1]
		self.drone_position[2] = beta*throttle+ (1-beta)*self.drone_position[2]
		return self.drone_position
		
	def pid(self):
		"""
		Last Changed Date: 11-02-2024
		Author : team-LD_3114
		Reason : Stabilize the drone 
		Proposal : (working)
		Frequency : 30Hz 
		Changes : adding switch_to_alien codes (not working properly), remove the chances of getting negative (no need), 
		buzzer (changes required), led (changes required)
		"""
		print("NORMAL PID")

		# sum error goes to 0 as we change from image pid to nav pid (changes required)
		self.switch_to_alien_flag = True
		if self.switch_to_normal_flag:
			self.sum_error[0] = 0
			self.sum_error[1] = 0
			self.switch_to_normal_flag = False
		self.it = self.it+1

		# ERROR 
		try:
			position = self.butter_position(roll = self.drone_whycon_pose_array._poses[0].position.x, pitch = self.drone_whycon_pose_array.poses[0].position.y , throttle = self.drone_whycon_pose_array.poses[0].position.z)

			self.error[0] = position[0] - self.set_points[self.set_point_current_index][0]
			self.error[1] = position[1] - self.set_points[self.set_point_current_index][1]
			self.error[2] = position[2] - self.set_points[self.set_point_current_index][2]
		except:
			pass


		# DERIVATIVE ERROR
		self.derivative_error[0] = (self.error[0] - self.previous_error[0])
		self.derivative_error[1] = (self.error[1] - self.previous_error[1])
		self.derivative_error[2] = (self.error[2] - self.previous_error[2])

		# Sum Error
		self.sum_error[0] = self.sum_error[0]+self.error[0]
		self.sum_error[1] = self.sum_error[1]+self.error[1]
		self.sum_error[2] = self.sum_error[2]+self.error[2]

		# Integral Errror
		self.integral_error[0] = self.sum_error[0] * self.Ki[0]
		self.integral_error[1] = self.sum_error[1] * self.Ki[1]
		self.integral_error[2] = self.sum_error[2] * self.Ki[2]

		# Calculated speed according to errors
		ROLL        = int(BASE_ROLL     + ((self.Kp[0] * self.error[0]) + (self.integral_error[0]) + (self.Kd[0] * self.derivative_error[0])))
		PITCH       = int(BASE_PITCH    - ((self.Kp[1] * self.error[1]) + (self.integral_error[1]) + (self.Kd[1] * self.derivative_error[1])))
		THROTTLE    = int(BASE_THROTTLE + ((self.Kp[2] * self.error[2]) + (self.integral_error[2]) + (self.Kd[2] * self.derivative_error[2]))) 
		
		# Remove the chances of going to negative (remove this)
		if ROLL < 0:
			ROLL = BASE_ROLL
		if PITCH < 0:
			PITCH = BASE_PITCH
		if THROTTLE < 0:
			THROTTLE = BASE_THROTTLE
		
		# assign previous error to current error (ooph, most valuable 3 lines)
		self.previous_error[0] = self.error[0]
		self.previous_error[1] = self.error[1]
		self.previous_error[2] = self.error[2]

		# HOVER and change to next set point
		error_limit = 0.8  # error range
		roll_hower_flag     = self.error[0] <  error_limit and self.error[0] > -error_limit # roll's flag
		pitch_hower_flag    = self.error[1] <  error_limit and self.error[1] > -error_limit # pitch's flag
		throttle_hower_flag = self.error[2] <  error_limit and self.error[2] > -error_limit # throttle's flag
		if (roll_hower_flag and pitch_hower_flag and throttle_hower_flag) :
			# change the set point and make sum_error to 0
			self.set_point_current_index =  self.set_point_current_index + 1
			self.sum_error[0] = 0
			self.sum_error[1] = 0
			
		# landing triggeer
		if(self.set_point_current_index > len(self.set_points)-1):
			self.is_landing = True
		
		# buzzer
		buzzer = 1000 # low beep
		if self.alien_published and self.buzzer: # trigger when alien published done
			self.buzzer_count = self.buzzer_count - 1 # timer according to loop counter
			buzzer = 2000 # high beep
			if self.buzzer_count<0: # when timer completed
				buzzer = 1000 # low beep
				self.buzzer = False # stop triggering buzzer again 
				self.set_point_current_index = len(self.set_points) - 3 # landing position
		# led
		led_gb = 2000 # green light
		if self.alien_published and self.led: # tigger when alien published done
			self.led_count = self.led_count - 1 # timer according to loop counter
			led_gb = 1500 if (self.led_count%2) else 2000 # led value goes high to low 
			if (self.led_count<0): # when timer completed
				self.led = False # stop triggering led again
			
		# Landing
		if self.is_landing : # if landing trigger
			self.landing_decrement_sum += LANDING_DECREMENTER # increase the subtractor of Max_throttle
		self.publish_data_to_rpi( roll = ROLL, pitch = PITCH, throttle = THROTTLE, buzzer_val = buzzer, led_val=led_gb) # publish rc_message

	def alien_pid(self):
		print("ALIEN PID")
		"""
		Last changed Date: 11-02-2023
		Author: team-LD_3114
		Reason: applying pid on image 
		proposal: changes required(pixel_normalized lies in between 1600 to 1800), cross check sign
		changes: switch_navtoalien, buzzer_counter, led_counter
		"""
		# switch to nav_pid to alien_pid
		self.switch_to_normal_flag = True
		if self.switch_to_alien_flag:
			self.sum_error[0] = 0
			self.sum_error[1] = 0
			self.switch_to_alien_flag = False
		

		pixel_normalizer = 2600
		# ERROR 
		position = self.butter_position(roll = self.drone_whycon_pose_array._poses[0].position.x, pitch = self.drone_whycon_pose_array.poses[0].position.y , throttle = self.drone_whycon_pose_array.poses[0].position.z)
		try:
			self.error[0] = (self.alien_centroid[0] - (self.drone_camera_pixel_x / 2)) / pixel_normalizer
			self.error[1] = (self.alien_centroid[1] - (self.drone_camera_pixel_y / 2)) / pixel_normalizer
			self.error[2] = position[2] - 23
		except:
			pass

		# DERIVATIVE ERROR
		self.derivative_error[0] = (self.error[0] - self.previous_error[0])
		self.derivative_error[1] = (self.error[1] - self.previous_error[1])
		self.derivative_error[2] = (self.error[2] - self.previous_error[2])

		# SUM ERROR
		self.sum_error[0] = self.sum_error[0]+self.error[0]
		self.sum_error[1] = self.sum_error[1]+self.error[1]
		self.sum_error[2] = self.sum_error[2]+self.error[2]

		# Integral Error
		self.integral_error[0] = self.sum_error[0] * self.Ki[0]
		self.integral_error[1] = self.sum_error[1] * self.Ki[1]
		self.integral_error[2] = self.sum_error[2] * self.Ki[2]

		print("error:",self.error[0], self.error[1])
		print("Derror:",self.derivative_error[0], self.derivative_error[1])
		print("Ierror:",self.integral_error[0], self.integral_error[1])

		# speed calculation (changes required in kp, kd, ki, and sign)
		ROLL        = int(BASE_ROLL     + ((250 * self.error[0]) + (self.integral_error[0]) + (500 * self.derivative_error[0])))
		PITCH       = int(BASE_PITCH    - ((250 * self.error[1]) + (self.integral_error[1]) + (500 * self.derivative_error[1])))
		THROTTLE    = int(BASE_THROTTLE + ((self.Kp[2] * self.error[2]) + (self.integral_error[2]) + (self.Kd[2] * self.derivative_error[2]))) 

		# remove the chances of getting negative values (look like garbage)
		if ROLL < 0:
			ROLL = BASE_ROLL
		if PITCH < 0:
			PITCH = BASE_PITCH
		if THROTTLE < 0:
			THROTTLE = BASE_THROTTLE

		# assign previous error to current error
		self.previous_error[0] = self.error[0]
		self.previous_error[1] = self.error[1]
		self.previous_error[2] = self.error[2]

		# HOVER
		error_limit_pixels = 30 / 1000 # error range (roll, and pitch)
		error_limit_throttle = 0.8 # throttle error range
		roll_hower_flag     = self.error[0] <  error_limit_pixels and self.error[0] > -error_limit_pixels
		pitch_hower_flag    = self.error[1] <  error_limit_pixels and self.error[1] > -error_limit_pixels
		throttle_hower_flag = self.error[2] <  error_limit_throttle and self.error[2] > -error_limit_throttle

		if (roll_hower_flag and pitch_hower_flag and throttle_hower_flag) :# trigger when flag get according to error in range
			# Success
			print("PUBLISHED")
			print("PUBLISHED")
			print("PUBLISHED")
			print("PUBLISHED")
			print("PUBLISHED")
			print("PUBLISHED")
			print("PUBLISHED")
			print("PUBLISHED")
			print("PUBLISHED")
			
			# message to be published
			alien_type = "alien_a" if self.alien_num==2 else ("alien_b" if self.alien_num == 3 else ("alien_c" if self.alien_num==4 else "alien_d"))
			self.publish_loc(f"{alien_type}",position[0],position[1]+1.0,32)

			# event after message published
			self.alien_published = True # as published occured
			self.buzzer_count = 9*self.alien_num # buzzer loop timer
			self.led_count = 4*self.buzzer_count # led loop timer


		
		# published rc_message according to image
		self.publish_data_to_rpi( roll = ROLL, pitch = PITCH, throttle = THROTTLE, buzzer_val = 1000)

		

	def publish_data_to_rpi(self, roll, pitch, throttle, buzzer_val=1000, led_val = 2000):
		"""
		Last changes Date: 10-02-2024
		Author : e-Yantra, team-LD_3114
		Reason : send calculated value to drone
		Changes : adding led (aux4), buzzer(aux3)
		Proposal : (working)
		"""
		self.rc_message.rc_throttle = int(throttle)
		self.rc_message.rc_roll = int(roll)
		self.rc_message.rc_pitch = int(pitch)
		self.rc_message.rc_yaw = int(1500)
		self.rc_message.aux3 = int(buzzer_val)
		self.rc_message.aux4 = int(led_val)

		# BUTTERWORTH FILTER
		span = 15
		for index, val in enumerate([roll, pitch, throttle]):
			DRONE_WHYCON_POSE[index].append(val)
			if len(DRONE_WHYCON_POSE[index]) == span:
				DRONE_WHYCON_POSE[index].pop(0)
			if len(DRONE_WHYCON_POSE[index]) != span-1:
				return
			order = 3
			fs = 60
			fc = 5
			nyq = 0.5 * fs
			wc = fc/nyq
			b, a = scipy.signal.butter(N=order, Wn=wc, btype='lowpass', analog=False, output='ba')
			filtered_signal = scipy.signal.lfilter(b, a, DRONE_WHYCON_POSE[index])
			if index == 0:
				self.rc_message.rc_roll = int(filtered_signal[-1])
			elif index == 1:
				self.rc_message.rc_pitch = int(filtered_signal[-1])
			elif index == 2:
				self.rc_message.rc_throttle = int(filtered_signal[-1])
		
		# limit the rc_values 
		if self.rc_message.rc_roll > MAX_ROLL:     
			self.rc_message.rc_roll = MAX_ROLL
		elif self.rc_message.rc_roll < MIN_ROLL:
			self.rc_message.rc_roll = MIN_ROLL
 
		if self.rc_message.rc_pitch > MAX_PITCH:
			self.rc_message.rc_pitch = MAX_PITCH
		elif self.rc_message.rc_pitch < MIN_PITCH:
			self.rc_message.rc_pitch = MIN_PITCH
		limit_throttle = int(MAX_THROTTLE-self.landing_decrement_sum) # adding decrement for landing
		
		if limit_throttle<1430: # look like garbage
			limit_throttle = 1430 # garbage

		if(limit_throttle < 1350) or (self.drone_position[2]>31.7) and self.is_landing: # as drone get close to ground, disarm it
			self.disarm()

		if self.rc_message.rc_throttle > limit_throttle:
			self.rc_message.rc_throttle = limit_throttle
		elif self.rc_message.rc_throttle < MIN_THROTTLE:
			self.rc_message.rc_throttle = MIN_THROTTLE

		self.filter_pub.publish(
			PIDError(
				roll_error=float(self.rc_message.rc_roll),
				pitch_error=float(self.rc_message.rc_pitch),
				throttle_error=float(self.rc_message.rc_throttle),
				yaw_error=-0.0,
				zero_error=0.0,
			)
		)
		self.rc_pub.publish(self.rc_message)


	# This function will be called as soon as this rosnode is terminated. So we disarm the drone as soon as we press CTRL + C. 
	# If anything goes wrong with the drone, immediately press CTRL + C so that the drone disamrs and motors stop 
	def publish_loc(self,org,x,y,z):
		self.alien_pub.publish(
			Biolocation(organism_type=org
			,whycon_x=float(x)
			,whycon_y=float(y)
			,whycon_z=float(z)
		))

	def shutdown_hook(self):
		self.node.get_logger().info("Calling shutdown hook")
		self.disarm()

	# Function to arm the drone 

	def arm(self):
		self.node.get_logger().info("Calling arm service")
		self.commandbool.value = True   
		self.future = self.arming_service_client.call_async(self.commandbool)

	# Function to disarm the drone 

	def disarm(self):

		# Create the disarm function
		self.node.get_logger().info("Calling disarm service")
		self.commandbool.value = False
		self.future = self.arming_service_client.call_async(self.commandbool)


def main(args=None):
	rclpy.init(args=args)

	node = rclpy.create_node('controller')
	node.get_logger().info(f"Node Started")
	node.get_logger().info("Entering PID controller loop")

	controller = DroneController(node)
	controller.arm()				
	node.get_logger().info("Armed")
	
	try:
		while rclpy.ok():
			"""
			Last Changed Date: 09-02-2024
			Author: e-Yantra, team-LD_3114
			Changes: adding alien_pid and make switch to change in between them
			Proposal: we can give more priority alien pid as if alien detect we cannot swithed to nav_pid untill we find the alien 
			"""
			if controller.alien_detect and (not controller.alien_published):
				controller.alien_pid()
			else:
				controller.pid()
			if node.get_clock().now().to_msg().sec - controller.last_whycon_pose_received_at > 1:
				node.get_logger().error("Unable to detect WHYCON poses")
			rclpy.spin_once(node)
		

	except Exception as err:
		print(err)

	finally:
		controller.shutdown_hook()
		node.destroy_node()
		rclpy.shutdown()



if __name__ == '__main__':
	main()
