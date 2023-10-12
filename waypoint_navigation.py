#!/usr/bin/env python3
'''
# * Team Id : #3114
# * Author List : Aman Raj, Pratyush Roshan Mallik, Gaurav Kumar Sharma, Chandan Priyadarshi
# * Filename: = position_hold.py
# * Theme: Luminosity Drone
# * Functions: 
# * Global Variables: NONE
'''

'''

This python file runs a ROS-node of name drone_control which holds the position of Swift-Drone on the given dummy.
This node publishes and subsribes the following topics:

		PUBLICATIONS			SUBSCRIPTIONS
		/drone_command			/whycon/poses
		/alt_error				/pid_tuning_altitude
		/pitch_error			/pid_tuning_pitch
		/roll_error				/pid_tuning_roll
					
								

Rather than using different variables, use list. eg : self.setpoint = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_setpoint = 1, self.y_setpoint = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.	
'''


# Importing the required libraries

from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time


class swift():
	"""docstring for swift"""
	def __init__(self):
		
		rospy.init_node('drone_control')	# initializing ros node with name drone_control

		# This corresponds to your current position of drone. This value must be updated each time in your whycon callback
		# [x,y,z]
		self.drone_position = [0.0,0.0,0.0]	

		# [x_setpoint, y_setpoint, z_setpoint]
		self.setpoints = [   [0, 0, 23],
                            [2, 0, 23],
                            [2, 2, 23],
                            [2, 2, 25],
                            [-5, 2, 25],
                            [-5, -3, 25],
                            [-5, -3, 21],
                            [7, -3, 21],
                            [7, 0, 21],
                            [0, 0, 19]  ] # whycon marker at the position of the dummy given in the scene. Make the whycon marker associated with position_to_hold dummy renderable and make changes accordingly
        
		self.current_setpoint_index = 0

		#Declaring a cmd of message type swift_msgs and initializing values
		self.cmd = swift_msgs()
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX3 = 1500
		self.cmd.rcAUX4 = 1500
        


		#initial setting of Kp, Kd and ki for [roll, pitch, throttle]. eg: self.Kp[2] corresponds to Kp value in throttle axis
		#after tuning and computing corresponding PID parameters, change the parameters



		#to be done

				# roll, pitch, throttle
		self.Kp = [ 30.78  , 34.5 , 63.2]
		self.Ki = [ 0.0008 , 0    ,0.0826]
		self.Kd = [ 600.5  , 579  , 655]
   
		#-----------------------Add other required variables for pid here ----------------------------------------------
		
			            # roll, pitch, throttle
		self.prev_error = [0.0, 0.0, 0.0]
		self.sum_error =  [0.0, 0.0, 0.0]

		self.min_values = [1350,1350,1000]
		self.max_values = [1700,1700,2000]

		self.derivative = [0.0, 0.0, 0.0]
		self.integral = [0.0, 0.0, 0.0]

		# Hint : Add variables for storing previous errors in each axis, like self.prev_error = [0,0,0] where corresponds to [pitch, roll, throttle]		#		 Add variables for limiting the values like self.max_values = [2000,2000,2000] corresponding to [roll, pitch, throttle]
		#													self.min_values = [1000,1000,1000] corresponding to [pitch, roll, throttle]
		#																	You can change the upper limit and lower limit accordingly. 
		#----------------------------------------------------------------------------------------------------------

		# # This is the sample time in which you need to run pid. Choose any time which you seem fit. Remember the stimulation step time is 50 ms
		# self.sample_time = 0.03333 # in seconds

		# Publishing /drone_command, /alt_error, /pitch_error, /roll_error
		self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
		#------------------------Add other ROS Publishers here-----------------------------------------------------

		self.alt_error_pub = rospy.Publisher('/alt_error', Float64, queue_size =1)
		self.roll_error_pub = rospy.Publisher('/roll_error',Float64, queue_size=1)
		self.pitch_error_pub = rospy.Publisher('/pitch_error',Float64, queue_size=1)
		
		self.zero_line_pub = rospy.Publisher('/zero_line',Float64, queue_size=1)	
		self.minus_one_pub = rospy.Publisher('/minus_one',Float64, queue_size=1)
		self.plus_one_pub = rospy.Publisher('/plus_one',Float64, queue_size=1)
		self.plus_half_pub = rospy.Publisher('/plus_half',Float64, queue_size=1)
		self.minus_half_pub = rospy.Publisher('/minus_half',Float64, queue_size=1)

	#-----------------------------------------------------------------------------------------------------------


		# Subscribing to /whycon/poses, /pid_tuning_altitude, /pid_tuning_pitch, pid_tuning_roll
		rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
		rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
		#-------------------------Add other ROS Subscribers here----------------------------------------------------
		rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)
		rospy.Subscriber('/pid_tuning_roll',PidTune,self.roll_set_pid)

		#------------------------------------------------------------------------------------------------------------

		self.arm() # ARMING THE DRONE


	# Disarming condition of the drone
	def disarm(self):
		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1100
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)


	# Arming condition of the drone : Best practise is to disarm and then arm the drone.
	def arm(self):

		self.disarm()

		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1500
		self.command_pub.publish(self.cmd)	# Publishing /drone_command
		rospy.sleep(1)


	# Whycon callback function
	# The function gets executed each time when /whycon node publishes /whycon/poses 
	def whycon_callback(self,msg):
		self.drone_position[0] = msg.poses[0].position.x
		#--------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------
		self.drone_position[1] = msg.poses[0].position.y
		self.drone_position[2] = msg.poses[0].position.z
		#---------------------------------------------------------------------------------------------------------------


	# Callback function for /pid_tuning_altitude
	# This function gets executed each time when /tune_pid publishes /pid_tuning_altitude
	def altitude_set_pid(self,alt):
		self.Kp[2] =   alt.Kp * 0.1 # This is just for an example. You can change the ratio/fraction value accordingly
		self.Ki[2] =   alt.Ki * 0.0007
		self.Kd[2] =   alt.Kd * 0.5
		
	#----------------------------Define callback function like altitide_set_pid to tune pitch, roll--------------
	def pitch_set_pid(self,pitch):
		self.Kp[1] = pitch.Kp * 0.01
		self.Ki[1] = pitch.Ki * 0.0005
		self.Kd[1] = pitch.Kd * 0.5	

	def roll_set_pid(self,roll):
		self.Kp[0] =  roll.Kp * 0.03
		self.Ki[0] =  roll.Ki * 0.00005
		self.Kd[0] =  roll.Kd * 0.5

	#----------------------------------------------------------------------------------------------------------------------

	def pid(self):

	#-----------------------------Write the PID algorithm here--------------------------------------------------------------

	# Steps:
	# 	1. Compute error in each axis. eg: error[0] = self.drone_position[0] - self.setpoint[0] ,where error[0] corresponds to error in x...
	#	2. Compute the error (for proportional), change in error (for derivative) and sum of errors (for integral) in each axis. Refer "Understanding PID.pdf" to understand PID equation.
	#	3. Calculate the pid output required for each axis. For eg: calcuate self.out_roll, self.out_pitch, etc.
	#	4. Reduce or add this computed output value on the avg value ie 1500. For eg: self.cmd.rcRoll = 1500 + self.out_roll. LOOK OUT FOR SIGN (+ or -). EXPERIMENT AND FIND THE CORRECT SIGN
	#	5. Don't run the pid continously. Run the pid only at the a sample time. self.sampletime defined above is for this purpose. THIS IS VERY IMPORTANT.
	#	6. Limit the output value and the final command value between the maximum(2000) and minimum(1000)range before publishing. For eg : if self.cmd.rcPitch > self.max_values[1]:
	#																														self.cmd.rcPitch = self.max_values[1]
	#	7. Update previous errors.eg: self.prev_error[1] = error[1] where index 1 corresponds to that of pitch (eg)
	#	8. Add error_sum

		if self.current_setpoint_index >= len(self.setpoints):
			return

		# creating reference lines on the plot
		self.zero_line = 0.00
		self.minus_one_line = -1
		self.minus_half_line = -0.2
		self.plus_half_line = 0.2
		self.plus_one_line = 1.00

		roll_error = self.drone_position[0] - self.setpoints[self.current_setpoint_index][0]
		pitch_error = self.drone_position[1] - self.setpoints[self.current_setpoint_index][1]
		alt_error = self.drone_position[2] - self.setpoints[self.current_setpoint_index][2]
		self.error = [roll_error, pitch_error, alt_error]

		self.integral = [self.sum_error[0] * self.Ki[0],
				    	 self.sum_error[1] * self.Ki[1],
					  	 self.sum_error[2] * self.Ki[2]]

		self.derivative = [self.error[0] - self.prev_error[0],
					  	   self.error[1] - self.prev_error[1],
					       self.error[2] - self.prev_error[2]]
		
		self.sum_error[0] += self.error[0]
		self.sum_error[1] += self.error[1]
		self.sum_error[2] += self.error[2]			

		self.cmd.rcRoll     = int(1500 - ((self.Kp[0] * self.error[0]) + (self.integral[0]) + (self.Kd[0] * self.derivative[0])))
		self.cmd.rcPitch    = int(1500 + ((self.Kp[1] * self.error[1]) + (self.integral[1]) + (self.Kd[1] * self.derivative[1])))
		self.cmd.rcThrottle = int(1550 + ((self.Kp[2] * self.error[2]) + (self.integral[2]) + (self.Kd[2] * self.derivative[2])))
		
		#print("Derivative: ",self.Kd[2] * self.derivative[2],"Integral: ",self.integral[2],"Error: ",self.Kp[2] * self.error[2])

 		# limiting the values

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

		
		# updating the errors


		self.prev_error = self.error



	#------------------------------------------------------------------------------------------------------------------------
		self.command_pub.publish(self.cmd)
		self.alt_error_pub.publish(self.error[2])
		self.roll_error_pub.publish(self.error[0])
		self.pitch_error_pub.publish(self.error[1])

		if all(abs(e) < 0.2 for e in self.error):
			self.current_setpoint_index += 1

		# publishing the reference lines
		self.zero_line_pub.publish(self.zero_line)
		#self.minus_one_pub.publish(self.minus_one_line)
		#self.plus_one_pub.publish(self.plus_one_line)
		self.plus_half_pub.publish(self.plus_half_line)
		self.minus_half_pub.publish(self.minus_half_line)



if __name__ == '__main__':

	swift_drone = swift()
	r = rospy.Rate(30) #specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz
	while not rospy.is_shutdown():
		swift_drone.pid()
		r.sleep()
		
        