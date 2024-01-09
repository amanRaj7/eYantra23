#!/usr/bin/env python3

"""
Controller for the drone
"""

# standard imports
import copy
import time

# third-party imports
import scipy.signal
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PidTune
from swift_msgs.msg import PIDError, RCMessage
from swift_msgs.srv import CommandBool



MIN_ROLL = 1250
BASE_ROLL = 1500
MAX_ROLL = 1600
SUM_ERROR_ROLL_LIMIT = 10000


DRONE_WHYCON_POSE = [[], [], []]

# Similarly, create upper and lower limits, base value, and max sum error values for roll and pitch
MIN_PITCH = 1250
BASE_PITCH = 1480
MAX_PITCH = 1600
SUM_ERROR_PITCH_LIMIT = 10000

MIN_THROTTLE = 1000
BASE_THROTTLE = 1450
MAX_THROTTLE = 2000
SUM_ERROR_THROTTLE_LIMIT = 1000

class DroneController():
    def __init__(self,node):
        self.node= node
        self.new_speed = [BASE_ROLL, BASE_PITCH, BASE_THROTTLE]
        self.rc_message = RCMessage()
        self.drone_whycon_pose_array = PoseArray()
        self.last_whycon_pose_received_at = 0
        self.commandbool = CommandBool.Request()
        service_endpoint = "/swift/cmd/arming"

        self.arming_service_client = self.node.create_client(CommandBool,service_endpoint)
        self.set_points = [0, 0, 22]         # Setpoints for x, y, z respectively      
        
        self.error = [0, 0, 0]         # Error for roll, pitch and throttle        
        self.old=0
        # Create variables for integral and differential error
        self.drone_position = [0,0,0]
        self.integral_error = [0, 0, 0]

        self.derivative_error = [0, 0, 0]

        # Create variables for previous error and sum_error
        
        self.previous_error = [0, 0, 0]

        self.sum_error = [0, 0, 0]

        self.Kp = [ 411 * 0.01  , 211 * 0.01  ,  0 * 0.01  ]

        # Similarly create variables for Kd and Ki
        self.Ki = [ 0 * 0.0001, 0 * 0.0001, 0* 0.0001]

        self.Kd = [ 822 * 0.1   , 822 * 0.1   , 0 * 0.1   ]

        # Create subscriber for WhyCon 
        
        self.whycon_sub = node.create_subscription(PoseArray,"/whycon/poses",self.whycon_poses_callback,1)
        
        # Similarly create subscribers for pid_tuning_altitude, pid_tuning_roll, pid_tuning_pitch and any other subscriber if required
       
        self.pid_alt = node.create_subscription(PidTune,"/pid_tuning_altitude",self.pid_tune_throttle_callback,1)
        self.pid_roll = node.create_subscription(PidTune,"/pid_tuning_roll",self.pid_tune_roll_callback,1)
        self.pid_pitch = node.create_subscription(PidTune,"/pid_tuning_pitch",self.pid_tune_pitch_callback,1)

        # Create publisher for sending commands to drone 

        self.rc_pub = node.create_publisher(RCMessage, "/swift/rc_command",1)

        # Create publisher for publishing errors for plotting in plotjuggler 
        
        self.pid_error_pub = node.create_publisher(PIDError, "/luminosity_drone/pid_error",1) 
        self.error_pub = node.create_publisher(PIDError, "/luminosity_drone/error",1)        
        self.filter_pub = node.create_publisher(PIDError, "/luminosity_drone/filter",1)

    def whycon_poses_callback(self, msg):
        self.last_whycon_pose_received_at = self.node.get_clock().now().seconds_nanoseconds()[0]
        self.drone_whycon_pose_array = msg
        # alpha=0.09
        # self.drone_position[0] = alpha*msg.poses[0].position.x+ (1-alpha)*self.drone_position[0] 
        # self.drone_position[1] = alpha*msg.poses[0].position.y+ (1-alpha)*self.drone_position[1] 
        # self.drone_position[2] = alpha*msg.poses[0].position.z+ (1-alpha)*self.drone_position[2] 
        
    # def low_pass_filter(self, roll, pitch, throttle):
    #     alpha = 1
    #     self.new_speed[0] = alpha * roll + (1-alpha) * self.new_speed[0]
    #     self.new_speed[1] = alpha * pitch + (1-alpha) * self.new_speed[1]
    #     self.new_speed[2] = alpha * throttle + (1-alpha) * self.new_speed[2]
    #     return self.new_speed
    def pid_tune_throttle_callback(self, alt):
        self.Kp[2] = alt.kp * 0.01
        self.Ki[2] = alt.ki * 0.0001
        self.Kd[2] = alt.kd * 0.1

    # Similarly add callbacks for other subscribers
    def pid_tune_roll_callback(self, roll):
        self.Kp[0] = roll.kp * 0.01
        self.Ki[0] = roll.ki * 0.0001
        self.Kd[0] = roll.kd * 0.1

    def pid_tune_pitch_callback(self, pitch):
        self.Kp[1] = pitch.kp * 0.01
        self.Ki[1] = pitch.ki * 0.0001
        self.Kd[1] = pitch.kd * 0.1

    def pid(self):          # PID algorithm

        # 0 : calculating Error, Derivative, Integral for Roll error : x axis
        try:
            self.error[0] = self.drone_whycon_pose_array.poses[0].position.x - self.set_points[0] 
        # Similarly calculate error for y and z axes 
            self.error[1] = self.drone_whycon_pose_array.poses[0].position.y - self.set_points[1]
            self.error[2] = self.drone_whycon_pose_array.poses[0].position.z - self.set_points[2]
        except:
            pass

        # Calculate derivative and intergral errors. Apply anti windup on integral error (You can use your own method for anti windup, an example is shown here)
        self.derivative_error[0] = (self.error[0] - self.previous_error[0])
        self.derivative_error[1] = (self.error[1] - self.previous_error[1])
        self.derivative_error[2] = (self.error[2] - self.previous_error[2])

        self.integral_error[0] = self.sum_error[0] * self.Ki[0]
        self.integral_error[1] = self.sum_error[1] * self.Ki[1]
        self.integral_error[2] = self.sum_error[2] * self.Ki[2]


        #  WINDUP TRY NEW AS WELL---------------------+++++++++++++ 
        # if self.integral_error[0] > SUM_ERROR_ROLL_LIMIT:
        #     self.integral_error[0] = SUM_ERROR_ROLL_LIMIT
        # if self.integral_error[0] < -SUM_ERROR_ROLL_LIMIT:
        #     self.integral_error[0] = -SUM_ERROR_ROLL_LIMIT

        # if self.integral_error[1] > SUM_ERROR_PITCH_LIMIT:
        #     self.integral_error[1] = SUM_ERROR_PITCH_LIMIT
        # if self.integral_error[1] < -SUM_ERROR_PITCH_LIMIT:
        #     self.integral_error[1] = -SUM_ERROR_PITCH_LIMIT
        
        if self.integral_error[2] > SUM_ERROR_THROTTLE_LIMIT:
            self.integral_error[2] = SUM_ERROR_THROTTLE_LIMIT
        if self.integral_error[2] < -SUM_ERROR_THROTTLE_LIMIT:
            self.integral_error[2] = -SUM_ERROR_THROTTLE_LIMIT

    
        self.sum_error[0] += self.error[0]
        self.sum_error[1] += self.error[1]
        self.sum_error[2] += self.error[2]

        
        # 1 : calculating Error, Derivative, Integral for Pitch error : y axis

        # 2 : calculating Error, Derivative, Integral for Alt error : z axis

        #self.Kp[2]=2000
        # Write the PID equations and calculate the self.rc_message.rc_throttle, self.rc_message.rc_roll, self.rc_message.rc_pitch
        ROLL        = int(BASE_ROLL     + ((self.Kp[0] * self.error[0]) + (self.integral_error[0]) + (self.Kd[0] * self.derivative_error[0])))
        PITCH       = int(BASE_PITCH    - ((self.Kp[1] * self.error[1]) + (self.integral_error[1]) + (self.Kd[1] * self.derivative_error[1])))
        THROTTLE    = int(BASE_THROTTLE + ((self.Kp[2] * self.error[2]) + (self.integral_error[2]) + (self.Kd[2] * self.derivative_error[2])))
        print(THROTTLE," ",self.error[2]," ",self.sum_error[2]," ",self.Kp[2])

        # Save current error in previous error
        self.previous_error = self.error
    #------------------------------------------------------------------------------------------------------------------------

        self.publish_data_to_rpi( roll = ROLL, pitch = PITCH, throttle = THROTTLE)

        #Replace the roll pitch and throttle values as calculated by PID 
        
        
        # Publish alt error, roll error, pitch error for plotjuggler debugging

        self.pid_error_pub.publish(
            PIDError(
                roll_error=float(self.error[0]),
                pitch_error=float(self.error[1]),
                throttle_error=float(self.error[2]),
                yaw_error=-0.0,
                zero_error=0.0,
            )
        )
        self.error_pub.publish(
            PIDError(
                roll_error=float(ROLL),
                pitch_error=float(PITCH),
                throttle_error=float(THROTTLE),
                yaw_error=-0.0,
                zero_error=0.0,
            )
        )


    def publish_data_to_rpi(self, roll, pitch, throttle):
        #roll, pitch, throttle = self.low_pass_filter(roll=roll, pitch=pitch, throttle=throttle)
        
        self.rc_message.rc_throttle = int(throttle)
        self.rc_message.rc_roll = int(roll)
        self.rc_message.rc_pitch = int(pitch)
        # Send constant 1500 to rc_message.rc_yaw
        self.rc_message.rc_yaw = int(1500)
        # BUTTERWORTH FILTER
        # span = 15
        # for index, val in enumerate([roll, pitch, throttle]):
        #     DRONE_WHYCON_POSE[index].append(val)
        #     if len(DRONE_WHYCON_POSE[index]) == span:
        #         DRONE_WHYCON_POSE[index].pop(0)
        #     if len(DRONE_WHYCON_POSE[index]) != span-1:
        #         return
        #     order = 3
        #     fs = 60
        #     fc = 5
        #     nyq = 0.5 * fs
        #     wc = fc/nyq
        #     b, a = scipy.signal.butter(N=order, Wn=wc, btype='lowpass', analog=False, output='ba')
        #     filtered_signal = scipy.signal.lfilter(b, a, DRONE_WHYCON_POSE[index])
        #     if index == 0:
        #         self.rc_message.rc_roll = int(filtered_signal[-1])
        #     elif index == 1:
        #         self.rc_message.rc_pitch = int(filtered_signal[-1])
        #     elif index == 2:
        #         self.rc_message.rc_throttle = int(filtered_signal[-1])
        
        # self.filter_pub.publish(
        #     PIDError(
        #         roll_error=float(self.old),
        #         pitch_error=float(self.rc_message.rc_throttle),
        #         throttle_error=0.0,
        #         yaw_error=-0.0,c = fc / nyq
        #     b, a = scipy.signal.butter(N=
        #         zero_error=0.0,
        #     )
        # )
        if self.rc_message.rc_roll > MAX_ROLL:     #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_roll = MAX_ROLL
        elif self.rc_message.rc_roll < MIN_ROLL:
            self.rc_message.rc_roll = MIN_ROLL

        # Similarly add bounds for pitch yaw and throttle 
        if self.rc_message.rc_pitch > MAX_PITCH:
            self.rc_message.rc_pitch = MAX_PITCH
        elif self.rc_message.rc_pitch < MIN_PITCH:
            self.rc_message.rc_pitch = MIN_PITCH
        
        if self.rc_message.rc_throttle > MAX_THROTTLE:
            self.rc_message.rc_throttle = MAX_THROTTLE
        elif self.rc_message.rc_throttle < MIN_THROTTLE:
            self.rc_message.rc_throttle = MIN_THROTTLE
        
        self.rc_pub.publish(self.rc_message)


    # This function will be called as soon as this rosnode is terminated. So we disarm the drone as soon as we press CTRL + C. 
    # If anything goes wrong with the drone, immediately press CTRL + C so that the drone disamrs and motors stop 

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
            controller.pid()
            if node.get_clock().now().to_msg().sec - controller.last_whycon_pose_received_at > 1:
                node.get_logger().error("Unable to detect WHYCON poses")
            rclpy.spin_once(node) # Sleep for 1/30 secs
        

    except Exception as err:
        print(err)

    finally:
        controller.shutdown_hook()
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
