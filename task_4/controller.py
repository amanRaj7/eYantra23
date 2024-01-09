#!/usr/bin/env python3

# Importing the libraries
import copy
import time
import scipy.signal
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PidTune
from swift_msgs.msg import PIDError, RCMessage
from swift_msgs.srv import CommandBool

# Global variables
# PID parameters for roll
MAX_ROLL = 1600
BASE_ROLL = 1500
MIN_ROLL = 1400
SUM_ERROR_ROLL_LIMIT = 1000

# PID parameters for pitch
MAX_PITCH = 1600
BASE_PITCH = 1500
MIN_PITCH = 1400
SUM_ERROR_PITCH_LIMIT = 1000

# PID parameters for throttle
MAX_THROTTLE = 1600
BASE_THROTTLE = 1500
MIN_THROTTLE = 1400
SUM_ERROR_THROTTLE_LIMIT = 1000

# PID Output
PID_OUTPUT_VALUES = [[], [], []] # [roll, pitch, throttle]

class DroneController():
    def __init__(self, node):
        # Initializing the controller
        self.node = node
        self.rc_message = RCMessage()
        self.drone_whycon_pose_array = PoseArray()
        self.is_flying = False
        self.last_whycon_pose_received_at = None
        self.commandbool = CommandBool.Request()

        # PID parameters for roll, pitch and throttle
        self.set_points = [0, 0, 0]
        self.error = [0, 0, 0]
        self.integral = [0, 0, 0]
        self.derivative = [0, 0, 0]
        self.sum_error = [0, 0, 0]
        self.prev_error = [0, 0, 0]

        # PID parameters
        self.kpv = [0.0, 0.0, 0.0]
        self.kiv = [0.0, 0.0, 0.0]
        self.kdv = [0.0, 0.0, 0.0]

        self.kp = [0.01*x for x in self.kpv]
        self.ki = [0.0001*x for x in self.kiv]
        self.kd = [0.1*x for x in self.kdv] 

        # Subscribers
        self.whycon_sub = self.node.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.pid_alt = self.node.create_subscription(PidTune, '/pid_tuning_altitude', self.pid_tune_throttle_callback, 1)
        self.pid_roll = self.node.create_subscription(PidTune, '/pid_tuning_roll', self.pid_tune_roll_callback, 1)
        self.pid_pitch = self.node.create_subscription(PidTune, '/pid_tuning_pitch', self.pid_tune_pitch_callback, 1)

        # Publishers
        self.rc_pub = self.node.create_publisher(RCMessage, '/luminosity_drone/rc_command', 0)
        
        # Services
        self.pid_error_pub = node.create_publisher(PIDError, 'luminosity/pid_error', 1)

        # Timer
        self.timer = self.node.create_timer(0.01, self.timer_callback)
        self.dt = None
        
    def pid_tune_throttle_callback(self, msg):
        self.kp[2] = msg.Kp
        self.ki[2] = msg.Ki
        self.kd[2] = msg.Kd
        
    def pid_tune_roll_callback(self, msg):
        self.kp[0] = msg.Kp
        self.ki[0] = msg.Ki
        self.kd[0] = msg.Kd
    
    def pid_tune_pitch_callback(self, msg):
        self.kp[1] = msg.Kp
        self.ki[1] = msg.Ki
        self.kd[1] = msg.Kd

    def whycon_callback(self, msg):
        self.last_whycon_pose_received_at = self.node.get_clock().now().seconds_nanoseconds()[0]
        self.drone_whycon_pose_array = msg

    def pid(self):
        print("In PID loop")
        self.started_controller_at = time.time()

        # calculating dt
        current_time = self.node.get_clock().now().seconds_nanoseconds()[0]
        self.dt = current_time - self.last_whycon_pose_received_at
        # self.dt = self.started_controller_at - (self.last_whycon_pose_received_at/1e9)
        print("dt: ", self.dt)

        # calculating Error, Derivative, Integral for roll, pitch and throttle
        self.error[0] = self.set_points[0] - self.drone_whycon_pose_array.poses[0].position.x 
        self.integral[0] = self.integral[0] + self.error[0]*self.dt
        self.derivative[0] = (self.error[0] - self.prev_error[0])/self.dt
        self.prev_error[0] = self.error[0]

        self.error[1] = self.set_points[1] - self.drone_whycon_pose_array.poses[0].position.y
        self.integral[1] = self.integral[1] + self.error[1]*self.dt
        self.derivative[1] = (self.error[1] - self.prev_error[1])/self.dt
        self.prev_error[1] = self.error[1]

        self.error[2] = self.set_points[2] - self.drone_whycon_pose_array.poses[0].position.z
        self.integral[2] = self.integral[2] + self.error[2]*self.dt
        self.derivative[2] = (self.error[2] - self.prev_error[2])/self.dt
        self.prev_error[2] = self.error[2]

        # calculating PID output for roll, pitch and throttle
        pid_output_roll = self.kp[0]*self.error[0] + self.ki[0]*self.integral[0] + self.kd[0]*self.derivative[0]
        pid_output_pitch = self.kp[1]*self.error[1] + self.ki[1]*self.integral[1] + self.kd[1]*self.derivative[1]
        pid_output_throttle = self.kp[2]*self.error[2] + self.ki[2]*self.integral[2] + self.kd[2]*self.derivative[2]

        # calculating the final PID output for roll, pitch and throttle
        pid_output_roll = BASE_ROLL + pid_output_roll
        pid_output_pitch = BASE_PITCH + pid_output_pitch
        pid_output_throttle = BASE_THROTTLE + pid_output_throttle

        # limiting the final PID output for roll, pitch and throttle 
        pid_output_roll = max(pid_output_roll, MIN_ROLL)
        pid_output_roll = min(pid_output_roll, MAX_ROLL)
        pid_output_pitch = max(pid_output_pitch, MIN_PITCH)
        pid_output_pitch = min(pid_output_pitch, MAX_PITCH)
        pid_output_throttle = max(pid_output_throttle, MIN_THROTTLE)
        pid_output_throttle = min(pid_output_throttle, MAX_THROTTLE)
        
        # publishing the final PID output for roll, pitch, yaw and throttle (commented because we are publishing the data to rpi)
        # self.rc_message.rc_yaw = 1500
        # self.rc_message.rc_roll = pid_output_roll
        # self.rc_message.rc_pitch = pid_output_pitch
        # self.rc_message.rc_throttle = pid_output_throttle
        # self.rc_pub.publish(self.rc_message)
        
        #----------------------------------#
        # publishing alt error, roll error, pitch error, drone message
        self.publish_data_to_rpi(roll = pid_output_roll, pitch = pid_output_pitch, throttle = pid_output_throttle)
        self.pid_error_pub.publish(
            PIDError(
                roll_error=self.error[0],
                pitch_error=self.error[1],
                throttle_error=self.error[2],
                yaw_error=0,
                zero_error=0
            )
        )

        self.error_pub.publish(
            PIDError(
                roll_error=self.error[0],
                pitch_error=self.error[1],
                throttle_error=self.error[2],
                yaw_error=0,
                zero_error=0
            )
        )
        
        #----------------------------------#

    def publish_data_to_rpi(self, roll, pitch, throttle):
        """
        Publishes the RC data to the Raspberry Pi.

        Args:
            roll (float): The roll value.
            pitch (float): The pitch value.
            throttle (float): The throttle value.
        """
        # Set the RC message values
        self.rc_message.rc_yaw = 1500
        self.rc_message.rc_roll = roll
        self.rc_message.rc_pitch = pitch
        self.rc_message.rc_throttle = throttle
        self.rc_pub.publish(self.rc_message)

        # apply low pass filter to the data (commented because I am not sure for using this)
        # PID_OUTPUT_VALUES[0].append(roll)
        # PID_OUTPUT_VALUES[1].append(pitch)
        # PID_OUTPUT_VALUES[2].append(throttle)
        # PID_OUTPUT_VALUES[0] = scipy.signal.medfilt(PID_OUTPUT_VALUES[0], 5)
        # PID_OUTPUT_VALUES[1] = scipy.signal.medfilt(PID_OUTPUT_VALUES[1], 5)
        # PID_OUTPUT_VALUES[2] = scipy.signal.medfilt(PID_OUTPUT_VALUES[2], 5)

        # publish the data to rpi 
        # self.rc_message.rc_yaw = 1500
        # self.rc_message.rc_roll = PID_OUTPUT_VALUES[0][-1]
        # self.rc_message.rc_pitch = PID_OUTPUT_VALUES[1][-1]
        # self.rc_message.rc_throttle = PID_OUTPUT_VALUES[2][-1]
        # self.rc_pub.publish(self.rc_message)
        
    def shutdown_hook(self):
        self.node.get_logger().info("Calling shutdown hook")
        self.disarm()
    
    def arm(self):
        self.node.get_logger().info("Calling arm service")
        service_endpoint = "/swift/cmd/arming"
        arming_service_client = self.node.create_client(CommandBool,service_endpoint)
        self.commandbool.value = True
        try:
            resp = arming_service_client.call(self.commandbool)
            return resp.success, resp.result
        except Exception as err:
            self.node.get_logger().info(err)

    def disarm(self):
        self.node.get_logger().info("Calling disarm service")
        service_endpoint = "/swift/cmd/disarming"
        disarming_service_client = self.node.create_client(CommandBool,service_endpoint)
        disarming_service_client.wait_for_service(10.0)
        self.commandbool.value = False
        try:
            resp = disarming_service_client.call(self.commandbool)
            return resp.success, resp.result
        except Exception as err:
           self.node.get_logger().info(err)
        self.is_flying = False
    
def main(args = None):
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

if __name__ == "__main__":
    main()
