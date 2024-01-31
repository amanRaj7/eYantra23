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



MIN_ROLL = 1400
BASE_ROLL = 1490
MAX_ROLL = 1600
SUM_ERROR_ROLL_LIMIT = 30


DRONE_WHYCON_POSE = [[], [], []]

# Similarly, create upper and lower limits, base value, and max sum error values for roll and pitch
MIN_PITCH = 1400
BASE_PITCH = 1480
MAX_PITCH = 1600
SUM_ERROR_PITCH_LIMIT = 30

MIN_THROTTLE = 1420
BASE_THROTTLE = 1460
MAX_THROTTLE = 1520
SUM_ERROR_THROTTLE_LIMIT = 10000

# LANDING MACROS
LANDING_TRIGGER_ITERATION = 500
LANDING_DECREMENTER = 1

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
        
        """OUR DEFINATION"""
        self.it = 0
        self.set_points = [0, 0, 22]         # Setpoints for x, y, z respectively      
        self.error = [0, 0, 0]         # Error for roll, pitch and throttle
        self.drone_position = [0,0,0]
        self.integral_error = [0, 0, 0]
        self.derivative_error = [0, 0, 0]
        self.previous_error = [0, 0, 0]
        self.sum_error = [0, 0, 0]
        self.Kp = [ 1040 * 0.01  , 930 * 0.01  ,  615 * 0.01  ]
        self.Ki = [ 60 * 0.0001  , 50 * 0.0001  , 215* 0.0001  ]
        self.Kd = [ 3500 * 0.1   , 4500 * 0.1   , 1840 * 0.1    ]

        # LANDING VARIABLES
        self.land_iteration_count = 0
        self.landing_decrement_sum = 0
        self.landing_saved_throttle = 0
        self.is_landing = False

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

    def butter_position(self, roll, pitch, throttle):
        # apply the low pass filter 
        alpha = 1
        beta = 0.9
        self.drone_position[0] = alpha*roll+ (1-alpha)*self.drone_position[0]
        self.drone_position[1] = alpha*pitch+ (1-alpha)*self.drone_position[1]
        self.drone_position[2] = beta*throttle+ (1-beta)*self.drone_position[2]
        return self.drone_position
        
    def pid(self):
        print(f"\nITERATION {self.it}")
        print(f"Time: {self.node.get_clock().now().seconds_nanoseconds()}")
        self.it = self.it+1
        # ERROR 
        try:
            position = self.butter_position(roll = self.drone_whycon_pose_array._poses[0].position.x, pitch = self.drone_whycon_pose_array.poses[0].position.y , throttle= self.drone_whycon_pose_array.poses[0].position.z)
            print(f"position: {position}")
            self.error[0] = position[0] - self.set_points[0]
            self.error[1] = position[1] - self.set_points[1]
            self.error[2] = position[2] - self.set_points[2]
        except:
            pass
        print(f"ERROR: {self. error}")

        # DERIVATIVE ERROR
        self.derivative_error[0] = (self.error[0] - self.previous_error[0])
        self.derivative_error[1] = (self.error[1] - self.previous_error[1])
        self.derivative_error[2] = (self.error[2] - self.previous_error[2])
        print(f"Derivative Error: {self.derivative_error}")

        self.sum_error[0] = self.sum_error[0]+self.error[0]
        self.sum_error[1] = self.sum_error[1]+self.error[1]
        self.sum_error[2] = self.sum_error[2]+self.error[2]
        

    
        

        #  WINDUP TRY NEW AS WELL---------------------+++++++++++++ 
        # if self.sum_error[0] > SUM_ERROR_ROLL_LIMIT:
        #     self.sum_error[0] = SUM_ERROR_ROLL_LIMIT
        # if self.sum_error[0] < -SUM_ERROR_ROLL_LIMIT:
        #     self.sum_error[0] = -SUM_ERROR_ROLL_LIMIT

        # if self.sum_error[1] > SUM_ERROR_PITCH_LIMIT:
        #     self.sum_error[1] = SUM_ERROR_PITCH_LIMIT
        # if self.sum_error[1] < -SUM_ERROR_PITCH_LIMIT:
        #     self.sum_error[1] = -SUM_ERROR_PITCH_LIMIT

        self.integral_error[0] = self.sum_error[0] * self.Ki[0]
        self.integral_error[1] = self.sum_error[1] * self.Ki[1]
        self.integral_error[2] = self.sum_error[2] * self.Ki[2]

        print(f"Integral Error: {self.integral_error}")
        print(f"Sum Error: {self.sum_error}")
        

    
        # self.sum_error[0] += self.error[0]
        # self.sum_error[1] += self.error[1]
        # self.sum_error[2] += self.error[2]

        
        # 1 : calculating Error, Derivative, Integral for Pitch error : y axis

        # 2 : calculating Error, Derivative, Integral for Alt error : z axis

        #self.Kp[2]=2000
        # Write the PID equations and calculate the self.rc_message.rc_throttle, self.rc_message.rc_roll, self.rc_message.rc_pitch
        print(f"\n\n{self.integral_error[0]}")
        ROLL        = int(BASE_ROLL     + ((self.Kp[0] * self.error[0]) + (self.integral_error[0]) + (self.Kd[0] * self.derivative_error[0])))
        PITCH       = int(BASE_PITCH    - ((self.Kp[1] * self.error[1]) + (self.integral_error[1]) + (self.Kd[1] * self.derivative_error[1])))
        THROTTLE    = int(BASE_THROTTLE + ((self.Kp[2] * self.error[2]) + (self.integral_error[2]) + (self.Kd[2] * self.derivative_error[2]))) 
        print(f"Calculated Speed: Roll: {ROLL}, Pitch: {PITCH}, Throttle: {THROTTLE}")
        # print(THROTTLE," ",self.error[2]," ",self.sum_error[2]," ",self.Kp[2])
        # if self.error[2] > 4:
        #     THROTTLE -= 20
        # if self.error[2] < 0.5:
        #     THROTTLE += 10
        # Save current error in previous error
        # print("old", self.previous_error)
        self.previous_error[0] = self.error[0]
        self.previous_error[1] = self.error[1]
        self.previous_error[2] = self.error[2]
        # print("new", self.previous_error)

        #LANDING
        error_limit = 0.8
        roll_hower_flag     = self.error[0] <  error_limit and self.error[0] > -error_limit
        pitch_hower_flag    = self.error[1] <  error_limit and self.error[1] > -error_limit
        throttle_hower_flag = self.error[2] <  error_limit and self.error[2] > -error_limit

        if not self.is_landing:
            if (roll_hower_flag and pitch_hower_flag and throttle_hower_flag):

                self.land_iteration_count += 1

                if self.land_iteration_count > LANDING_TRIGGER_ITERATION :
                    #self.landing_decrement_sum -= LANDING_DECREMENTER
                    self.is_landing = True
                print("HOVER")
            else :
                self.land_iteration_count = 0
                
        
        if self.is_landing :
            print("LANDING")
            self.landing_decrement_sum += LANDING_DECREMENTER

            # saved throttle first time is_landing is triggered
            if self.landing_saved_throttle == 0:
                self.landing_saved_throttle = THROTTLE

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
                roll_error=float(self.error[2]),
                pitch_error=float(self.derivative_error[2]),
                throttle_error=float(self.sum_error[2]),
                yaw_error=-0.0,
                zero_error=0.0,
            )
        )


    def publish_data_to_rpi(self, roll, pitch, throttle):

        self.rc_message.rc_throttle = int(throttle)
        self.rc_message.rc_roll = int(roll)
        self.rc_message.rc_pitch = int(pitch)
        self.rc_message.rc_yaw = int(1500)

        """APPLYING BUTTERWORTH FILTER"""
        # Filter
        # alpha = 0.7
        # self.new_speed[0] = alpha*roll+ (1-alpha)*self.new_speed[0]
        # self.new_speed[1] = alpha*pitch+ (1-alpha)*self.new_speed[1]
        # self.new_speed[2] = alpha*throttle+ (1-alpha)*self.new_speed[2]
        # roll = self.new_speed[0]
        # pitch = self.new_speed[1]
        # throttle = self.new_speed[2]

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
        
        
        if self.rc_message.rc_roll > MAX_ROLL:     
            self.rc_message.rc_roll = MAX_ROLL
        elif self.rc_message.rc_roll < MIN_ROLL:
            self.rc_message.rc_roll = MIN_ROLL
 
        if self.rc_message.rc_pitch > MAX_PITCH:
            self.rc_message.rc_pitch = MAX_PITCH
        elif self.rc_message.rc_pitch < MIN_PITCH:
            self.rc_message.rc_pitch = MIN_PITCH
        limit_throttle = MAX_THROTTLE-self.landing_decrement_sum
        if limit_throttle<1440:
            limit_throttle = 1440

        if(limit_throttle < 1350) or (self.error[2]>9.7) and self.is_landing:
            self.disarm()
        if self.rc_message.rc_throttle > limit_throttle:
            self.rc_message.rc_throttle = limit_throttle
        elif self.rc_message.rc_throttle < MIN_THROTTLE:
            self.rc_message.rc_throttle = MIN_THROTTLE

        #  # LANDING THORTTLE LIMIT
        # if self.is_landing :
        #     self.lan

        self.filter_pub.publish(
            PIDError(
                roll_error=float(self.rc_message.rc_roll),
                pitch_error=float(self.rc_message.rc_pitch),
                throttle_error=float(self.rc_message.rc_throttle),
                yaw_error=-0.0,
                zero_error=0.0,
            )
        )
        
        print(f"Limitted Speed: Roll: {self.rc_message.rc_roll}, Pitch: {self.rc_message.rc_pitch}, Throttle: {self.rc_message.rc_throttle}")
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
