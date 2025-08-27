import time
import socket
import numpy as np

import rclpy
from rclpy.node import Node
import socket
import numpy as np
from sensor_msgs.msg import Twist
from std_msgs.msg import Header
import struct


class PID:
    def __init__(
        self,
        # PID Parameters
        K_p=0.0,  # Proportional parameter
        K_i=0.0,  # Integral parameter
        K_d=0.0,  # Differential parameter
        dt=50e-3,  # Differential time step
    ):
        # Simulation Time Step
        self.dt = dt

        # PID Gains
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        # Error signals
        self.prev_e = None
        self.curr_e = 0.0
        self.accu_e = 0.0
        self.diff_e = 0.0

    def get_output(self, e):
        # Update error signals
        if self.prev_e is None:
            self.prev_e = e

        self.curr_e = e
        self.accu_e += e * self.dt
        self.diff_e = (self.curr_e - self.prev_e) / self.dt

        # Compute output
        output = self.K_p * e + self.K_i * self.accu_e + self.K_d * self.diff_e

        # Update previous error
        self.prev_e = self.curr_e

        return output

    def reset(self):
        self.prev_e = None
        self.curr_e = 0.0
        self.accu_e = 0.0
        self.diff_e = 0.0


def get_pose(point_cloud):
    if point_cloud.shape != (3, 2):
        return None, None

    distance_matrix = np.abs(
        [
            [np.linalg.norm(marker - marker_) for marker_ in point_cloud]
            for marker in point_cloud
        ]
    )

    apex = np.argmax(np.sum(distance_matrix, axis=1))

    position = np.mean(np.delete(point_cloud, apex, axis=0), axis=0)

    front_vector = point_cloud[apex] - position

    front_vector = front_vector / np.linalg.norm(front_vector)

    return position, front_vector


class UGVSimpleGoToGoal(Node):
    def __init__(self):
        super().__init__("ugv_simple_go_to_goal")

        self.udp_ip = "127.0.0.1"
        self.udp_port = 6666

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))

        self.target = np.array([0.0, 0.0])  # Robot's goal 

        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)

        self.fps = 30.0
        self.dt = 1.0 / self.fps

        self.timer = self.create_timer(self.dt, self.receive_udp)

        self.OrientationController = PID(K_p=60.0, K_i=0.0, K_d=0.0, dt=self.dt)
        self.VelocityController = PID(K_p=20.0, K_i=0.0, K_d=0.0, dt=self.dt)

    def receive_udp(self):
        try:
            # Receive UPD message
            buffer, _ = self.sock.recvfrom(1024)

            point_cloud = (
                np.frombuffer(buffer, dtype=np.float32).reshape(3, -1).T[:, :2]
            )

            # Getting GoPiGo3 Differential Drive Model Position
            position, front_vector = get_pose(point_cloud)

            if position is None:
                raise ValueError

            path_vector = self.target - position
            target_distance = np.linalg.norm(path_vector)
            target_vector = path_vector / target_distance

            # Error signals
            orientation_error = np.cross(front_vector, target_vector)
            position_error = np.dot(front_vector, target_vector) * target_distance

            # Controllers
            theta_dot = self.OrientationController.get_output(orientation_error)
            v_B = self.VelocityController.get_output(position_error)

            # Stop condition
            if target_distance < 0.05:
                v_B, theta_dot = 0.0, 0.0  # Stop robot

            msg = Twist()
            msg.linear.x = v_B  # Linear velocity of the robot
            msg.angular.z = theta_dot  # Angular velocity of the robot

            self.publisher.publish(msg)

        except BlockingIOError:
            pass

        except ValueError:
            self.get_logger().error(f"Incorrect Parsing")

        except Exception as e:
            self.get_logger().error(f"{str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = UGVSimpleGoToGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
