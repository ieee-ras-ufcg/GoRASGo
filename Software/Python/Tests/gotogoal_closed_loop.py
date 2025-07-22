import time
import socket
import numpy as np
from gopigo3 import GoPiGo3


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


gpg = GoPiGo3()
gpg.reset_all()

dt = 1.0 / 30.0
r = 0.0325
s = 0.115

OrientationController = PID(K_p=3.0, K_i=0.0, K_d=0.0, dt=dt)

VelocityController = PID(K_p=1.0, K_i=0.0, K_d=0.0, dt=dt)

try:
    print("[INFO] Starting UDP client...")
    gpg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gpg_socket.bind(("0.0.0.0", 25565))

except Exception as e:
    print("[ERROR] Could not start UDP client")
    print(e)
    exit()

print("[INFO] UDP client set")

try:
    print("[INFO] Running... Press Ctrl+C to stop")

    target = np.array([0.0, 0.0])

    while True:
        # Receive UPD message
        buffer, _ = gpg_socket.recvfrom(1024)

        point_cloud = np.frombuffer(buffer, dtype=np.float32).reshape(3, -1).T[:, :2]

        print(point_cloud)

        # Getting GoPiGo3 Differential Drive Model Position
        position, front_vector = get_pose(point_cloud)

        if position is None:
            print("Format not correct!")
            continue

        path_vector = target - position
        target_distance = np.linalg.norm(path_vector)
        target_vector = path_vector / target_distance

        # Error signals
        orientation_error = np.cross(front_vector, target_vector)
        position_error = np.dot(front_vector, target_vector) * target_distance

        # Controllers
        theta_dot = OrientationController.get_output(orientation_error)
        v_B = VelocityController.get_output(position_error)

        # Differential Drive Model
        theta_dot_L, theta_dot_R = np.linalg.inv(
            [[r / 2.0, r / 2.0], [-r / s, r / s]]
        ) @ np.array([v_B, theta_dot])

        # Stop condition
        if target_distance < 0.05:
            theta_dot_L, theta_dot_R = 0.0, 0.0  # Stop joints

        theta_dot_L, theta_dot_R = np.clip([theta_dot_L, theta_dot_R], -1000, 1000)

        # Set velocities to motors
        gpg.set_motor_dps(gpg.MOTOR_LEFT, theta_dot_L)
        gpg.set_motor_dps(gpg.MOTOR_RIGHT, theta_dot_R)

except KeyboardInterrupt:
    print("\n[INFO] Execution stopped externally")
    print("[INFO] Shutting down motors")
    gpg.set_motor_dps(gpg.MOTOR_LEFT, 0)
    gpg.set_motor_dps(gpg.MOTOR_RIGHT, 0)
