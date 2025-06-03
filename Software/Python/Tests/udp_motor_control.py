import time
import socket
import numpy as np
from gopigo3 import GoPiGo3


class FirstOrderLowPass:
    def __init__(
        self,
        fc,  # Cutoff frequency, in Hz
        fs,  # Sampling frequency, in Hz
    ):
        self.Ts = 1.0 / fs  # Sampling time
        self.tau = 1.0 / (2 * np.pi * fc)  # Filter time constant
        self.alpha = self.Ts / (self.tau + self.Ts)
        self.y_prev = 0.0

    def filter(self, x):
        y = self.alpha * x + (1 - self.alpha) * self.y_prev
        self.y_prev = y

        return y


class LowPassFilter:
    def __init__(
        self,
        order,  # Order of the filter
        fc,  # Cutoff frequency, in Hz
        fs,  # Sampling frequency, in Hz
    ):
        self.stages = [FirstOrderLowPass(fc, fs) for _ in range(order)]

    def filter(self, x):
        for stage in self.stages:
            x = stage.filter(x)

        return x


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


gpg = GoPiGo3()
gpg.reset_all()

PID_L = PID(K_p=0.25, K_i=0.5, K_d=0.0)
LPF_L = LowPassFilter(1, 10, 1/PID_L.dt)

PID_R = PID(K_p=0.25, K_i=0.5, K_d=0.0)
LPF_R = LowPassFilter(1, 10, 1/PID_R.dt)

try:
    print("[INFO] Starting UDP client...")
    gpg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gpg_socket.bind(("0.0.0.0", 25565))

    sim_address = ("192.168.0.101", 25565)

except Exception as e:
    print("[ERROR] Could not start UDP client")
    print(e)
    exit()

print("[INFO] UDP client set")

last_encoder_left = 0
last_encoder_right = 0
start = time.time()

try:
    print("[INFO] Running... Press Ctrl+C to stop")

    while True:
        # Receive UPD message
        data, _ = gpg_socket.recvfrom(1024)

        # Parse wheel velocities
        ref_speed_L, ref_speed_R = map(int, map(float, data.decode().split(" ")))

        # Limit velocity values
        ref_speed_L, ref_speed_R = np.clip([ref_speed_L, ref_speed_R], -1000, 1000)

        # Update time variables
        finish = time.time()

        # Estimate actual speed
        encoder_L = gpg.get_motor_encoder(gpg.MOTOR_LEFT)
        encoder_R = gpg.get_motor_encoder(gpg.MOTOR_RIGHT)

        mea_speed_L = (encoder_L - last_encoder_left) / (finish - start)
        mea_speed_R = (encoder_R - last_encoder_right) / (finish - start)

        # Compute control action
        power_L = PID_L.get_output(ref_speed_L - mea_speed_L)
        power_R = PID_R.get_output(ref_speed_R - mea_speed_R)

        power_L, power_R = np.clip([power_L, power_R], -100, 100)

        # Set velocities to motors
        gpg.set_motor_power(gpg.MOTOR_LEFT, power_L)
        gpg.set_motor_power(gpg.MOTOR_RIGHT, power_R)

        # Send data to simulation
        gpg_socket.sendto(
            f"{ref_speed_L} {mea_speed_L} {power_L} {ref_speed_R} {mea_speed_R} {power_R}".encode(),
            sim_address,
        )

        # Update variables
        start = finish
        last_encoder_left = encoder_L
        last_encoder_right = encoder_R

except KeyboardInterrupt:
    print("\n[INFO] Execution stopped externally")
    print("[INFO] Shutting down motors")
    gpg.set_motor_dps(gpg.MOTOR_LEFT, 0)
    gpg.set_motor_dps(gpg.MOTOR_RIGHT, 0)
