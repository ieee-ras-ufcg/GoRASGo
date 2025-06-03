import time
import socket
import numpy as np
from gopigo3 import GoPiGo3

gpg = GoPiGo3()

try:
    print("[INFO] Starting UDP client...")
    gpg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gpg_socket.bind(("0.0.0.0", 25565))

except:
    print("[ERROR] Could not start UDP client")
    exit()

print("[INFO] UDP client set")

last_encoder_left = 0
last_encoder_right = 0
start = time.time()

try:
    print("[INFO] Running... Press Ctrl+C to stop")

    while True:
        # Receive UPD message
        data, address = gpg_socket.recvfrom(1024) 

        # Parse wheel velocities
        phi_dot_L, phi_dot_R = map(int, map(float, data.decode().split(" ")))

        # Limit velocity values
        phi_dot_L, phi_dot_R = np.clip(-1000, 1000, [phi_dot_L, phi_dot_R])

        # Update time variables
        finish = time.time()
        dt = finish - start
        start = finish

        # Estimate actual speed
        encoder_left = gpg.get_motor_encoder(gpg.MOTOR_LEFT)
        encoder_right = gpg.get_motor_encoder(gpg.MOTOR_RIGHT)
        speed_left = (encoder_left - last_encoder_left) / dt
        speed_right = (encoder_right - last_encoder_right) / dt
        last_encoder_left = encoder_left
        last_encoder_right = encoder_right

        # Set velocities to motors
        gpg.set_motor_dps(gpg.MOTOR_LEFT, phi_dot_L)
        gpg.set_motor_dps(gpg.MOTOR_RIGHT, phi_dot_R)

        print(f"L: {phi_dot_L:.2f}{speed_left:.2f} ({phi_dot_L - speed_left:.2f}) | R: {phi_dot_R:.2f}{speed_right:.2f} ({phi_dot_R - speed_right:.2f})")

except KeyboardInterrupt:
    print("\n[INFO] Execution stopped externally")
    print("[INFO] Motors shutdown")
    gpg.set_motor_dps(gpg.MOTOR_LEFT, 0)
    gpg.set_motor_dps(gpg.MOTOR_RIGHT, 0)
