import time
import socket
import numpy as np
from gopigo3 import GoPiGo3

gpg = GoPiGo3()
gpg.reset_all()

try:
    print("[INFO] Starting UDP client...")
    gpg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gpg_socket.bind(("0.0.0.0", 25565))

    sim_address = ("192.168.0.111", 25565)

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
        data, address = gpg_socket.recvfrom(1024)

        # Parse wheel velocities
        ref_speed_L, ref_speed_R = list(map(int, map(float, data.decode().split(" "))))

        # Limit velocity values
        ref_speed_L, ref_speed_R = np.clip(-1000, 1000, [ref_speed_L, ref_speed_R])

        # Update time variables
        finish = time.time()

        # Estimate actual speed
        encoder_L = gpg.get_motor_encoder(gpg.MOTOR_LEFT)
        encoder_R = gpg.get_motor_encoder(gpg.MOTOR_RIGHT)
        mea_speed_L = (encoder_L - last_encoder_left) / (finish - start)
        mea_speed_R = (encoder_R - last_encoder_right) / (finish - start)

        # Set velocities to motors
        gpg.set_motor_dps(gpg.MOTOR_LEFT, ref_speed_L)
        gpg.set_motor_dps(gpg.MOTOR_RIGHT, ref_speed_R)

        # Send data to simulation
        gpg_socket.sendto(
            f"{ref_speed_L} {mea_speed_L} 0.0 {ref_speed_R} {mea_speed_R} 0.0".encode(),
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
