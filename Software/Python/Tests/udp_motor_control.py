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

try:
    print("[INFO] Running... Press Ctrl+C to stop")

    while True:
        # Receive UPD message
        data, address = gpg_socket.recvfrom(1024) 

        # Parse wheel velocities
        phi_dot_L, phi_dot_R = map(int, map(float, data.decode().split(" ")))

        # Limit velocity values
        phi_dot_L, phi_dot_R = np.clip(-1000, 1000, [phi_dot_L, phi_dot_R])

        # Set velocities to motors
        gpg.set_motor_dps(gpg.MOTOR_LEFT, phi_dot_L)
        gpg.set_motor_dps(gpg.MOTOR_RIGHT, phi_dot_R)

except KeyboardInterrupt:
    print("\n[INFO] Execution stopped externally")
    print("[INFO] Motors shutdown")
    gpg.set_motor_dps(gpg.MOTOR_LEFT, 0)
    gpg.set_motor_dps(gpg.MOTOR_RIGHT, 0)
