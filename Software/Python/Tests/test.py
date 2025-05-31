import time
from gopigo3 import GoPiGo3

gpg = GoPiGo3()

gpg.set_motor_power(gpg.MOTOR_LEFT, 127)
gpg.set_motor_power(gpg.MOTOR_RIGHT, 127)

time.sleep(1)

gpg.set_motor_power(gpg.MOTOR_LEFT, 0)
gpg.set_motor_power(gpg.MOTOR_RIGHT, 0)
