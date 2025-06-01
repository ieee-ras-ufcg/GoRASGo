import time
from gopigo3 import GoPiGo3

gpg = GoPiGo3()

gpg.set_motor_power(gpg.MOTOR_LEFT, 100)
gpg.set_motor_power(gpg.MOTOR_RIGHT, 100)

time.sleep(1)

gpg.set_motor_power(gpg.MOTOR_LEFT, 0)
gpg.set_motor_power(gpg.MOTOR_RIGHT, 0)
