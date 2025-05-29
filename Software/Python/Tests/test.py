import time
from gopigo3 import GoPiGo3
import RPi.GPIO

gpg = GoPiGo3()

gpg.set_motor_power(gpg.MOTOR_LEFT, 100)
gpg.set_motor_power(gpg.MOTOR_RIGHT, 100)

time.sleep(2)

gpg.set_motor_power(gpg.MOTOR_LEFT, 0)
gpg.set_motor_power(gpg.MOTOR_RIGHT, 0)
