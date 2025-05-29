# https://www.dexterindustries.com
# https://github.com/ieee-ras-ufcg/GoRASGo
#
# Copyright (c) 2017 Dexter Industries
# Released under the MIT license (http://choosealicense.com/licenses/mit/).
# For more information, see https://github.com/DexterInd/GoPiGo3/blob/master/LICENSE.md
#
# This code is for power management on a Raspberry Pi with GoPiGo3.
#
# GPIO 22 will be configured as input with pulldown. If pulled high, the RPi will halt.
#
# GPIO 23 needs to remain low impedance (output) set to a HIGH state. If GPIO 23 gets
# left floating (high impedance) the GoPiGo3 assumes the RPi has shut down fully.
# SW should never write GPIO 23 to LOW or set it as an INPUT.

import os
import time
import RPi.GPIO as GPIO

# Set the numbering mode for referencing GPIO pins
GPIO.setmode(GPIO.BCM)

# Set pin 22 as input with pulldown
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Set pin 23 as output
GPIO.setup(23, GPIO.OUT)

# Write HIGH to activate board - the green LED shall stop blinking
GPIO.output(23, True)

while not GPIO.input(22):
    time.sleep(0.1)  # Wait until next read

# Turn off pin and turn off rasp
GPIO.output(23, False)
os.system("shutdown now -h")
