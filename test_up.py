# test_up.py
from djitellopy import Tello
import time

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.takeoff()
print(f"Height after takeoff: {tello.get_height()}")

for wait in [3, 5, 7, 10]:
    time.sleep(1)
    try:
        tello.move_up(20)
        print(f"Up OK after {wait}sec wait! H:{tello.get_height()}")
        break
    except Exception as e:
        print(f"Wait {wait}sec - Up failed: {e}")

tello.land()