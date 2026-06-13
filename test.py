from djitellopy import Tello
import time

tello = Tello()
tello.connect()

bat = tello.get_battery()
print(f"Battery: {bat}%")

if bat < 20:
    print("Battery too low! Exit program.")
    tello.end()
    exit()

print("Battery OK - safe to fly")