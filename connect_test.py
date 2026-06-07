# connect_test.py
from djitellopy import Tello

tello = Tello()
tello.connect()

print(f"배터리: {tello.get_battery()}%")
print("연결 성공!")