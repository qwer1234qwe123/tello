from djitellopy import Tello
import cv2
import os
import time

save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

# =============================================
#  촬영 목록 - 비행 세션별로 나눔
#  한 세션당 2~3개만 촬영 (배터리 절약)
# =============================================
ALL_PLANS = {
    "1": [
        (100, 150, "A_100h_150d"),
        (100, 100, "B_100h_100d"),
    ],
    "2": [
        ( 50, 100, "C_50h_100d"),
        (150, 150, "D_150h_150d"),
    ],
    "3": [
        ( 50,  50, "E_50h_50d"),
        (100, 200, "F_100h_200d"),
    ],
}

def safe_move(tello, direction, cm):
    while cm > 0:
        step = min(cm, 100)
        if direction == "up":        tello.move_up(step)
        elif direction == "down":    tello.move_down(step)
        elif direction == "forward": tello.move_forward(step)
        elif direction == "back":    tello.move_back(step)
        cm -= step
        time.sleep(1.5)

def capture_photos(frame_read, tello, label, count_start, n=15):
    count = count_start
    print(f"\n  [CAPTURE] {label}  target:{n}")
    print(f"  SPACE=save | N=next | Q=quit")

    # 유효 프레임 대기
    for _ in range(60):
        frame = frame_read.frame
        if frame is not None and frame.size > 0 and frame.shape[0] > 10:
            break
        time.sleep(0.1)
    else:
        print("  Camera timeout! Skipping...")
        return count, False

    while True:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            time.sleep(0.03)
            continue

        display = frame.copy()
        saved = count - count_start
        h   = tello.get_height()
        bat = tello.get_battery()

        # 배터리 낮으면 경고 색상
        bat_color = (0, 255, 255) if bat > 30 else (0, 0, 255)

        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(display, f"Saved:{saved}/{n}  H:{h}cm  Bat:{bat}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bat_color, 2)
        cv2.putText(display, "SPACE=save  N=next  Q=quit",
                    (10, display.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # 배터리 30% 이하 경고
        if bat <= 30:
            cv2.putText(display, "!! LOW BATTERY - LAND SOON !!",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        fw, fh = display.shape[1], display.shape[0]
        cx, cy = fw//2, fh//2
        cv2.line(display, (cx-40, cy), (cx+40, cy), (0,255,0), 1)
        cv2.line(display, (cx, cy-40), (cx, cy+40), (0,255,0), 1)
        cv2.circle(display, (cx, cy), 50, (0,255,0), 1)

        cv2.imshow("Tello Capture", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            fname = f"{save_dir}/{label}_{count:04d}.jpg"
            cv2.imwrite(fname, frame)
            count += 1
            print(f"  Saved: {fname}  ({count-count_start}/{n})")
            if count - count_start >= n:
                print(f"  Done! {n} photos taken.")
                break

        elif key in (ord('n'), ord('N')):
            print(f"  Next ({count-count_start} saved)")
            break

        elif key in (ord('q'), ord('Q')):
            return count, True

    return count, False


# =============================================
#  메인 - 세션 선택
# =============================================
print("\n====================================")
print("  Tello Capture Guide")
print("  Session 1: A(100h/150d)  B(100h/100d)")
print("  Session 2: C(50h/100d)   D(150h/150d)")
print("  Session 3: E(50h/50d)    F(100h/200d)")
print("====================================")
print("\nCharge battery to 80%+ before each session!")
session = input("Select session (1/2/3): ").strip()

if session not in ALL_PLANS:
    print("Invalid session!")
    exit()

capture_plan = ALL_PLANS[session]
print(f"\nSession {session} selected: {[p[2] for p in capture_plan]}")

# 드론 연결
tello = Tello()
tello.connect()
bat = tello.get_battery()
print(f"Battery: {bat}%")

if bat < 40:
    print(f"WARNING: Battery too low ({bat}%)! Please charge first.")
    exit()

tello.streamon()
print("Stream ON - waiting 3sec...")
time.sleep(3)

frame_read = tello.get_frame_read()
time.sleep(2)

# 카메라 확인
for _ in range(30):
    f = frame_read.frame
    if f is not None and f.size > 0 and f.shape[0] > 10:
        print(f"Camera OK! {f.shape}")
        break
    time.sleep(0.2)

# 이륙
print("\nTakeoff!")
tello.takeoff()
time.sleep(2.5)

# 기본 높이 100cm
current = tello.get_height()
print(f"Height: {current}cm -> 100cm")
if current < 90:
    safe_move(tello, "up", 100 - current)
elif current > 110:
    safe_move(tello, "down", current - 100)
time.sleep(1)

total_count = 0
quit_flag   = False
prev_forward = 0

for i, (height, forward, label) in enumerate(capture_plan):
    if quit_flag:
        break

    # 배터리 체크
    bat = tello.get_battery()
    if bat < 25:
        print(f"\nBattery critical ({bat}%)! Landing now.")
        break

    print(f"\n[{i+1}/{len(capture_plan)}] {label}")
    print(f"  height:{height}cm  distance:{forward}cm  battery:{bat}%")

    # 원점 복귀
    if prev_forward > 0:
        print(f"  Return forward {prev_forward}cm")
        safe_move(tello, "forward", prev_forward)

    # 높이 조정
    current_h = tello.get_height()
    diff = height - current_h
    if diff > 10:
        safe_move(tello, "up", diff)
    elif diff < -10:
        safe_move(tello, "down", abs(diff))
    time.sleep(0.5)

    # 후진
    print(f"  Back {forward}cm")
    safe_move(tello, "back", forward)
    prev_forward = forward
    time.sleep(0.5)

    total_count, quit_flag = capture_photos(
        frame_read, tello, label, total_count, n=15
    )

# 착지
print(f"\nTotal: {total_count} photos saved!")
print("Landing...")
if prev_forward > 0:
    safe_move(tello, "forward", prev_forward)
tello.land()
tello.streamoff()
cv2.destroyAllWindows()
print(f"Saved to: {os.path.abspath(save_dir)}")
print(f"\nNext: run session again with different number!")