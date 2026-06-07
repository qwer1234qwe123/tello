from djitellopy import Tello
import cv2
import os
import time

save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

# 기존 사진 개수 확인
existing = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
print(f"Existing photos: {existing}")

# =============================================
#  대회 경로 기준 새 세션
#  C장애물 통과 후 → 도착점까지 실제 시점
# =============================================
NEW_PLANS = {
    "4": [
        (120, 100, "G_120h_100d"),   # C장애물 통과 직후
        (100, 100, "H_100h_100d"),   # 접근 중
    ],
    "5": [
        ( 75,  75, "I_75h_75d"),     # 착지 준비
        ( 50,  50, "J_50h_50d"),     # 착지 직전
    ],
    "6": [
        (100,  75, "K_100h_75d"),    # 중간 거리
        (120, 100, "L_120h_100d"),   # 높은 고도 중간 거리
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

def capture_photos(frame_read, tello, label, count_start, n=17):
    count = count_start
    print(f"\n  [CAPTURE] {label}  target:{n}")
    print(f"  SPACE=save | N=next | Q=quit")

    for _ in range(60):
        frame = frame_read.frame
        if frame is not None and frame.size > 0 and frame.shape[0] > 10:
            break
        time.sleep(0.1)
    else:
        print("  Camera timeout!")
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
        bat_color = (0,255,255) if bat > 30 else (0,0,255)

        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(display, f"Saved:{saved}/{n}  H:{h}cm  Bat:{bat}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bat_color, 2)
        cv2.putText(display, f"Total so far: {count}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(display, "SPACE=save  N=next  Q=quit",
                    (10, display.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if bat <= 30:
            cv2.putText(display, "!! LOW BATTERY !!",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

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
                print(f"  Done!")
                break

        elif key in (ord('n'), ord('N')):
            print(f"  Next ({count-count_start} saved)")
            break

        elif key in (ord('q'), ord('Q')):
            return count, True

    return count, False

def hand_capture(frame_read, tello, count_start, n=80):
    """이륙 없이 손으로 잡고 촬영"""
    count = count_start
    print(f"\n  [HAND CAPTURE] target:{n}")
    print(f"  드론 손으로 들고 다양한 각도로 촬영!")
    print(f"  SPACE=save | Q=quit")

    for _ in range(30):
        frame = frame_read.frame
        if frame is not None and frame.size > 0:
            break
        time.sleep(0.1)

    while True:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            time.sleep(0.03)
            continue

        display = frame.copy()
        saved = count - count_start
        fw, fh = display.shape[1], display.shape[0]
        cx, cy = fw//2, fh//2

        cv2.putText(display, "HAND CAPTURE MODE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        cv2.putText(display, f"Saved:{saved}/{n}  Total:{count}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(display, "Various angles! Close/Far/Left/Right",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.putText(display, "SPACE=save  Q=quit",
                    (10, fh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.line(display, (cx-40, cy), (cx+40, cy), (0,165,255), 1)
        cv2.line(display, (cx, cy-40), (cx, cy+40), (0,165,255), 1)
        cv2.circle(display, (cx, cy), 50, (0,165,255), 1)

        cv2.imshow("Tello Capture", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            fname = f"{save_dir}/HAND2_{count:04d}.jpg"
            cv2.imwrite(fname, frame)
            count += 1
            print(f"  Saved: {fname}  ({count-count_start}/{n})")
            if count - count_start >= n:
                print(f"  Done! {n} hand photos taken.")
                break

        elif key in (ord('q'), ord('Q')):
            break

    return count

# =============================================
#  메인
# =============================================
print("\n====================================")
print("  대회 경로 기준 추가 촬영")
print("  Session 4: G(175h/100d)  H(100h/100d)")
print("  Session 5: I(75h/75d)    J(50h/50d)")
print("  Session 6: K(100h/75d)   L(125h/100d)")
print("  Session H: 손으로 잡고 80장")
print("====================================")
print(f"\n기존 사진: {existing}장")
session = input("Select session (4/5/6/H): ").strip().upper()

tello = Tello()
tello.connect()
bat = tello.get_battery()
print(f"Battery: {bat}%")

tello.streamon()
print("Stream ON - waiting 3sec...")
time.sleep(3)

frame_read = tello.get_frame_read()
time.sleep(2)

for _ in range(30):
    f = frame_read.frame
    if f is not None and f.size > 0 and f.shape[0] > 10:
        print(f"Camera OK! {f.shape}")
        break
    time.sleep(0.2)

# =============================================
#  손으로 잡고 촬영 (H 세션)
# =============================================
if session == "H":
    print("\n손으로 잡고 촬영 모드 (이륙 없음)")
    total_count = hand_capture(frame_read, tello, existing, n=80)
    tello.streamoff()
    cv2.destroyAllWindows()
    new_count = total_count - existing
    print(f"\n새로 찍은 사진: {new_count}장")
    print(f"전체 사진: {total_count}장")
    exit()

# =============================================
#  드론 비행 세션 (4/5/6)
# =============================================
if session not in NEW_PLANS:
    print("Invalid session!")
    tello.streamoff()
    exit()

if bat < 40:
    print(f"WARNING: Battery too low ({bat}%)!")
    tello.streamoff()
    exit()

capture_plan = NEW_PLANS[session]
print(f"\nSession {session}: {[p[2] for p in capture_plan]}")

print("\nTakeoff!")
tello.takeoff()
print("Stabilizing after takeoff - waiting 4sec...")
time.sleep(4)  # 이륙 후 충분히 대기

# 카메라 워밍업
print("Camera warming up...")
for _ in range(30):
    f = frame_read.frame
    if f is not None and f.size > 0 and f.shape[0] > 10:
        break
    time.sleep(0.1)

current = tello.get_height()
print(f"Height: {current}cm -> 100cm")
if current < 90:
    time.sleep(1)
    safe_move(tello, "up", 100 - current)
elif current > 110:
    time.sleep(1)
    safe_move(tello, "down", current - 100)
time.sleep(1.5)

total_count = existing
quit_flag   = False
prev_forward = 0

for i, (height, forward, label) in enumerate(capture_plan):
    if quit_flag:
        break

    bat = tello.get_battery()
    if bat < 25:
        print(f"\nBattery critical ({bat}%)! Landing.")
        break

    print(f"\n[{i+1}/{len(capture_plan)}] {label}")
    print(f"  height:{height}cm  distance:{forward}cm  bat:{bat}%")

    if prev_forward > 0:
        print(f"  Return forward {prev_forward}cm")
        safe_move(tello, "forward", prev_forward)

    current_h = tello.get_height()
    diff = height - current_h
    if diff > 10:
        safe_move(tello, "up", diff)
    elif diff < -10:
        safe_move(tello, "down", abs(diff))
    time.sleep(0.5)

    print(f"  Back {forward}cm")
    safe_move(tello, "back", forward)
    prev_forward = forward
    time.sleep(0.5)

    total_count, quit_flag = capture_photos(
        frame_read, tello, label, total_count, n=17
    )

print(f"\nTotal: {total_count}장 (new: {total_count-existing}장)")
print("Landing...")
if prev_forward > 0:
    safe_move(tello, "forward", prev_forward)
tello.land()
tello.streamoff()
cv2.destroyAllWindows()
print(f"Saved to: {os.path.abspath(save_dir)}")