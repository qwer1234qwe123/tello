from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time
import os
import threading
import signal
import sys

# =============================================
#  대회 룰 기반 비행 코드
#  Start → A → B → C → 원기둥 착지
#  Ctrl+C: 즉시 착지
# =============================================

MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.25

stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\n[!] Ctrl+C - Emergency landing!")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)

latest_frame = None
frame_lock = threading.Lock()

def frame_capture_thread(frame_read):
    while not stop_flag:
        frame = frame_read.frame
        if frame is not None and frame.size > 0:
            with frame_lock:
                global latest_frame
                latest_frame = frame.copy()
        time.sleep(0.03)

def get_frame():
    with frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return None

def detect_cylinder(frame, model):
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    best_box = None
    best_conf = 0
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w  = int(x2 - x1)
            h  = int(y2 - y1)
            best_box = (cx, cy, w, h, float(conf))
    return best_box

def show_frame(frame, det, status, tello):
    display = frame.copy()
    fw, fh = display.shape[1], display.shape[0]
    cx_s, cy_s = fw//2, fh//2
    cv2.line(display, (cx_s-50, cy_s), (cx_s+50, cy_s), (0,255,0), 1)
    cv2.line(display, (cx_s, cy_s-50), (cx_s, cy_s+50), (0,255,0), 1)
    cv2.circle(display, (cx_s, cy_s), 60, (0,255,0), 1)
    if det is not None:
        cx, cy, w, h, conf = det
        cv2.rectangle(display, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (0,0,255), 2)
        cv2.putText(display, f"cylinder {conf:.2f}",
                   (cx-w//2, max(cy-h//2-10, 0)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
        dx = cx - cx_s
        cv2.putText(display, f"dx:{dx} cy:{cy}",
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    h_val = tello.get_height()
    bat   = tello.get_battery()
    bat_color = (0,255,255) if bat > 30 else (0,0,255)
    cv2.putText(display, f"H:{h_val}cm Bat:{bat}%",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bat_color, 2)
    cv2.putText(display, status,
               (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.imshow("Tello Flight", display)
    cv2.waitKey(1)

def safe_move(tello, direction, cm, wait=2.0):
    if stop_flag:
        return
    while cm > 0:
        step = min(cm, 100)
        try:
            if direction == "forward": tello.move_forward(step)
            elif direction == "back":  tello.move_back(step)
            elif direction == "left":  tello.move_left(step)
            elif direction == "right": tello.move_right(step)
            elif direction == "up":    tello.move_up(step)
            elif direction == "down":  tello.move_down(step)
            time.sleep(wait)
            if direction == "forward":
                tello.send_rc_control(0, 0, 0, 0)  # 정지 명령으로 관성 제동
                time.sleep(0.3)            
        except Exception as e:
            print(f"  Move error ({direction} {step}cm): {e}")
        cm -= step

def safe_rotate(tello, degrees, wait=2.0):
    if stop_flag:
        return
    try:
        if degrees > 0:
            tello.rotate_clockwise(degrees)
        else:
            tello.rotate_counter_clockwise(abs(degrees))
        time.sleep(wait)
    except Exception as e:
        print(f"  Rotate error: {e}")

def adjust_height(tello, target_cm):
    if stop_flag:
        return
    current = tello.get_height()
    diff = target_cm - current
    print(f"  Height: {current}cm -> {target_cm}cm")
    if diff > 10:
        safe_move(tello, "up", diff)
    elif diff < -10:
        safe_move(tello, "down", abs(diff))
    time.sleep(1)

def check_stop():
    if stop_flag:
        raise KeyboardInterrupt

def align_lr(tello, model, max_try=3):
    for attempt in range(max_try):
        if stop_flag:
            return False
        frame = get_frame()
        if frame is None:
            return False
        det = detect_cylinder(frame, model)
        show_frame(frame, det, f"Aligning {attempt+1}/{max_try}", tello)
        if det is None:
            return False
        fw = frame.shape[1]
        cx_s = fw // 2
        cx, cy, w, h, conf = det
        dx = int(cx - cx_s)
        print(f"  Align {attempt+1}: dx={dx}")
        if abs(dx) <= 30:
            print(f"  Centered!")
            return True
        move_cm = max(20, min(abs(dx) // 5, 30))
        try:
            if dx > 0:
                tello.move_right(move_cm)
            else:
                tello.move_left(move_cm)
            time.sleep(1.5)
        except Exception as e:
            print(f"  Align skip: {e}")
            return False
    return False

def emergency_land(tello):
    print("[!] Emergency landing!")
    for _ in range(5):
        try:
            tello.land()
            break
        except:
            time.sleep(1)

# =============================================
#  메인
# =============================================
print("Loading YOLO model...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found!")
    exit()
model = YOLO(MODEL_PATH)
print("Model loaded!")

tello = Tello()
tello.connect()
bat = tello.get_battery()
print(f"Battery: {bat}%")

if bat < 10:
    print(f"Battery too low ({bat}%)!")
    exit()

tello.streamon()
print("Stream ON - waiting 3sec...")
time.sleep(3)
frame_read = tello.get_frame_read()
time.sleep(2)

t = threading.Thread(target=frame_capture_thread, args=(frame_read,), daemon=True)
t.start()
time.sleep(1)

for _ in range(30):
    f = get_frame()
    if f is not None:
        print(f"Camera OK! {f.shape}")
        break
    time.sleep(0.2)

print("\n====================================")
print("  대회 비행 코드 v2")
print("  Start → A → B → C → 착지")
print("  Ctrl+C: 즉시 착지")
print("====================================")
print("\n장애물 통과 높이:")
print("  A번: 100~175cm → 통과높이 130cm")
print("  B번:  50~125cm → 통과높이  80cm")
print("  C번: 100~175cm → 통과높이 130cm")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

try:
    # ==========================================
    #  [1] 이륙 + 1m 호버링 1초 (10점)
    # ==========================================
    print("\n[1] Takeoff & Hovering (10점)")
    tello.takeoff()
    print("  Stabilizing 3sec...")
    time.sleep(3)
    check_stop()

    # 이륙 후 고정 상승
    current_h = tello.get_height()
    print(f"  Current height: {current_h}cm")

    time.sleep(2)

    try:
        print("  Up 30cm")
        tello.move_up(30)
        time.sleep(2)

        print("  Up 20cm")
        tello.move_up(20)
        time.sleep(2)

    except Exception as e:
        print(f"  Up error: {e}")

    print(f"  Ready at {tello.get_height()}cm")
    time.sleep(1)
    check_stop()

    # ==========================================
    #  [2] A장애물 통과 (20점)
    #  높이 110cm / 전진 120cm
    # ==========================================
    print("\n[2] Obstacle A (20점)")
    print(f"  Height: {tello.get_height()}cm - Forward 120cm")
    safe_move(tello, "forward", 70)
    safe_move(tello, "forward", 70)
    check_stop()
    print("  A passed!")
    time.sleep(1)

    # ==========================================
    #  [3] B장애물 통과 (20점)
    #  하강 80cm → 전진 120cm
    # ==========================================
    print("\n[3] Obstacle B (20점)")
    print("  Descending to 80cm...")
    safe_move(tello, "down", 30)   
    check_stop()
    time.sleep(0.8)
    print(f"  Current height: {tello.get_height()}cm")
    print("  Height 80cm - Forward 100cm")
    safe_move(tello, "forward", 50)
    check_stop()
    time.sleep(0.8)
    print("  B passed!")

    # ==========================================
    #  [4] C장애물 통과 (20점)
    #  B통과 후 우회전 90도
    #  130cm로 상승 → 전진 100cm
    # ==========================================
    print("\n[4] Obstacle C (20점)")
    safe_rotate(tello, 90)     # 우회전 90도
    check_stop()
    safe_move(tello, "forward", 30)
    time.sleep(0.8)
    check_stop()
    adjust_height(tello, 110)  # 고도 변경 2회 (80 → 130)
    check_stop()
    print("  Height 130cm - Forward 130cm")
    safe_move(tello, "forward", 80)
    time.sleep(0.8)
    check_stop()
    #safe_move(tello, "forward", 60)
    #check_stop()
    print("  C passed!")

    # ==========================================
    #  [5] 원기둥 착지 (30점)
    # ==========================================
    print("\n[5] Landing on cylinder (30점)")

    # 착지 탐지 높이
    adjust_height(tello, 40)
    print(f"Landing start height = {tello.get_height()}cm")

    check_stop()
    time.sleep(1)

    # 1차 탐지
    detected = False
    for i in range(30):
        if stop_flag:
            break

        frame = get_frame()

        if frame is None:
            time.sleep(0.1)
            continue

        det = detect_cylinder(frame, model)
        h_val = tello.get_height()

        show_frame(frame, det, f"Searching H:{h_val}cm", tello)

        if det is not None:
            print(f"  Found! conf={det[4]:.2f}")
            detected = True
            break

        time.sleep(0.1)

    if detected and not stop_flag:
        for step in range(5):
            if stop_flag:
                break

            print(f"\n  Step {step+1}/5")

            # ==================================
            # Step 1~3
            # 전진 → 하강 → 탐색
            # 탐지 실패 시 좌우 탐색
            # ==================================
            if step == 0:
                print("  Forward 30cm")
                safe_move(tello, "forward", 20)

                try:
                    tello.move_down(20)
                    time.sleep(0.8)
                except Exception as e:
                    print(f"  Down skip: {e}")

                frame = get_frame()
                det = detect_cylinder(frame, model) if frame is not None else None

                if det is None:
                    print("  Search Left")
                    safe_move(tello, "left", 20)

                    frame = get_frame()
                    det = detect_cylinder(frame, model) if frame is not None else None

                    if det is None:
                        print("  Search Right")
                        safe_move(tello, "right", 40)

                # 탐지 후 정렬 (Step 4와 동일 로직)
                if stop_flag:
                    break

                frame = get_frame()

                if frame is not None:
                    det = detect_cylinder(frame, model)
                    show_frame(frame, det, f"Step {step+1}/5 align", tello)

                print("  Aligning cylinder...")
                align_lr(tello, model, max_try=1)
                print(f"  H after: {tello.get_height()}cm")
                
            elif step == 1:
                print("  Forward 25cm")
                safe_move(tello, "forward", 20) # 오타 수정 완료 (afe_move -> safe_move)

                try:
                    tello.move_down(10)
                    time.sleep(0.8)
                except Exception as e:
                    print(f"  Down skip: {e}")

                frame = get_frame()
                det = detect_cylinder(frame, model) if frame is not None else None

                if det is None:
                    print("  Search Left")
                    safe_move(tello, "left", 20)

                    frame = get_frame()
                    det = detect_cylinder(frame, model) if frame is not None else None

                    if det is None:
                        print("  Search Right")
                        safe_move(tello, "right", 40)
            
            elif step == 2:
                frame = get_frame()
                det = detect_cylinder(frame, model) if frame is not None else None

                if det is not None:
                    print(f"  Cylinder detected conf={det[4]:.2f}")
                else:
                    print("  Cylinder not detected - continue")

                try:
                    tello.move_down(10)
                    time.sleep(0.8)
                except Exception as e:
                    print(f"  Down skip: {e}")

                print("  Forward 25cm")
                safe_move(tello, "forward", 20)
                print(f"  H after: {tello.get_height()}cm")

            # ==================================
            # Step 4
            # 전진 → 탐지 → 정렬
            # ==================================
            elif step == 3:
                print("  Forward 25cm")
                safe_move(tello, "forward", 20)

                if stop_flag:
                    break

                frame = get_frame()

                if frame is not None:
                    det = detect_cylinder(frame, model)
                    show_frame(frame, det, f"Step {step+1}/5 align", tello)

                print("  Aligning cylinder...")
                align_lr(tello, model, max_try=1)
                print(f"  H after: {tello.get_height()}cm")

            # ==================================
            # Step 5
            # 전진 → 강제착륙
            # ==================================
            else:
                print("  Forward 25cm")
                safe_move(tello, "forward", 20)
                print("\n  FORCE LANDING")
                tello.land()
                break

            time.sleep(0.5)

        else: # for 루프가 정상 종료되었을 때 (break 되지 않았을 때)
            print("\n  Landing!")
            tello.land()

    else: # 1차 탐지 실패 시 룰 기반 착지
        print("  Not found - rule based landing")
        adjust_height(tello, 40)
        check_stop()
        safe_move(tello, "forward", 80)
        print("  Landing!")
        tello.land()

except KeyboardInterrupt:
    emergency_land(tello)

except Exception as e:
    print(f"\nERROR: {e}")
    emergency_land(tello)

finally:
    stop_flag = True
    try:
        tello.streamoff()
    except:
        pass
    cv2.destroyAllWindows()
    print("\nDone!")