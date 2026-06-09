from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time
import os
import threading
import signal
import sys

MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.25
CY_CLOSE = 300

stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\nCtrl+C detected! Landing...")
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
    cv2.line(display, (0, CY_CLOSE), (fw, CY_CLOSE), (255,0,255), 1)

    if det is not None:
        cx, cy, w, h, conf = det
        color = (0,0,255) if cy <= CY_CLOSE else (255,100,0)
        cv2.rectangle(display, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color, 2)
        cv2.circle(display, (cx, cy), 5, color, -1)
        cv2.putText(display, f"cylinder {conf:.2f}",
                   (cx-w//2, max(cy-h//2-10, 0)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        dx = cx - cx_s
        cv2.putText(display, f"dx:{dx}  cy:{cy}",
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    else:
        cv2.putText(display, "No cylinder",
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    h_val = tello.get_height()
    bat   = tello.get_battery()
    bat_color = (0,255,255) if bat > 30 else (0,0,255)
    cv2.putText(display, f"H:{h_val}cm  Bat:{bat}%",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bat_color, 2)
    cv2.putText(display, status,
               (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.imshow("Landing Test", display)
    cv2.waitKey(1)

def safe_move(tello, direction, cm):
    if stop_flag:
        return
    try:
        if direction == "forward": tello.move_forward(cm)
        elif direction == "back":  tello.move_back(cm)
        elif direction == "left":  tello.move_left(cm)
        elif direction == "right": tello.move_right(cm)
        elif direction == "up":    tello.move_up(cm)
        elif direction == "down":  tello.move_down(cm)
        time.sleep(1.5)
    except Exception as e:
        print(f"  Move error: {e}")

def align_lr(tello, model, max_try=3):
    """중앙 맞을 때까지 최대 max_try번 반복"""
    for attempt in range(max_try):
        if stop_flag:
            return False
        frame = get_frame()
        if frame is None:
            return False
        det = detect_cylinder(frame, model)
        show_frame(frame, det, f"Aligning {attempt+1}/{max_try}...", tello)
        if det is None:
            print(f"  Not detected - skip align")
            return False
        fw = frame.shape[1]
        cx_s = fw // 2
        cx, cy, w, h, conf = det
        dx = int(cx - cx_s)
        print(f"  Align {attempt+1}/{max_try}: dx={dx}")
        if abs(dx) <= 30:
            print(f"  Centered! dx={dx}")
            return True
        move_cm = max(20, min(abs(dx) // 5, 30))
        try:
            if dx > 0:
                print(f"  Right {move_cm}cm")
                tello.move_right(move_cm)
            else:
                print(f"  Left {move_cm}cm")
                tello.move_left(move_cm)
            time.sleep(1.5)
        except Exception as e:
            print(f"  Align skip: {e}")
            return False
    return False

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
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
time.sleep(3)
frame_read = tello.get_frame_read()
time.sleep(2)

t = threading.Thread(target=frame_capture_thread, args=(frame_read,), daemon=True)
t.start()
print("Frame thread started!")
time.sleep(1)

for _ in range(30):
    f = get_frame()
    if f is not None:
        print(f"Camera OK! {f.shape}")
        break
    time.sleep(0.2)

print("\n====================================")
print("  Auto Landing Test")
print("  원기둥 정면 100cm 에 놓기")
print("  3초 후 자동 시작!")
print("====================================")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

try:
    # 이륙
    print("\n[1] Takeoff!")
    tello.takeoff()
    print("Stabilizing 5sec...")
    time.sleep(5)

    if stop_flag:
        tello.land()
        sys.exit()

    # 현재 높이 유지 (하강 없음 - Auto land 방지)
    current_h = tello.get_height()
    print(f"Ready at {current_h}cm")

    # 탐지 - 안되면 1번만 20cm 하강 후 재탐지
    print("\n[2] Searching cylinder...")
    detected = False

    # 1차 탐지 시도
    for i in range(30):
        if stop_flag:
            break
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        det = detect_cylinder(frame, model)
        h_val = tello.get_height()
        show_frame(frame, det, f"Searching... H:{h_val}cm", tello)
        if det is not None:
            print(f"  Found! conf={det[4]:.2f} cy={det[1]} H:{h_val}cm")
            detected = True
            break
        time.sleep(0.1)

    # 1차 실패시 20cm 하강 후 2차 탐지
    if not detected and not stop_flag:
        h_val = tello.get_height()
        print(f"  Not found at H:{h_val}cm - descend 20cm and retry")
        try:
            tello.move_down(20)
            time.sleep(2)
        except Exception as e:
            print(f"  Descend error: {e}")

        for i in range(30):
            if stop_flag:
                break
            frame = get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            det = detect_cylinder(frame, model)
            h_val = tello.get_height()
            show_frame(frame, det, f"Searching (2nd)... H:{h_val}cm", tello)
            if det is not None:
                print(f"  Found! conf={det[4]:.2f} cy={det[1]} H:{h_val}cm")
                detected = True
                break
            time.sleep(0.1)

    if not detected or stop_flag:
        print("Not found or stopped! Landing...")
        tello.land()
    else:
        # ==========================================
        #  접근: 4번 x 20cm = 80cm
        #  스텝 1~2: 정렬(반복) + 전진
        #  스텝 3~4: 정렬 없이 직진
        # ==========================================
        print("\n[3] Approaching (4 x 25cm = 100cm)...")

        for step in range(4):
            if stop_flag:
                break

            print(f"\n  Step {step+1}/4")

            # 스텝 1~2만 좌우 정렬 반복
            if step < 2:
                print(f"  Aligning (step {step+1}/4)")
                align_lr(tello, model, max_try=3)
                if stop_flag:
                    break
            else:
                print(f"  No alignment - straight")

            # 전진 25cm
            print(f"  Forward 25cm")
            safe_move(tello, "forward", 25)

            current_h = tello.get_height()
            print(f"  H after: {current_h}cm")

            # 화면 표시
            frame = get_frame()
            if frame is not None:
                if step < 2:
                    det = detect_cylinder(frame, model)
                    show_frame(frame, det, f"Step {step+1}/4", tello)
                    if det is not None:
                        print(f"  cy={det[1]}")
                else:
                    show_frame(frame, None, f"Step {step+1}/4 straight", tello)

            time.sleep(0.5)

        if stop_flag:
            tello.land()
            sys.exit()

        # ==========================================
        #  하강
        # ==========================================
        print("\n[4] Descending...")
        while tello.get_height() > 20 and not stop_flag:
            frame = get_frame()
            if frame is not None:
                det = detect_cylinder(frame, model)
                h_val = tello.get_height()
                show_frame(frame, det, f"Descending H:{h_val}cm", tello)

            print(f"  Down 20cm (H:{tello.get_height()}cm)")
            safe_move(tello, "down", 20)

        # 착지
        print("\n[5] Landing!")
        tello.land()

except Exception as e:
    print(f"\nERROR: {e}")
    print("Emergency landing!")
    try:
        tello.land()
    except:
        pass

finally:
    stop_flag = True
    tello.streamoff()
    cv2.destroyAllWindows()
    print("\nDone!")