from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time
import os

MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.2

# cy 기준 (화면 세로 720px 기준)
# 원기둥 cy가 이 값 이하면 충분히 가까운 것 → 하강
CY_CLOSE = 400

def safe_move(tello, direction, cm):
    while cm > 0:
        step = min(cm, 100)
        if direction == "up":        tello.move_up(step)
        elif direction == "down":    tello.move_down(step)
        elif direction == "forward": tello.move_forward(step)
        elif direction == "back":    tello.move_back(step)
        elif direction == "left":    tello.move_left(step)
        elif direction == "right":   tello.move_right(step)
        cm -= step
        time.sleep(1.5)

frame_skip_count = 0
last_det = None

def get_fresh_frame(frame_read):
    for _ in range(10):
        frame = frame_read.frame
        if frame is not None and frame.size > 0:
            return frame.copy()
        time.sleep(0.05)
    return None

def detect_cylinder(frame, model):
    global frame_skip_count, last_det
    frame_skip_count += 1
    # 3프레임마다 1번만 YOLO 실행
    if frame_skip_count % 3 != 0:
        return last_det

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
    last_det = best_box
    return best_box

def show_frame(frame, det, status, tello):
    display = frame.copy()
    fw, fh = display.shape[1], display.shape[0]
    cx_s, cy_s = fw//2, fh//2

    cv2.line(display, (cx_s-50, cy_s), (cx_s+50, cy_s), (0,255,0), 1)
    cv2.line(display, (cx_s, cy_s-50), (cx_s, cy_s+50), (0,255,0), 1)
    cv2.circle(display, (cx_s, cy_s), 60, (0,255,0), 1)
    cv2.line(display, (0, CY_CLOSE), (fw, CY_CLOSE), (255,0,255), 1)
    cv2.putText(display, f"CLOSE LINE", (fw-150, CY_CLOSE-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,255), 1)

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

def align_lr(tello, frame_read, model):
    frame = get_fresh_frame(frame_read)
    if frame is None:
        return
    det = detect_cylinder(frame, model)
    show_frame(frame, det, "Aligning LR...", tello)
    if det is None:
        return
    fw = frame.shape[1]
    cx_s = fw // 2
    cx, cy, w, h, conf = det
    dx = int(cx - cx_s)
    margin = 80
    if abs(dx) > margin:
        move_cm = max(20, min(abs(dx) // 7, 50))  # 최소 20cm
        try:
            if dx > 0:
                print(f"  Right {move_cm}cm (dx={dx})")
                tello.move_right(move_cm)
            else:
                print(f"  Left {move_cm}cm (dx={dx})")
                tello.move_left(move_cm)
            time.sleep(1.2)
        except Exception as e:
            print(f"  Align error: {e} - skip")

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
print("Stream ON - waiting 3sec...")
time.sleep(3)

frame_read = tello.get_frame_read()
time.sleep(2)

for _ in range(30):
    f = frame_read.frame
    if f is not None and f.size > 0:
        print(f"Camera OK! {f.shape}")
        break
    time.sleep(0.1)

print("\n====================================")
print("  Auto Landing Test")
print("  원기둥 정면 100cm, 높이 30cm 시작")
print("  3초 후 자동 시작!")
print("====================================")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

try:
    # ==========================================
    #  이륙
    # ==========================================
    print("\n[1] Takeoff!")
    tello.takeoff()
    print("Stabilizing 7sec...")
    time.sleep(7)  # 충분히 대기

    # ==========================================
    #  30cm로 낮추기
    # ==========================================
    current_h = tello.get_height()
    print(f"Height: {current_h}cm -> 30cm")
    if current_h > 40:
        diff = current_h - 30
        print(f"  Down {diff}cm - waiting 2sec before move...")
        time.sleep(2)  # 하강 전 추가 대기
        tello.move_down(min(diff, 100))
        time.sleep(3)  # 하강 후 안정화
    print(f"Ready at {tello.get_height()}cm")

    # ==========================================
    #  카메라 켜서 원기둥 탐지
    # ==========================================
    print("\n[2] Searching cylinder...")
    detected = False
    for i in range(60):  # 최대 6초
        frame = get_fresh_frame(frame_read)
        if frame is None:
            continue
        det = detect_cylinder(frame, model)
        show_frame(frame, det, f"Searching {i+1}/60", tello)
        if det is not None:
            print(f"  Found! conf={det[4]:.2f} cy={det[1]}")
            detected = True
            break
        time.sleep(0.1)

    if not detected:
        print("Not found! Landing...")
        tello.land()
    else:
        # ==========================================
        #  전진 루프
        #  cy > CY_CLOSE → 멀다 → 전진 + 좌우정렬
        #  cy <= CY_CLOSE → 가깝다 → 하강
        #  탐지 안됨 → 매우 가까움 → 하강
        # ==========================================
        print("\n[3] Approaching cylinder...")
        not_found_count = 0

        while True:
            frame = get_fresh_frame(frame_read)
            if frame is None:
                continue

            det = detect_cylinder(frame, model)
            show_frame(frame, det, "Approaching...", tello)

            if det is None:
                not_found_count += 1
                print(f"  Not detected ({not_found_count}/3) - very close!")
                if not_found_count >= 3:
                    print("  Start descending!")
                    break
                time.sleep(0.2)
                continue

            not_found_count = 0
            cx, cy, w, h, conf = det
            print(f"  cy={cy} (close<={CY_CLOSE})")

            # 좌우 정렬
            align_lr(tello, frame_read, model)

            # cy가 기준선 위로 올라오면 하강
            if cy <= CY_CLOSE:
                print(f"  Close! cy={cy} -> descend")
                break

            # 전진 30cm
            print(f"  Forward 30cm (cy={cy})")
            try:
                tello.move_forward(30)
                time.sleep(1.5)
            except Exception as e:
                print(f"  Forward error: {e}")

        # ==========================================
        #  하강 루프
        # ==========================================
        print("\n[4] Descending...")
        while tello.get_height() > 20:
            frame = get_fresh_frame(frame_read)
            if frame is None:
                continue

            det = detect_cylinder(frame, model)
            h_val = tello.get_height()
            show_frame(frame, det, f"Descending H:{h_val}cm", tello)

            if det is not None:
                align_lr(tello, frame_read, model)

            print(f"  Down 20cm (H:{tello.get_height()}cm)")
            try:
                tello.move_down(20)
                time.sleep(1.5)
            except Exception as e:
                print(f"  Down error: {e}")

        # ==========================================
        #  착지
        # ==========================================
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
    tello.streamoff()
    cv2.destroyAllWindows()
    print("\nDone!")