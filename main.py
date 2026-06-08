from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time
import os

# =============================================
#  대회 비행 메인 코드
#  Start → A장애물 → B장애물 → C장애물 → 착지
# =============================================

MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.5  # YOLO 탐지 신뢰도 임계값

# =============================================
#  안전 이동 함수 (100cm 제한 분할)
# =============================================
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

# =============================================
#  YOLO 원기둥 탐지 함수
#  반환: (cx, cy, w, h) 또는 None
# =============================================
def detect_cylinder(frame, model):
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    best_box = None
    best_conf = 0

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w  = x2 - x1
            h  = y2 - y1
            best_box = (cx, cy, w, h)

    return best_box

# =============================================
#  원기둥 정렬 함수
#  화면 중앙으로 드론 이동
# =============================================
def align_to_cylinder(tello, frame, model, max_attempts=30):
    print("  Aligning to cylinder...")
    fw = frame.shape[1]
    fh = frame.shape[0]
    center_x = fw // 2
    center_y = fh // 2
    margin = 60  # 중앙 허용 범위 (픽셀)

    for attempt in range(max_attempts):
        frame = tello.get_frame_read().frame
        det = detect_cylinder(frame, model)

        if det is None:
            print(f"  Attempt {attempt+1}: cylinder not found")
            time.sleep(0.3)
            continue

        cx, cy, w, h = det
        dx = cx - center_x  # 양수: 오른쪽, 음수: 왼쪽
        dy = cy - center_y  # 양수: 아래, 음수: 위

        print(f"  Attempt {attempt+1}: cx={cx} cy={cy} dx={dx} dy={dy}")

        # 정렬 완료 확인
        if abs(dx) < margin and abs(dy) < margin:
            print("  Aligned!")
            return True

        # 좌우 정렬
        if abs(dx) > margin:
            move_cm = min(abs(dx) // 10, 30)
            if dx > 0:
                tello.move_right(move_cm)
            else:
                tello.move_left(move_cm)
            time.sleep(0.8)

        # 상하 정렬 (화면 상하 = 드론 전후)
        if abs(dy) > margin:
            move_cm = min(abs(dy) // 10, 20)
            if dy > 0:
                tello.move_forward(move_cm)
            else:
                tello.move_back(move_cm)
            time.sleep(0.8)

    print("  Alignment failed - proceeding anyway")
    return False

# =============================================
#  착지 함수
#  YOLO로 탐지하면서 서서히 하강
# =============================================
def land_on_cylinder(tello, frame_read, model):
    print("\n[LANDING] Starting precision landing...")

    # 1단계: 원기둥 탐지 및 정렬
    frame = frame_read.frame
    det = detect_cylinder(frame, model)

    if det is None:
        print("  Cylinder not detected! Landing at current position...")
        tello.land()
        return

    print(f"  Cylinder detected! w={det[2]} h={det[3]}")

    # 2단계: 정렬
    align_to_cylinder(tello, frame, model)

    # 3단계: 현재 높이 확인 후 하강
    height = tello.get_height()
    print(f"  Current height: {height}cm")

    # 100cm 이상이면 정렬하면서 하강
    while height > 100:
        frame = frame_read.frame
        det = detect_cylinder(frame, model)

        if det:
            align_to_cylinder(tello, frame, model, max_attempts=5)

        safe_move(tello, "down", 30)
        height = tello.get_height()
        print(f"  Height: {height}cm")
        time.sleep(0.5)

    # 4단계: 100cm 이하면 마지막 정렬 후 착지
    print("  Final alignment before landing...")
    frame = frame_read.frame
    align_to_cylinder(tello, frame, model, max_attempts=10)

    print("  Landing!")
    tello.land()


# =============================================
#  메인 비행 루프
# =============================================
def main():
    # YOLO 모델 로드
    print("Loading YOLO model...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)
    print("Model loaded!")

    # 텔로 연결
    tello = Tello()
    tello.connect()
    bat = tello.get_battery()
    print(f"Battery: {bat}%")

    if bat < 50:
        print("WARNING: Battery too low!")
        return

    tello.streamon()
    time.sleep(3)
    frame_read = tello.get_frame_read()
    time.sleep(2)

    # 카메라 확인
    for _ in range(30):
        f = frame_read.frame
        if f is not None and f.size > 0:
            print(f"Camera OK! {f.shape}")
            break
        time.sleep(0.1)

    try:
        # ==========================================
        #  이륙 및 호버링 (단계 1: +10점)
        # ==========================================
        print("\n[STEP 1] Takeoff & Hovering")
        tello.takeoff()
        time.sleep(3)

        # 1m 이상 호버링 1초
        current_h = tello.get_height()
        if current_h < 100:
            safe_move(tello, "up", 100 - current_h)
        print(f"  Hovering at {tello.get_height()}cm - 1sec")
        time.sleep(1.5)

        # ==========================================
        #  A장애물 통과 (단계 2: +20점)
        #  높이: 125cm, 전진: 200cm (출발~장애물)
        # ==========================================
        print("\n[STEP 2] Obstacle A")
        safe_move(tello, "up", 25)     # 125cm 고도
        time.sleep(0.5)
        safe_move(tello, "forward", 200)  # 장애물 통과
        print("  Obstacle A passed!")
        time.sleep(0.5)

        # ==========================================
        #  B장애물 통과 (단계 3: +20점)
        #  높이: 75cm (낮게), 전진: 100cm
        #  방향 전환 (ㄱ자 경로)
        # ==========================================
        print("\n[STEP 3] Obstacle B")
        safe_move(tello, "down", 50)   # 75cm 고도 (고도 변경 1회)
        time.sleep(0.5)
        tello.rotate_clockwise(90)     # ㄱ자 방향 전환
        time.sleep(1.5)
        safe_move(tello, "forward", 200)  # 장애물 통과
        print("  Obstacle B passed!")
        time.sleep(0.5)

        # ==========================================
        #  C장애물 통과 (단계 4: +20점)
        #  높이: 125cm, 전진: 100cm
        # ==========================================
        print("\n[STEP 4] Obstacle C")
        safe_move(tello, "up", 50)     # 125cm 고도 (고도 변경 2회)
        time.sleep(0.5)
        safe_move(tello, "forward", 200)  # 장애물 통과
        print("  Obstacle C passed!")
        time.sleep(0.5)

        # ==========================================
        #  도착점 탐지 및 착지 (단계 5: +30점)
        # ==========================================
        print("\n[STEP 5] Landing on cylinder")

        # 착지 전 고도 낮추기
        current_h = tello.get_height()
        if current_h > 120:
            safe_move(tello, "down", current_h - 100)
        time.sleep(1)

        # YOLO로 원기둥 탐지 및 착지
        land_on_cylinder(tello, frame_read, model)

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Emergency landing!")
        tello.land()

    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        print("\nMission complete!")


if __name__ == "__main__":
    main()
