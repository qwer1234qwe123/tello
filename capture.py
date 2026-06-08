from djitellopy import Tello
import cv2
import os

# 저장 폴더 생성
save_dir = "dataset/images2"
os.makedirs(save_dir, exist_ok=True)

tello = Tello()
tello.connect()
print(f"배터리: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

count = 0

print("==================================")
print("  [SPACE] 사진 촬영")
print("  [Q] 종료")
print("==================================")

while True:
    frame = frame_read.frame

    display = frame.copy()
    cv2.putText(display, f"Saved: {count}장", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Tello Camera", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # 스페이스바 촬영
        filename = f"{save_dir}/img_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"촬영: {filename}")

    elif key == ord('q'):  # 종료
        
        break

tello.streamoff()
cv2.destroyAllWindows()
print(f"총 {count}장 저장 완료!")