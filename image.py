from ultralytics import YOLO

model = YOLO("models/best.pt")

# 검증 실행 → 그래프 자동 생성
model.val(data="dataset/data.yaml", project="runs/detect", name="val")