from ultralytics import YOLO

# Modell initialisieren
model = YOLO("yolo11n.pt")  # Sie können yolov8s.pt oder ein anderes Modell verwenden

# Training
model.train(data="kitti.yaml", epochs=10, imgsz=640, batch=16)

model.val(data="kitti.yaml")
# Ergebnisse anzeigen
