from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Modell initialisieren
model = YOLO("yolo11n.pt")  # Sie können yolov8s.pt oder ein anderes Modell verwenden

# Training
model.train(
    data="kitti.yaml",
    epochs=100,
    imgsz=(1248, 512),
    batch=64,
)
model.val(data="kitti.yaml")
# Ergebnisse anzeigen
