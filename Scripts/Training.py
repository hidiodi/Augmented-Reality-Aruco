from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Modell initialisieren
#model = YOLO("yolo11n.pt")  # Sie können yolov11s.pt oder ein anderes Modell verwenden

model = YOLO("runs/detect/train8/weights/best.pt")  # Pfad zu den vortrainierten Gewichten
# Training
model.train(
    data="kitti.yaml",
    epochs=100,
    batch=32,
    augment=True,
    )
model.val(data="kitti.yaml")
# Ergebnisse anzeigen
