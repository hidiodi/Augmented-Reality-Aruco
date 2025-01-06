\begin{lstlisting}[language=Python, caption=Generating predictions using YOLOv11, label=lst:prediction]
from ultralytics import YOLO
import cv2

# Load trained model
model_path = "runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

# Predict on a sample image
image_path = "datasets/prepared_dataset/images/006059.png"
results = model.predict(image_path)

# Process results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    scores = result.boxes.conf.cpu().numpy()  # Confidence scores
    classes = result.boxes.cls.cpu().numpy()  # Class labels
\end{lstlisting}
