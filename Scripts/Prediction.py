from ultralytics import YOLO
import os
import cv2

# Absoluter Pfad zur trainierten Modell-Datei
model_path = "C:/Users/timSc/Documents/GitHub/Augmented-Reality-Aruco/runs/detect/train/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modell-Datei nicht gefunden: {model_path}")

# Trainiertes Modell laden
model = YOLO(model_path)

# Absoluter Bildpfad
image_path = "C:/Users/timSc/Documents/GitHub/Augmented-Reality-Aruco/datasets/prepared_dataset/images/006227.png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Bild-Datei nicht gefunden: {image_path}")

# Ground-Truth-Label-Pfad
label_path = "C:/Users/timSc/Documents/GitHub/Augmented-Reality-Aruco/datasets/prepared_dataset/labels/006227.txt"
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label-Datei nicht gefunden: {label_path}")

# Vorhersage auf das Bild
results = model.predict(image_path)

# Bild laden
image = cv2.imread(image_path)

# Klassen-Labels (entsprechend Ihrer `kitti.yaml`)
class_labels = ["car", "pedestrian"]

# Funktion zur Berechnung der IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# Ergebnisse verarbeiten und Bounding Boxes zeichnen
predicted_boxes = []
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding Box Koordinaten
    scores = result.boxes.conf.cpu().numpy()  # Konfidenzen
    classes = result.boxes.cls.cpu().numpy()  # Klassen

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = scores[i]
        cls = int(classes[i])
        
        # Bounding Box zeichnen
        color = (0, 255, 0)  # Grün
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Text für Klasse und Konfidenz
        label = f"{class_labels[cls]}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        predicted_boxes.append((x1, y1, x2, y2))

# Ground-Truth-Labels lesen und zeichnen
ious = []
with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center, y_center, width, height = map(float, parts[1:5])
    
    img_height, img_width = image.shape[:2]
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    # Ground-Truth-Bounding Box zeichnen
    color = (0, 0, 255)  # Rot
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Text für Klasse
    label = f"{class_labels[class_id]}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # IoU berechnen
    max_iou = 0
    for pred_box in predicted_boxes:
        iou = calculate_iou((x1, y1, x2, y2), pred_box)
        if iou > max_iou:
            max_iou = iou
    ious.append(max_iou)

# IoU-Werte ausgeben
for i, iou in enumerate(ious):
    print(f"Ground Truth Box {i}: IoU = {iou:.2f}")

# Ausgabe-Bild speichern oder anzeigen
output_path = "output_image.png"
cv2.imwrite(output_path, image)
cv2.imshow("Ergebnisse", image)
cv2.waitKey(0)
cv2.destroyAllWindows()