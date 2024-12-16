from ultralytics import YOLO

# Trainiertes Modell laden
model = YOLO("runs/detect/train/weights/best.pt")
# Bildpfad
image_path = "/Users/jonasalber/Documents/GitHub/datasets/prepared_dataset/validation/images/006037.png"

# Vorhersage auf das Bild
results = model(image_path)

# Ergebnisse ausgeben
print(results)
# Ergebnisse verarbeiten
for result in results:
    boxes = result.boxes.xyxy  # Bounding Box Koordinaten (xmin, ymin, xmax, ymax)
    scores = result.boxes.conf  # Konfidenzwerte
    classes = result.boxes.cls  # Klassen-IDs

    print("Erkannte Objekte:")
    for i in range(len(boxes)):
        print(f"Klasse: {int(classes[i])}, Konfidenz: {scores[i]:.2f}, Box: {boxes[i].tolist()}")

import cv2

# Bild laden
image = cv2.imread(image_path)

# Klassen-Labels (entsprechend Ihrer `kitti.yaml`)
class_labels = ["car", "pedestrian"]

# Bounding Boxes zeichnen
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

# Ausgabe-Bild speichern oder anzeigen
output_path = "output_image.png"
cv2.imwrite(output_path, image)
cv2.imshow("Ergebnisse", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
