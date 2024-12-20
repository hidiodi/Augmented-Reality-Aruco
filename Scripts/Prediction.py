from ultralytics import YOLO
import os
import cv2
import numpy as np

# Absoluter Pfad zur trainierten Modell-Datei
model_path = "runs/detect/train/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modell-Datei nicht gefunden: {model_path}")

# Trainiertes Modell laden
model = YOLO(model_path)
bild = "006227"
# Absoluter Bildpfad
image_path = "datasets/prepared_dataset/images/"+bild+".png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Bild-Datei nicht gefunden: {image_path}")

# Ground-Truth-Label-Pfad
label_path = "datasets/prepared_dataset/labels/"+bild+".txt"
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label-Datei nicht gefunden: {label_path}")

# Ground-Truth-Label-Pfad
original_label_path = "datasets/KITTI_Selection/labels/"+bild+".txt"
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label-Datei nicht gefunden: {label_path}")

calib_path = "datasets/KITTI_Selection/calib/"+bild+".txt"
if not os.path.exists(calib_path):
    raise FileNotFoundError(f"Label-Datei nicht gefunden: {calib_path}")

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

# Function to read the ground truth labels
def read_ground_truth(file_path):
    ground_truth = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        obj_type = parts[0]  # Object type (e.g., Car)
        x_min, y_min, x_max, y_max = map(float, parts[1:5])  # Bounding box
        gt_distance = float(parts[5])  # Ground truth distance
        ground_truth.append({
            'type': obj_type,
            'bbox': [x_min, y_min, x_max, y_max],
            'gt_distance': gt_distance
        })
    return ground_truth

ground_truth_boxes = []
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

    ground_truth_boxes.append((x1, y1, x2, y2))

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

# Intrinsic Matrix lesen
def read_intrinsic_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Parse the three lines into a 3x3 matrix
    matrix = []
    for line in lines:
        row = list(map(float, line.strip().split()))
        matrix.append(row)
    return np.array(matrix)


#Distance Calculation
K = read_intrinsic_matrix(calib_path)

# Camera height above the ground (in meters)
camera_height = 1.65

def CalculateDistance(original_label_path, predicted_boxes, read_ground_truth, K, camera_height):
    for i, box in enumerate(predicted_boxes):
        # Bounding box coordinates from YOLO (in pixels)
        x_min, y_min, x_max, y_max = box  
        print(f"Bounding Box Coordinates: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        
        # Bottom center of the bounding box
        x_center = (x_min + x_max) / 2
        y_center = y_max  # Use y_max for the bottom center
        print(f"Bounding Box Center: ({x_center}, {y_center})")

        # Step 1: Inverse of the Intrinsic Matrix
        K_inv = np.linalg.inv(K)
        print(f"Inverse Matrix:\n{K_inv}")

        # Step 2: Compute the ray direction from the camera (normalized)
        img_height = 375
        y_center_bottom_adjusted = img_height - y_center  # Invert y-coordinate to match camera coordinate system

        # Compute the ray direction using the adjusted y-coordinate
        ray_direction = K_inv @ np.array([x_center, y_center_bottom_adjusted, 1])
        print(f"Ray Direction: {ray_direction}")

        # Ray direction vector is now in camera coordinates [r_x, r_y, r_z]
        r_x, r_y, r_z = ray_direction
        print(f"Ray Direction: [{r_x:.2f}, {r_y:.2f}, {r_z:.2f}]")

        # Step 3: Solve for t where the ray intersects the ground (Y=0)
        # t is the scaling factor for the ray to reach the ground
        if r_y == 0:
            print("Warning: r_y is zero, can't compute t")
            continue  # Skip this box as it won't intersect the ground
        t = camera_height / r_y  # Assuming ground is at Y = 0
        print(f"Scaling factor t: {t}")

        # Step 4: Calculate the horizontal distance (D) using Pythagoras
        L = ray_direction * t
        horizontal_distance = np.sqrt(np.inner(L,L) - camera_height**2)  # Only consider x and z for horizontal distance

        # Get ground truth data
        gt = read_ground_truth(original_label_path)
        print(f"Horizontal Distance to the car: {horizontal_distance:.2f} meters compared to the real distance of {gt[i]['gt_distance']} meters")

# Example usage (make sure the necessary variables are defined):

CalculateDistance(original_label_path, predicted_boxes, read_ground_truth, K, camera_height)
CalculateDistance(original_label_path, ground_truth_boxes, read_ground_truth, K, camera_height)

# Ausgabe-Bild speichern oder anzeigen
output_path = "output_image.png"
cv2.imwrite(output_path, image)
cv2.imshow("Ergebnisse", image)
cv2.waitKey(0)
cv2.destroyAllWindows()