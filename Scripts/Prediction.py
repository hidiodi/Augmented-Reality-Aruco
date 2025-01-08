from ultralytics import YOLO
import os
import cv2
import numpy as np
from distanceCalculation import calculate_distance_to_bbox
import pandas as pd

# Absoluter Pfad zur trainierten Modell-Datei
model_path = "runs/detect/train2/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modell-Datei nicht gefunden: {model_path}")

# Trainiertes Modell laden
model = YOLO(model_path)

image__folder_path = "datasets/prepared_dataset/images/"
img_list = [datei for datei in os.listdir(image__folder_path) if datei.endswith('.png')]

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
 


def calculate_distance_from_bbox(predicted_boxes, img_height, img_width, K, camera_height)->list:
    gt = read_ground_truth(original_label_path)
    distances_bb = []
    distances = []
    for i, box in enumerate(predicted_boxes):
        newCalcDist = calculate_distance_to_bbox(img_height,box, camera_height)
        # 1. Bounding Box Center (Bottom-Center)
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = y_max  # Bottom of the box

        # 2. Inverse of Intrinsic Matrix
        K_inv = np.linalg.inv(K)

        # 3. Convert Pixel Coordinates to Camera Coordinates
        ray_direction = K_inv @ np.array([x_center, y_center, 1])
        ray_direction /= np.linalg.norm(ray_direction)
        r_x, r_y, r_z = ray_direction

        # 4. Calculate Scaling Factor (t) for Ground Plane Intersection
        t = camera_height / r_y

        # 5. Compute Intersection Point in Camera Coordinates
        intersection_camera = t * ray_direction

        # 6. Compute Horizontal Distance
        horizontal_distance = np.sqrt(intersection_camera[0]**2 + intersection_camera[2]**2)
        print(f"Horizontal Distance to the Object: {horizontal_distance:.2f} meters compared to {gt[i]['gt_distance']} meters")
        #distances_bb = horizontal_distance, (gt[i]['gt_distance'])
        distances_bb = newCalcDist, (gt[i]['gt_distance'])
        distances.append(distances_bb)
    return distances

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
for img in img_list:
    bild = img.split('.')[0]
    print(f"Verarbeite Bild: {bild}")
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
    foundElementList = []
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
            cv2.putText(image, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            foundElementList.append({"id":i, "label":label})
            predicted_boxes.append((x1, y1, x2, y2))

    # Ground-Truth-Labels lesen und zeichnen
    ground_truth_boxes = []
    ious = []
    with open(label_path, "r") as f:
        lines = f.readlines()

    iou_threshold = 0.5
    matched = []  # Initialize matched as a list to track matched ground truth boxes
    ious = []     # To store IoU values for each ground truth box
    tp, fp, fn = 0, 0, 0  # Initialize true positives, false positives, and false negatives

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
        #label = f"{class_labels[class_id]}"
        #cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # IoU berechnen
        max_iou = 0
        imageId = 0
        best_gt = None
        for i, pred_box in enumerate(predicted_boxes):
            iou = calculate_iou((x1, y1, x2, y2), pred_box)
            if iou > max_iou:
                max_iou = iou
                best_gt = (x1, y1, x2, y2)
                imageId = i

        if max_iou > iou_threshold:
            if best_gt not in matched:  # Avoid multiple matches
                tp += 1
                matched.append(best_gt)
                foundElementList[imageId]["gt"] = (x1, y1, x2, y2)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                foundElementList[imageId]["iou"] = max_iou
                foundElementList[imageId]["precision"] = precision
                foundElementList[imageId]["recall"] = recall
            else:
                fp += 1  # Multiple predictions for the same ground truth
        else:
            fp += 1  # False positive if IoU is below the threshold

        ious.append(max_iou)

    # Calculate false negatives after processing all predictions
    fn = len(ground_truth_boxes) - len(matched)

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # IoU, precision und recall werte ausgeben
    for i, iou in enumerate(ious):
        print(f"Ground Truth Box {i}: IoU = {iou:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

    #Distance Calculation
    K = read_intrinsic_matrix(calib_path)
    #print(f"Intrinsic Matrix: \n{K}")
    camera_height = 1.65  # Camera height from the ground

    # Function to compute the horizontal distance
    

    distanceList = calculate_distance_from_bbox(predicted_boxes, img_height, img_width, K, camera_height)
    #calculate_distance_from_bbox(ground_truth_boxes, img_height, img_width, K, camera_height)

    # Ausgabe-Bild speichern oder anzeigen
    output_path = "doc/Output/output_image"+bild+".png"
    output_doc_path = "doc/Output/output_image"+bild+".txt"
    

    for idx, (calculated_distance, gt_distance) in enumerate(distanceList):
            foundElementList[idx]["calcDist"] = calculated_distance
            foundElementList[idx]["gtDist"] = gt_distance
            
    # Write the results to the text file in a table format
    with open(output_doc_path, 'w') as f:
        # Write the header
        f.write(f"{'ID':<5}{'Label: Confidence':<20}{'IoU':<10}{'Precision':<10}{'Recall':<10}{'CalcDist':<15}{'GtDist':<15}\n")
        f.write("="*85 + "\n")
        
        # Write each entry in the foundElementList
        for element in foundElementList:
            f.write(f"{element['id']:<5}{element['label']:<20}{element.get('iou', 'N/A'):<10.2f}{element.get('precision', 'N/A'):<10.2f}{element.get('recall', 'N/A'):<10.2f}{element.get('calcDist', 'N/A'):<15.2f}{element.get('gtDist', 'N/A'):<15.2f}\n")
    
    cv2.imwrite(output_path, image)
    #cv2.imshow("Ergebnisse", image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


# import matplotlib.pyplot as plt

# # Filter out empty lists and None values, then extract the calculated and ground truth distances
# calculated_distances = [dist[0] for dist in distances if isinstance(dist, tuple) and len(dist) == 2]
# ground_truth_distances = [dist[1] for dist in distances if isinstance(dist, tuple) and len(dist) == 2]

# # Create a scatter plot comparing calculated distances vs ground truth distances
# plt.figure(figsize=(10, 6))
# plt.scatter(calculated_distances, ground_truth_distances, color='b', marker='o', s=100, edgecolor='k')

# # Add a line y = x for orientation
# # We need to create a line that spans the range of the data
# max_value = max(max(calculated_distances), max(ground_truth_distances))  # Get max value for scaling the line
# x_line = [0, max_value]
# y_line = x_line  # Since it's y = x

# plt.plot(x_line, y_line, color='b', linestyle='-', linewidth=2, label="y = x")  # Red dashed line for y = x

# # Set labels and title
# plt.xlabel("Distance calculated using camera information", fontsize=14)
# plt.ylabel("Distance provided in ground truth", fontsize=14)
# plt.title("Horizontal Distances to Objects", fontsize=14)

# # Add gridlines for readability
# plt.grid(True, linestyle='--', alpha=0.7)

# # Show legend for the line
# plt.legend()

# # Show the plot
# plt.show()