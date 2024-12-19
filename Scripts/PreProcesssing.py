import os
import cv2 as cv
# Pfade
base_dir = "datasets/KITTI_Selection"
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")
output_dir = "datasets/prepared_dataset"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# Mapping von Objekt-Typen zu YOLO-Klassen
object_classes = {
    "car": 0,
    "pedestrian": 1,
    # Weitere Klassen können hier hinzugefügt werden
}

def convert_label(kitti_label_path, yolo_label_path, img_width, img_height):
    with open(kitti_label_path, "r") as f:
        lines = f.readlines()
    
    yolo_labels = []
    for line in lines:
        parts = line.strip().split()
        object_type = parts[0].lower()  # Objekt-Typ
        xmin, ymin, xmax, ymax = map(float, parts[1:5])
        
        if object_type not in object_classes:
            continue
        
        # Berechnung des YOLO-Formats
        class_id = object_classes[object_type]
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    # Speichern der YOLO-Label
    with open(yolo_label_path, "w") as f:
        f.writelines(yolo_labels)

# Verarbeitung der Labels
for img_file in os.listdir(image_dir):
    if not img_file.endswith(".png"):
        continue
    
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, img_file.replace(".png", ".txt"))
    yolo_label_path = os.path.join(output_dir, "labels", img_file.replace(".png", ".txt"))
    
    # Bildgröße ermitteln
    from PIL import Image
    with Image.open(img_path) as img:
        img_width, img_height = img.size
    
    # Labels konvertieren
    convert_label(label_path, yolo_label_path, img_width, img_height)
    
    new_width, new_height = 1248, 512
    # Bild laden und skalieren
    img = cv.imread(os.path.join(image_dir, img_file))
    resized_img = cv.resize(img, (new_width, new_height))
    

    # Bild kopieren
    img_output_path = os.path.join(output_dir, "images", img_file)
    cv.imwrite(img_output_path, resized_img)

print("Dataset erfolgreich vorbereitet!")
