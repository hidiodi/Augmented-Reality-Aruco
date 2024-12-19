import os
import yaml

# Verzeichnisse und Klassen definieren
base_dir = "prepared_dataset"
train_dir = os.path.join(base_dir, "images")  # Annahme: Kein separates Validierungsset
classes = ["car", "pedestrian"]  # Klassen im Datensatz

# YAML-Inhalt erstellen
kitti_yaml = {
    "train": train_dir,
    "val": train_dir,  # Validierungsset kann hier angepasst werden
    "nc": len(classes),  # Anzahl der Klassen
    "names": classes,  # Klassenamen
}

# YAML-Datei speichern
yaml_file = "kitti.yaml"
with open(yaml_file, "w") as f:
    yaml.dump(kitti_yaml, f)

print(f"{yaml_file} wurde erfolgreich generiert.")
