import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

faces = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    print(f"📂 Loading {person}")
    label_map[label_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (200, 200))
        faces.append(img)
        labels.append(label_id)

    label_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save(os.path.join(BASE_DIR, "trainer.yml"))

with open(os.path.join(BASE_DIR, "labels.txt"), "w") as f:
    for k, v in label_map.items():
        f.write(f"{k}:{v}\n")

print("✅ Model trained successfully")
