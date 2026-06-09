import cv2
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
proto = os.path.join(BASE_DIR, "deploy.prototxt")
model = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
embed_model = os.path.join(BASE_DIR, "nn4.small2.v1.t7")
dataset_dir = os.path.join(BASE_DIR, "dataset")
embeddings_file = os.path.join(BASE_DIR, "embeddings.pkl")

# Load models
detector = cv2.dnn.readNetFromCaffe(proto, model)
embedder = cv2.dnn.readNetFromTorch(embed_model)

known_embeddings = []
known_names = []

# Loop over people
for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        # Detect face
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
        detector.setInput(blob)
        detections = detector.forward()

        if detections.shape[2] > 0 and detections[0, 0, 0, 2] > 0.6:
            box = detections[0, 0, 0, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Get embedding
            face_blob = cv2.dnn.blobFromImage(face, 1/255.0, (96, 96),
                                              (0, 0, 0), swapRB=True, crop=True)
            embedder.setInput(face_blob)
            vec = embedder.forward()

            known_embeddings.append(vec.flatten())
            known_names.append(person)

# Save embeddings
with open(embeddings_file, "wb") as f:
    pickle.dump((np.array(known_embeddings), known_names), f)

print("✅ Embeddings extracted successfully")
