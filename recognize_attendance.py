import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from scipy.spatial import distance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
embeddings_file = os.path.join(BASE_DIR, "embeddings.pkl")
CSV_FILE = os.path.join(BASE_DIR, "Attendance.csv")

# Load embeddings
with open(embeddings_file, "rb") as f:
    knownEmbeddings, knownNames = pickle.load(f)

# Setup attendance log
marked_names = set()
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("Name,Date,Time\n")
else:
    with open(CSV_FILE, "r") as f:
        next(f, None)
        for line in f:
            name = line.split(",")[0].strip()
            marked_names.add(name)

# Load face detector
proto = os.path.join(BASE_DIR, "deploy.prototxt")
model = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(proto, model)

# Load embedder
embed_model = os.path.join(BASE_DIR, "nn4.small2.v1.t7")
embedder = cv2.dnn.readNetFromTorch(embed_model)

def mark_attendance(name):
    if name not in marked_names:
        now = datetime.now()
        with open(CSV_FILE, "a") as f:
            f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
        marked_names.add(name)
        return "MARKED"
    else:
        return "ALREADY MARKED"

# Start camera
cap = cv2.VideoCapture(0)
print("📷 Camera started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Embedding
            face_blob = cv2.dnn.blobFromImage(face, 1/255.0, (96, 96),
                                              (0, 0, 0), swapRB=True, crop=True)
            embedder.setInput(face_blob)
            vec = embedder.forward().flatten()

            # Compare with known embeddings
            distances = [distance.euclidean(vec, known) for known in knownEmbeddings]
            min_idx = np.argmin(distances)
            name = knownNames[min_idx] if distances[min_idx] < 0.6 else "Unknown"

            status = mark_attendance(name) if name != "Unknown" else "Unknown"
            color = (0, 255, 0) if status == "MARKED" else (0, 0, 255)
            display_text = f"{name} - {status}" if name != "Unknown" else name

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Attendance system stopped.")
