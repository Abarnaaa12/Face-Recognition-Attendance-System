import cv2
import os
import time

# ---------------- CONFIG ----------------
BASE_DIR = r"C:\Users\rishi\OneDrive\Desktop\Attendance System"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
NUM_IMAGES = 20           # total images to capture per person
DELAY_BETWEEN = 1.0       # seconds between captures
FACE_SIZE = (96, 96)      # size expected by embedding model

# ---------------- INPUT ----------------
name = input("Enter person name: ").strip()
person_dir = os.path.join(DATASET_DIR, name)
os.makedirs(person_dir, exist_ok=True)

# ---------------- FACE DETECTION ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam!")
    exit()

print("📷 Camera started. Follow instructions to turn your face.")

count = 0
instructions = [
    "Look straight at the camera",
    "Turn your face slightly LEFT",
    "Turn your face slightly RIGHT",
]

while count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]               # Keep in COLOR
        face_resized = cv2.resize(face, FACE_SIZE)

        # Save face image
        count += 1
        img_path = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(img_path, face_resized)

        # Draw rectangle and instructions
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, instructions[count % len(instructions)], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {count}/{NUM_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        time.sleep(DELAY_BETWEEN)

    cv2.imshow("Face Dataset Capture (COLOR)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit early
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Dataset collection completed for {name}. Images saved at {person_dir}")
