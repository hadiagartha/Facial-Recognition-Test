import cv2
import face_recognition
import numpy as np
import os

# Configuration
SAVE_PATH = "staff_vectors"
# We'll capture 20 embeddings to cover all angles
REQUIRED_SAMPLES = 20 

if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

name = input("Enter Staff Name: ").strip().upper()
cap = cv2.VideoCapture(0)
vectors = []

print(f"--- INITIALIZING BIOMETRIC SCAN FOR {name} ---")
print("Slowly move your head: Left, Right, Up, Down, and Tilt.")

while len(vectors) < REQUIRED_SAMPLES:
    success, frame = cap.read()
    if not success: break

    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face and encode
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    if encodings:
        vectors.append(encodings[0])
        progress = int((len(vectors) / REQUIRED_SAMPLES) * 100)
        
        # Visual Feedback
        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Scanning: {progress}%", (left, top-10), 0, 0.6, (0, 255, 0), 1)

    cv2.imshow("Agartha Enrollment System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

if len(vectors) == REQUIRED_SAMPLES:
    # Save as a single numpy array file
    np.save(os.path.join(SAVE_PATH, f"{name}.npy"), np.array(vectors))
    print(f"\nSUCCESS: Personnel Vector for {name} generated and stored.")
else:
    print("\nERROR: Scan interrupted. Personnel file not saved.")