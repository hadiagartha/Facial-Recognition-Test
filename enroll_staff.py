import cv2
import face_recognition
import numpy as np
import os
import time

# --- Configuration ---
STAFF_NAME = "HADI"          
OUTPUT_DIR = "staff_vectors" 
REQUIRED_VECTORS = 15        
CAPTURE_DELAY = 0.3          # <--- Defined correctly here

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

cap = cv2.VideoCapture(0)
captured_vectors = []
last_capture_time = time.time()

print(f"--- BIOMETRIC ENROLLMENT: {STAFF_NAME} ---")
print("INSTRUCTIONS: Slowly rotate your head (Left, Right, Up, Down).")
time.sleep(2)

while len(captured_vectors) < REQUIRED_VECTORS:
    success, frame = cap.read()
    if not success: break

    display_frame = frame.copy()
    current_time = time.time()

    # --- FIXED LINE BELOW: Corrected spelling to CAPTURE_DELAY ---
    if current_time - last_capture_time > CAPTURE_DELAY:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            captured_vectors.append(face_encodings[0])
            last_capture_time = current_time
            print(f"Captured Angle {len(captured_vectors)}/{REQUIRED_VECTORS}")

    # UI Overlay
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 0, 0), 2)
        
    cv2.rectangle(display_frame, (50, 400), (590, 430), (40, 40, 40), -1)
    progress_w = int((len(captured_vectors) / REQUIRED_VECTORS) * 540)
    cv2.rectangle(display_frame, (50, 400), (50 + progress_w, 430), (0, 255, 0), -1)
    cv2.putText(display_frame, f"ENROLLING {STAFF_NAME}: {len(captured_vectors)}/{REQUIRED_VECTORS}", 
                (60, 422), 0, 0.6, (255, 255, 255), 2)

    cv2.imshow("Agartha World - Biometric Enrollment", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

if len(captured_vectors) >= REQUIRED_VECTORS:
    vector_path = os.path.join(OUTPUT_DIR, f"{STAFF_NAME.lower()}.npy")
    np.save(vector_path, np.array(captured_vectors))
    print(f"\n[SUCCESS] Profile for {STAFF_NAME} finalized.")
else:
    print("\n[CANCELLED] Enrollment incomplete.")

cap.release()
cv2.destroyAllWindows()