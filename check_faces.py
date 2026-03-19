import face_recognition
import cv2
import os
import numpy as np # Need this for distance math

def test_recognition():
    print("--- AGARTHA FACE TEST STARTING ---")
    
    # 1. SETUP PATHS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    staff_folder = os.path.join(script_dir, "staff_photos")
    
    known_face_encodings = []
    known_face_names = []

    # 2. AUTOMATIC LOADER
    valid_extensions = (".jpg", ".jpeg", ".png")
    if not os.path.exists(staff_folder):
        print(f"Error: Folder {staff_folder} not found!")
        return

    for filename in os.listdir(staff_folder):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(staff_folder, filename)
            try:
                image = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    known_face_encodings.append(encs[0])
                    known_face_names.append(os.path.splitext(filename)[0].upper())
                    print(f"Successfully encoded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # 3. CAPTURE LIVE FRAME
    video_capture = cv2.VideoCapture(0)
    print("\nWebcam opening... Stay still!")
    for _ in range(15): video_capture.read() # Longer wait for auto-focus
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret: return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print(f"Found {len(face_encodings)} face(s) in the live frame.")

    # 4. HIGH-PRECISION IDENTIFICATION
    for face_encoding in face_encodings:
        # Get the 'distance' (0.0 = identical, 1.0 = total stranger)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Pick the index with the SMALLEST distance
        best_match_index = np.argmin(face_distances)
        distance = face_distances[best_match_index]
        
        # STRICTOR THRESHOLD: 0.45 is much safer than 0.6
        if distance < 0.45:
            name = known_face_names[best_match_index]
            confidence = (1 - distance) * 100
            print(f"RESULT: {name} (Confidence: {confidence:.2f}%)")
        else:
            print(f"RESULT: Unknown Person (Closest match was {known_face_names[best_match_index]} at {distance:.2f} distance)")

    cv2.imwrite("last_test_capture.jpg", frame)
    print("\nTest complete. Image saved as 'last_test_capture.jpg'")

if __name__ == "__main__":
    test_recognition()