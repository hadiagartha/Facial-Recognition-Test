import cv2
import psycopg2 
import time
import os
import face_recognition
import numpy as np
from ultralytics import YOLO

# --- CONFIG ---
ZONE_NAME = "Main Entrance"
LOG_STAY_DURATION = 5 
STAFF_FOLDER = "staff_photos"
DISTANCE_THRESHOLD = 0.55  # Relaxed slightly from 0.45 for better recognition
RECOGNITION_FRAME_SKIP = 10 # Only check faces every 10 frames to save CPU

# --- POSTGRES CONFIG ---
DB_PASS = "Hadi@1823" 
DB_CONFIG = {"database": "postgres", "user": "postgres", "password": DB_PASS, "host": "127.0.0.1", "port": "5432"}

model = YOLO('yolov8s.pt') 
cap = cv2.VideoCapture(0)
known_encodings = []
known_names = []
identified_people = {} 
counted_guests = set()
first_seen_times = {}
db_online = False
frame_count = 0 

def load_staff():
    if not os.path.exists(STAFF_FOLDER): os.makedirs(STAFF_FOLDER)
    for file in os.listdir(STAFF_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img = face_recognition.load_image_file(f"{STAFF_FOLDER}/{file}")
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(os.path.splitext(file)[0].upper())
    print(f"--- STAFF LOADED: {len(known_names)} Identifiers Active ---")

def get_conn(): return psycopg2.connect(**DB_CONFIG)

def setup_db(reset=False):
    global db_online
    try:
        conn = get_conn(); cur = conn.cursor()
        if reset: cur.execute("DROP TABLE IF EXISTS guest_logs")
        cur.execute('''CREATE TABLE IF NOT EXISTS guest_logs 
                      (id SERIAL PRIMARY KEY, zone_name TEXT, guest_id INTEGER, 
                       stay_duration REAL, confidence REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit(); cur.close(); conn.close()
        db_online = True
        print("--- POSTGRES CONNECTED ---")
    except Exception as e: print(f"DB Offline: {e}"); db_online = False

load_staff()
setup_db(reset=True)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    # Speed hack: Resize frame for processing, but keep original for display
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    results = model.track(frame, persist=True, classes=[0], verbose=False, conf=0.5)
    annotated_frame = frame.copy()
    live_guest_count = 0
    curr_time = time.time()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for i, p_id in enumerate(ids):
            x, y, w, h = boxes[i]
            x1, y1 = max(0, int(x-w/2)), max(0, int(y-h/2))
            x2, y2 = min(frame.shape[1], int(x+w/2)), min(frame.shape[0], int(y+h/2))
            
            # --- OPTIMIZED IDENTIFICATION ---
            # Only run face recognition if:
            # 1. We don't know who this is yet
            # 2. We are on a 'skip' frame (saves CPU)
            # 3. The person is close enough (box area > 8000)
            if p_id not in identified_people and frame_count % RECOGNITION_FRAME_SKIP == 0:
                if w * h > 8000:
                    face_crop = frame[y1:y2, x1:x2]
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb_crop)
                    
                    if encs and len(known_encodings) > 0:
                        distances = face_recognition.face_distance(known_encodings, encs[0])
                        best_match_index = np.argmin(distances)
                        dist = distances[best_match_index]
                        
                        # DEBUG: See the distance in terminal
                        print(f"ID {p_id} distance: {dist:.2f} (Target < {DISTANCE_THRESHOLD})")

                        if dist < DISTANCE_THRESHOLD:
                            identified_people[p_id] = known_names[best_match_index]
                        else:
                            identified_people[p_id] = "GUEST"

            identity = identified_people.get(p_id, "IDENTIFYING...")
            
            # Draw and Log logic (same as before)
            color = (255, 0, 0) if identity not in ["GUEST", "IDENTIFYING..."] else (0, 255, 0)
            label = f"STAFF: {identity}" if color == (255, 0, 0) else f"GUEST {p_id}"
            if identity == "GUEST": live_guest_count += 1

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1-10), 0, 0.6, color, 2)

    # Dashboard display...
    cv2.imshow("Agartha Optimized", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); cv2.destroyAllWindows()