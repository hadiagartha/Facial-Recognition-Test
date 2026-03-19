import cv2
import psycopg2 
import time
import os
import face_recognition
import numpy as np
from ultralytics import YOLO

# --- System Configuration ---
ZONE_NAME = "Main Entrance"
LOG_STAY_DURATION = 5 
STAFF_FOLDER = "staff_photos"
DISTANCE_THRESHOLD = 0.55  # Euclidean distance limit for face matching
RECOGNITION_FRAME_SKIP = 5 # Frequency of facial encoding processing (frames)

# --- Database Credentials ---
DB_PASS = "Hadi@1823" 
DB_CONFIG = {
    "database": "postgres", "user": "postgres", "password": DB_PASS,
    "host": "127.0.0.1", "port": "5432"
}

# --- Object Detection & Tracking State ---
model = YOLO('yolov8s.pt') 
cap = cv2.VideoCapture(0)
known_encodings = []
known_names = []
identified_people = {} # Cache for person ID mapping
counted_guests = set()
first_seen_times = {}
db_online = False
frame_count = 0 

def load_staff_database():
    """Initializes and encodes personnel images from the local repository."""
    if not os.path.exists(STAFF_FOLDER): os.makedirs(STAFF_FOLDER)
    for file in os.listdir(STAFF_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img = face_recognition.load_image_file(f"{STAFF_FOLDER}/{file}")
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(os.path.splitext(file)[0].upper())
    print(f"Personnel Database Initialized: {len(known_names)} records loaded.")

def setup_database_schema(reset=False):
    """Establishes connection to PostgreSQL and initializes the guest_logs table."""
    global db_online
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        if reset: cur.execute("DROP TABLE IF EXISTS guest_logs")
        cur.execute('''CREATE TABLE IF NOT EXISTS guest_logs 
                      (id SERIAL PRIMARY KEY, zone_name TEXT, guest_id INTEGER, 
                       stay_duration REAL, confidence REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit(); cur.close(); conn.close()
        db_online = True
    except Exception as e: 
        print(f"Database Connection Error: {e}")
        db_online = False

def log_visitor_entry(v_id, duration, confidence):
    """Inserts visitor telemetry data into the persistent storage backend."""
    global db_online
    try:
        conn = psycopg2.connect(**DB_CONFIG); cur = conn.cursor()
        cur.execute("INSERT INTO guest_logs (zone_name, guest_id, stay_duration, confidence) VALUES (%s, %s, %s, %s)", 
                   (ZONE_NAME, v_id, round(duration, 2), round(confidence, 2)))
        conn.commit(); cur.close(); conn.close()
        db_online = True
    except: db_online = False

# Initialize System Components
load_staff_database()
setup_database_schema(reset=False)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    # Generate tracking predictions via YOLOv8
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
            
            # --- Biometric Verification Logic ---
            if p_id not in identified_people and frame_count % RECOGNITION_FRAME_SKIP == 0:
                if w * h > 8000: # Threshold for adequate facial resolution
                    face_crop = frame[y1:y2, x1:x2]
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb_crop)
                    
                    if encs and len(known_encodings) > 0:
                        distances = face_recognition.face_distance(known_encodings, encs[0])
                        best_match_index = np.argmin(distances)
                        if distances[best_match_index] < DISTANCE_THRESHOLD:
                            identified_people[p_id] = known_names[best_match_index]
                        else:
                            identified_people[p_id] = "GUEST"

            identity = identified_people.get(p_id, "PROCESSING...")
            
            # Dynamic Labeling and Classification
            is_staff = identity not in ["GUEST", "PROCESSING..."]
            color = (255, 0, 0) if is_staff else (0, 255, 0)
            label = f"STAFF: {identity}" if is_staff else f"GUEST {p_id}"
            
            if identity == "GUEST": 
                live_guest_count += 1
                # Telemetry logging for guest dwell time
                if p_id not in counted_guests:
                    if p_id not in first_seen_times: first_seen_times[p_id] = curr_time
                    dwell_time = curr_time - first_seen_times[p_id]
                    if dwell_time >= LOG_STAY_DURATION:
                        counted_guests.add(p_id)
                        log_visitor_entry(p_id, dwell_time, confs[i])

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1-10), 0, 0.6, color, 2)

    # --- System Status HUD ---
    cv2.rectangle(annotated_frame, (5, 5), (280, 115), (40, 40, 40), -1)
    cv2.putText(annotated_frame, f"LIVE GUESTS: {live_guest_count}", (20, 35), 0, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"TOTAL LOGGED: {len(counted_guests)}", (20, 70), 0, 0.7, (0, 255, 0), 2)
    status_color = (0, 255, 0) if db_online else (0, 0, 255)
    cv2.circle(annotated_frame, (20, 100), 6, status_color, -1)
    cv2.putText(annotated_frame, "DB PERSISTENCE ACTIVE", (35, 105), 0, 0.4, (200, 200, 200), 1)

    cv2.imshow("Surveillance Analytics Dashboard", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); cv2.destroyAllWindows()