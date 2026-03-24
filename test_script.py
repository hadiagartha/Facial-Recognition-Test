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
VECTOR_FOLDER = "staff_vectors"
DISTANCE_THRESHOLD = 0.60
RECOGNITION_FRAME_SKIP = 3

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
identified_people = {} 
counted_guests = set()
first_seen_times = {}
db_online = False
frame_count = 0 

def load_staff_database():
    global known_encodings, known_names
    if not os.path.exists(VECTOR_FOLDER): os.makedirs(VECTOR_FOLDER)
    for file in os.listdir(VECTOR_FOLDER):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0].upper()
            vectors = np.load(os.path.join(VECTOR_FOLDER, file))
            for vec in vectors:
                known_encodings.append(vec)
                known_names.append(name)
    print(f"Personnel Database Initialized: {len(set(known_names))} profiles loaded.")

def setup_database_schema(reset=False):
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
    global db_online
    try:
        conn = psycopg2.connect(**DB_CONFIG); cur = conn.cursor()
        cur.execute("INSERT INTO guest_logs (zone_name, guest_id, stay_duration, confidence) VALUES (%s, %s, %s, %s)", 
                   (ZONE_NAME, v_id, round(duration, 2), round(confidence, 2)))
        conn.commit(); cur.close(); conn.close()
        db_online = True
    except: db_online = False

# Initialize
load_staff_database()
setup_database_schema(reset=False)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    results = model.track(frame, persist=True, classes=[0], verbose=False, conf=0.5)
    annotated_frame = frame.copy()
    live_total_count = 0
    staff_on_duty = set()
    curr_time = time.time()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for i, p_id in enumerate(ids):
            x, y, w, h = boxes[i]
            live_total_count += 1
            
            # --- IMPROVED DYNAMIC ANCHOR (Larger Box) ---
            # Expanded multipliers to capture more of the head and shoulders
            vx1 = int(x - w * 0.22)
            vy1 = int(y - h * 0.48) 
            vx2 = int(x + w * 0.22)
            vy2 = int(y - h * 0.05) 

            # --- Biometric Logic ---
            if p_id not in identified_people:
                identified_people[p_id] = ("GUEST", 0)

            if frame_count % RECOGNITION_FRAME_SKIP == 0:
                if w * h > 4000:
                    cx1, cy1 = max(0, int(x-w/2)), max(0, int(y-h/2))
                    cx2, cy2 = min(frame.shape[1], int(x+w/2)), min(frame.shape[0], int(y+h/2))
                    face_crop = frame[cy1:cy2, cx1:cx2]
                    if face_crop.size > 0:
                        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        encs = face_recognition.face_encodings(rgb_crop)
                        if encs and len(known_encodings) > 0:
                            distances = face_recognition.face_distance(known_encodings, encs[0])
                            best_match_idx = np.argmin(distances)
                            current_dist = distances[best_match_idx]
                            if current_dist < DISTANCE_THRESHOLD:
                                conf_percent = max(0, (1 - (current_dist / DISTANCE_THRESHOLD)) * 100)
                                identified_people[p_id] = (known_names[best_match_idx], conf_percent)
                            else:
                                identified_people[p_id] = ("GUEST", 0)

            identity, confidence = identified_people[p_id]
            is_staff = identity not in ["GUEST"]
            if is_staff: staff_on_duty.add(identity)

            # --- Visitor Logging ---
            if identity == "GUEST" and p_id not in counted_guests:
                if p_id not in first_seen_times: first_seen_times[p_id] = curr_time
                dwell = curr_time - first_seen_times[p_id]
                if dwell >= LOG_STAY_DURATION:
                    counted_guests.add(p_id)
                    log_visitor_entry(p_id, LOG_STAY_DURATION, confs[i])

            # Visual Rendering
            color = (255, 120, 0) if is_staff else (0, 255, 0)
            label = f"{identity} | {confidence:.0f}%" if confidence > 0 else identity
            
            # Draw targeting box
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), color, 2)
            # Draw label at bottom
            cv2.putText(annotated_frame, label, (vx1, vy2 + 20), 0, 0.5, color, 1)

    # --- RIGHT-ALIGNED HUD ---
    overlay = annotated_frame.copy()
    f_w = frame.shape[1]
    box_h = 110 + (len(staff_on_duty) * 25) if staff_on_duty else 105
    cv2.rectangle(overlay, (f_w - 330, 10), (f_w - 10, box_h), (20, 20, 20), -1)
    
    cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

    bx = f_w - 315
    cv2.putText(annotated_frame, f"LIVE COUNT: {live_total_count}", (bx, 35), 0, 0.6, (0, 255, 255), 1)
    cv2.putText(annotated_frame, f"TOTAL LOGGED: {len(counted_guests)}", (bx, 65), 0, 0.6, (0, 255, 0), 1)
    y_ptr = 95
    for n in staff_on_duty:
        cv2.putText(annotated_frame, f"ON SITE: {n}", (bx, y_ptr), 0, 0.45, (255, 180, 50), 1)
        y_ptr += 25

    status_col = (0, 255, 0) if db_online else (0, 0, 255)
    cv2.circle(annotated_frame, (bx, box_h - 12), 4, status_col, -1)
    cv2.putText(annotated_frame, "DB PERSISTENCE ACTIVE", (bx + 15, box_h - 10), 0, 0.35, (200, 200, 200), 1)

    cv2.imshow("Agartha Surveillance Analytics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); cv2.destroyAllWindows()