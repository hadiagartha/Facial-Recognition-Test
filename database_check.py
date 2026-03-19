import psycopg2
import os
from datetime import datetime

# --- POSTGRES CONFIG ---
DB_CONFIG = {
    "database": "postgres",
    "user": "postgres",
    "password": "Hadi@1823",
    "host": "127.0.0.1",
    "port": "5432"
}

def generate_agartha_report():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 1. THE BIG NUMBERS
        cursor.execute("SELECT COUNT(DISTINCT guest_id), AVG(stay_duration), MAX(stay_duration) FROM guest_logs")
        total_unique, avg_stay, max_stay = cursor.fetchone()

        # 2. PEAK HOUR ANALYSIS (Postgres Syntax)
        cursor.execute("""
            SELECT TO_CHAR(timestamp, 'HH24:00') as hour, COUNT(*) as count 
            FROM guest_logs 
            GROUP BY hour 
            ORDER BY count DESC 
            LIMIT 1
        """)
        peak_data = cursor.fetchone()
        peak_hour = peak_data[0] if peak_data else "N/A"
        peak_count = peak_data[1] if peak_data else 0

        # 3. HOURLY VISITOR BREAKDOWN (For the "Internship Win")
        cursor.execute("""
            SELECT TO_CHAR(timestamp, 'HH24:00') as hour, COUNT(*) 
            FROM guest_logs 
            GROUP BY hour 
            ORDER BY hour ASC
        """)
        hourly_breakdown = cursor.fetchall()

        # --- REPORT GENERATION ---
        report_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"Agartha_Postgres_Report_{report_date}.txt"
        
        report_text = f"""
==================================================
AGARTHA WORLD: ANALYTICS REPORT (POSTGRES)
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
==================================================

[OVERALL VISITOR STATS]
--------------------------------------------------
TOTAL UNIQUE GUESTS:    {total_unique if total_unique else 0}
AVERAGE STAY TIME:      {avg_stay if avg_stay else 0.0:.2f} seconds
LONGEST STAY:           {max_stay if max_stay else 0.0:.2f} seconds

[TRAFFIC ANALYSIS]
--------------------------------------------------
PEAK OPERATIONAL HOUR:  {peak_hour} ({peak_count} detections)

HOURLY FLOW:"""
        
        for hour, count in hourly_breakdown:
            report_text += f"\n  - {hour}: {count} Guests"

        report_text += "\n\n[STAFF NOTES]\n--------------------------------------------------"
        report_text += "\n* Facial Recognition active for Staff bypass."
        report_text += f"\n* Database: PostgreSQL (Port 5432) at {DB_CONFIG['host']}"
        report_text += "\n\nCONFIDENTIAL: Agartha Park Internal Use Only\n=================================================="

        # Output to console
        print(report_text)

        # Save to file
        with open(filename, "w") as f:
            f.write(report_text)
        print(f"\n--> Report successfully exported to: {filename}")

    except Exception as e:
        print(f"Postgres Reporting Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    generate_agartha_report()