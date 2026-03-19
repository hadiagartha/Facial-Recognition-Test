import psycopg2
from datetime import datetime

# --- POSTGRES CONFIG (Must match test_script.py) ---
DB_CONFIG = {
    "database": "postgres",
    "user": "postgres",
    "password": "Hadi@1823", 
    "host": "127.0.0.1",
    "port": "5432"
}

def generate_daily_report():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # We only want data from today
        today_date = datetime.now().strftime('%Y-%m-%d')

        # 1. SUMMARY STATS FOR TODAY
        cursor.execute("""
            SELECT COUNT(DISTINCT guest_id), AVG(stay_duration), MAX(stay_duration) 
            FROM guest_logs 
            WHERE timestamp::date = CURRENT_DATE
        """)
        total_unique, avg_stay, max_stay = cursor.fetchone()

        # 2. PEAK HOUR FOR TODAY
        cursor.execute("""
            SELECT TO_CHAR(timestamp, 'HH24:00') as hour, COUNT(*) as count 
            FROM guest_logs 
            WHERE timestamp::date = CURRENT_DATE
            GROUP BY hour 
            ORDER BY count DESC 
            LIMIT 1
        """)
        peak_data = cursor.fetchone()
        peak_hour = peak_data[0] if peak_data else "N/A"
        peak_count = peak_data[1] if peak_data else 0

        # 3. HOURLY BREAKDOWN FOR TODAY
        cursor.execute("""
            SELECT TO_CHAR(timestamp, 'HH24:00') as hour, COUNT(*) 
            FROM guest_logs 
            WHERE timestamp::date = CURRENT_DATE
            GROUP BY hour 
            ORDER BY hour ASC
        """)
        hourly_breakdown = cursor.fetchall()

        # --- GENERATE OUTPUT ---
        report_text = f"""
==================================================
AGARTHA WORLD: DAILY OPERATIONAL REPORT
Date: {today_date}
Generated: {datetime.now().strftime("%H:%M:%S")}
==================================================

[TODAY'S VISITOR STATS]
--------------------------------------------------
TOTAL UNIQUE GUESTS:    {total_unique if total_unique else 0}
AVERAGE STAY TIME:      {avg_stay if avg_stay else 0.0:.2f} seconds
LONGEST STAY:           {max_stay if max_stay else 0.0:.2f} seconds

[TRAFFIC FLOW]
--------------------------------------------------
PEAK OPERATIONAL HOUR:  {peak_hour} ({peak_count} detections)
"""
        for hour, count in hourly_breakdown:
            report_text += f"\n  - {hour}: {count} Guests"

        report_text += "\n\n=================================================="
        
        print(report_text)
        
        # Save to file
        filename = f"Daily_Report_{today_date}.txt"
        with open(filename, "w") as f:
            f.write(report_text)
            
    except Exception as e:
        print(f"Reporting Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    generate_daily_report()