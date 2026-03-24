import psycopg2
from datetime import datetime

# --- System Configuration ---
DB_CONFIG = {
    "database": "postgres", "user": "postgres", "password": "Hadi@1823", 
    "host": "127.0.0.1", "port": "5432"
}

def generate_operational_report():
    """Aggregates visitor telemetry data for daily operational analysis."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        today_date = datetime.now().strftime('%Y-%m-%d')

        # 1. Primary Aggregate Statistics
        cursor.execute("""
            SELECT COUNT(DISTINCT guest_id), AVG(stay_duration), MAX(stay_duration) 
            FROM guest_logs 
            WHERE timestamp::date = CURRENT_DATE
        """)
        total_unique, avg_stay, max_stay = cursor.fetchone()

        # 2. Peak Traffic Load Distribution
        cursor.execute("""
            SELECT TO_CHAR(timestamp, 'HH24:00') as hour, COUNT(*) as count 
            FROM guest_logs 
            WHERE timestamp::date = CURRENT_DATE
            GROUP BY hour ORDER BY count DESC LIMIT 1
        """)
        peak_data = cursor.fetchone()
        peak_hour = peak_data[0] if peak_data else "N/A"

        # 3. Time-Series Breakdown
        cursor.execute("""
            SELECT TO_CHAR(timestamp, 'HH24:00') as hour, COUNT(*) 
            FROM guest_logs WHERE timestamp::date = CURRENT_DATE
            GROUP BY hour ORDER BY hour ASC
        """)
        hourly_breakdown = cursor.fetchall()

        # --- Report Construction ---
        report = f"""
==================================================
SURVEILLANCE ANALYTICS: OPERATIONAL SUMMARY
Target Date: {today_date} | Generated: {datetime.now().strftime("%H:%M:%S")}
==================================================

[Biometric Visitor Analytics]
--------------------------------------------------
Unique Visitors Tracked:  {total_unique if total_unique else 0}
Mean Dwell Duration:      {avg_stay if avg_stay else 0.0:.2f}s
Maximum Dwell Duration:   {max_stay if max_stay else 0.0:.2f}s

[Traffic Distribution Metrics]
--------------------------------------------------
Peak Operational Hour:    {peak_hour}
"""
        for hour, count in hourly_breakdown:
            report += f"\n  - Interval {hour}: {count} Visitors"

        report += "\n\n=================================================="
        print(report)
        
        # File System Persistence
        with open(f"Operational_Report_{today_date}.txt", "w") as f:
            f.write(report)
            
    except Exception as e:
        print(f"Operational Reporting Fault: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    generate_operational_report()