import sqlite3
from datetime import datetime

# ---------------- SAVE REPORT ----------------
def save_report(patient_name, dr, heart, hyper, file_path):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # ✅ added meeting_link column support
    c.execute('''
        INSERT INTO reports 
        (patient_name, dr, heart, hyper, report_file, meeting_link, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (patient_name, str(dr), heart, hyper, file_path, "", datetime.now()))

    conn.commit()
    conn.close()


# ---------------- GET REPORTS ----------------
def get_reports():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("SELECT * FROM reports ORDER BY id DESC")
    data = c.fetchall()

    conn.close()
    return data


# ---------------- UPDATE MEETING LINK ----------------
def update_meeting(report_id, link):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute(
        "UPDATE reports SET meeting_link=? WHERE id=?",
        (link, report_id)
    )

    conn.commit()
    conn.close()