import sqlite3
from datetime import datetime


# ---------------- GET DOCTORS ----------------
def get_doctors():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # ✅ UPDATED QUERY (added specialization)
    try:
        c.execute("SELECT id, name, reg_no, specialization FROM users WHERE role='doctor'")
    except:
        # fallback (if column not added yet)
        c.execute("SELECT id, name, reg_no FROM users WHERE role='doctor'")

    data = c.fetchall()
    conn.close()
    return data


# ---------------- SEND REQUEST ----------------
def send_request(patient_id, doctor_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO requests (patient_id, doctor_id, status)
        VALUES (?, ?, 'pending')
        """,
        (patient_id, doctor_id)
    )
    conn.commit()
    conn.close()


# ---------------- GET REQUESTS FOR DOCTOR ----------------
def get_requests(doctor_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """
        SELECT r.id, u.name
        FROM requests r
        JOIN users u ON r.patient_id = u.id
        WHERE r.doctor_id = ? AND r.status = 'pending'
        """,
        (doctor_id,)
    )
    data = c.fetchall()
    conn.close()
    return data


# ---------------- ACCEPT REQUEST ----------------
def accept_request(req_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE requests SET status = 'accepted' WHERE id = ?", (req_id,))
    conn.commit()
    conn.close()


# ---------------- GET ACCEPTED USERS ----------------
def get_connections(user_id, role):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    if role == "patient":
        c.execute(
            """
            SELECT u.id, u.name
            FROM requests r
            JOIN users u ON r.doctor_id = u.id
            WHERE r.patient_id = ? AND r.status = 'accepted'
            """,
            (user_id,)
        )
    else:
        c.execute(
            """
            SELECT u.id, u.name
            FROM requests r
            JOIN users u ON r.patient_id = u.id
            WHERE r.doctor_id = ? AND r.status = 'accepted'
            """,
            (user_id,)
        )

    data = c.fetchall()
    conn.close()
    return data


# ---------------- SEND MESSAGE ----------------
def send_message(patient_id, doctor_id, sender, msg):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO chats (patient_id, doctor_id, sender, message, time)
        VALUES (?, ?, ?, ?, ?)
        """,
        (patient_id, doctor_id, sender, msg, str(datetime.now()))
    )
    conn.commit()
    conn.close()


# ---------------- GET MESSAGES ----------------
def get_messages(patient_id, doctor_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """
        SELECT sender, message, time
        FROM chats
        WHERE patient_id = ? AND doctor_id = ?
        """,
        (patient_id, doctor_id)
    )
    data = c.fetchall()
    conn.close()
    return data