import sqlite3

def create_tables():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # ---------------- USERS TABLE ----------------
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        reg_no TEXT,
        certificate TEXT,
        verified INTEGER DEFAULT 1
    )
    ''')

    # ✅ ADD SPECIALIZATION COLUMN (SAFE ADD)
    try:
        c.execute("ALTER TABLE users ADD COLUMN specialization TEXT")
    except:
        pass  # already exists → ignore

    # ---------------- REPORTS TABLE ----------------
    c.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        dr TEXT,
        heart TEXT,
        hyper TEXT,
        report_file TEXT,
        meeting_link TEXT,
        created_at TEXT
    )
    ''')

    # ---------------- REQUESTS TABLE ----------------
    c.execute('''
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        doctor_id INTEGER,
        status TEXT
    )
    ''')

    # ---------------- CHAT TABLE ----------------
    c.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        doctor_id INTEGER,
        sender TEXT,
        message TEXT,
        time TEXT
    )
    ''')

    conn.commit()
    conn.close()