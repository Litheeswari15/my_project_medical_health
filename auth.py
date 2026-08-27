import sqlite3

def connect():
    return sqlite3.connect("users.db", check_same_thread=False)

# ✅ ADDED specialization (default value → no break)
def signup(role, name, email, password, reg_no="", certificate="", specialization=""):
    conn = connect()
    c = conn.cursor()

    # doctor verification = 0, patient = 1
    verified = 1 if role == "patient" else 0

    try:
        # ✅ TRY WITH SPECIALIZATION
        c.execute('''
INSERT INTO users (role, name, email, password, reg_no, certificate, specialization, verified)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', (role, name, email, password, reg_no, certificate, specialization, 1))

        conn.commit()
        return True  # ✅ SUCCESS

    except sqlite3.IntegrityError:
        # ✅ EMAIL ALREADY EXISTS
        return False

    except:
        # fallback (old structure support)
        try:
            c.execute('''
INSERT INTO users (role, name, email, password, reg_no, certificate, verified)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', (role, name, email, password, reg_no, certificate, 1))

            conn.commit()
            return True
        except:
            return False

    finally:
        conn.close()


def login(email, password):
    conn = connect()
    c = conn.cursor()

    c.execute("""
    SELECT * FROM users WHERE email=? AND password=?
    """, (email, password))

    user = c.fetchone()
    conn.close()

    return user