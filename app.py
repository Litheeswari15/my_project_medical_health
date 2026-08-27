"""
app.py — SmartVision AI
========================
Main Streamlit application.

Run: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from fpdf import FPDF
from datetime import datetime
import os

from report_db  import save_report, get_reports, update_meeting
from database   import create_tables
from auth       import signup, login
from connection import (
    get_doctors, send_request, get_requests, accept_request,
    get_connections, send_message, get_messages
)
from predict import (
    preprocess_bytes,
    predict_retinal as _predict_retinal,
    predict_skin    as _predict_skin,
)

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
create_tables()

if "user"      not in st.session_state: st.session_state.user      = None
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page"      not in st.session_state: st.session_state.page      = "login"

IMG_SIZE = 224

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

:root {
    --bg:       #080d18;
    --surf:     #0f1623;
    --surf2:    #161e2e;
    --surf3:    #1c2640;
    --accent:   #00d4ff;
    --purple:   #7c3aed;
    --green:    #10b981;
    --red:      #ef4444;
    --amber:    #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --border:   rgba(255,255,255,0.07);
    --glow:     0 0 24px rgba(0,212,255,0.12);
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0f1623 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── SIDEBAR LOGO ── */
.sv-logo {
    padding: 22px 20px 18px;
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(124,58,237,0.06));
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.sv-logo-title {
    font-family: 'Syne', sans-serif;
    font-size: 19px; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
}
.sv-logo-sub { font-size: 10px; color: var(--muted); letter-spacing: 2.5px; text-transform: uppercase; margin-top: 2px; }

/* ── USER CHIP ── */
.sv-user-chip {
    margin: 10px 14px;
    padding: 12px 14px;
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 12px;
}
.sv-user-name { font-family:'Syne',sans-serif; font-weight:700; font-size:14px; }
.sv-user-role { font-size:11px; color:var(--muted); margin-top:2px; }

/* ── BADGES ── */
.badge {
    display:inline-block; padding:2px 9px; border-radius:20px;
    font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;
}
.badge-blue   { background:rgba(0,212,255,0.12);  color:#00d4ff;  border:1px solid rgba(0,212,255,0.25); }
.badge-purple { background:rgba(124,58,237,0.12); color:#a78bfa;  border:1px solid rgba(124,58,237,0.25); }
.badge-green  { background:rgba(16,185,129,0.12); color:#10b981;  border:1px solid rgba(16,185,129,0.25); }
.badge-red    { background:rgba(239,68,68,0.12);  color:#ef4444;  border:1px solid rgba(239,68,68,0.25); }
.badge-amber  { background:rgba(245,158,11,0.12); color:#f59e0b;  border:1px solid rgba(245,158,11,0.25); }
.badge-grey   { background:rgba(100,116,139,0.12);color:#94a3b8;  border:1px solid rgba(100,116,139,0.25);}

/* ── PAGE HEADER ── */
.page-hdr {
    padding: 26px 36px 22px;
    background: linear-gradient(135deg, var(--surf), var(--surf2));
    border-bottom: 1px solid var(--border);
    position: relative; overflow: hidden;
}
.page-hdr::after {
    content:''; position:absolute; top:-60px; right:-60px;
    width:240px; height:240px;
    background: radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 70%);
    border-radius:50%; pointer-events:none;
}
.page-title { font-family:'Syne',sans-serif; font-size:26px; font-weight:800;
    background:linear-gradient(135deg,#fff 40%,#00d4ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.page-sub { font-size:13px; color:var(--muted); margin-top:3px; }

/* ── MAIN CONTENT WRAP ── */
.main-wrap { padding: 24px 32px; }

/* ── STAT CARDS ── */
.stat-row { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:22px; }
.stat-card {
    background:var(--surf); border:1px solid var(--border); border-radius:14px;
    padding:18px 20px; text-align:center;
}
.stat-num { font-family:'Syne',sans-serif; font-size:34px; font-weight:800;
    background:linear-gradient(135deg,#00d4ff,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stat-lbl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:1.5px; margin-top:2px; }

/* ── CARDS ── */
.sv-card {
    background:var(--surf); border:1px solid var(--border); border-radius:14px;
    padding:18px 20px; margin-bottom:12px; position:relative; overflow:hidden;
    transition: border-color .2s, box-shadow .2s;
}
.sv-card:hover { border-color:rgba(0,212,255,0.18); box-shadow:var(--glow); }
.sv-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,var(--accent),var(--purple)); opacity:0; transition:opacity .2s; }
.sv-card:hover::before { opacity:1; }

/* ── DOCTOR CARD ── */
.doc-card { display:flex; align-items:center; gap:14px; }
.doc-avatar { width:42px; height:42px; border-radius:10px; flex-shrink:0;
    background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(124,58,237,.15));
    display:flex; align-items:center; justify-content:center; font-size:18px; }
.doc-name { font-weight:600; font-size:14px; }
.doc-spec { font-size:12px; color:var(--muted); margin-top:1px; }

/* ── CHAT ── */
.chat-box { max-height:280px; overflow-y:auto; padding:12px;
    background:var(--bg); border-radius:10px; border:1px solid var(--border); margin-bottom:10px; }
.msg-p { text-align:right; margin:6px 0; }
.msg-d { text-align:left;  margin:6px 0; }
.bbl { display:inline-block; max-width:74%; padding:9px 13px; border-radius:14px; font-size:13px; line-height:1.5; }
.bbl-p { background:linear-gradient(135deg,rgba(0,212,255,.18),rgba(124,58,237,.18));
    border:1px solid rgba(0,212,255,.2); border-radius:14px 3px 14px 14px; }
.bbl-d { background:var(--surf2); border:1px solid var(--border); border-radius:3px 14px 14px 14px; }
.msg-time { font-size:10px; color:var(--muted); margin-top:2px; }

/* ── REPORT CARD ── */
.rep-card { background:var(--surf); border:1px solid var(--border); border-radius:13px; padding:16px 18px; margin-bottom:10px; }
.rep-name { font-family:'Syne',sans-serif; font-weight:700; font-size:15px; }
.rep-meta { font-size:11px; color:var(--muted); margin-top:2px; }
.rep-tags { display:flex; gap:7px; flex-wrap:wrap; margin-top:10px; }

/* ── MEETING CARD ── */
.meet-card {
    background:linear-gradient(135deg,rgba(0,212,255,.04),rgba(124,58,237,.04));
    border:1px solid rgba(0,212,255,.14); border-radius:13px; padding:16px 18px; margin-bottom:10px;
}
.meet-time { font-size:12px; color:var(--accent); font-weight:600; margin-top:4px; }

/* ── RESULT BOX ── */
.result-box {
    background:linear-gradient(135deg,var(--surf),var(--surf2));
    border:1px solid rgba(0,212,255,.18); border-radius:14px; padding:22px; margin:14px 0;
}
.result-title { font-family:'Syne',sans-serif; font-size:16px; font-weight:700;
    color:var(--accent); margin-bottom:14px; }
.result-row { display:flex; justify-content:space-between; align-items:center;
    padding:9px 0; border-bottom:1px solid var(--border); }
.result-row:last-child { border-bottom:none; }
.result-key { color:var(--muted); font-size:13px; }
.result-val { font-weight:600; font-size:13px; }

/* ── REJECT BOX ── */
.reject-box {
    background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.25);
    border-radius:14px; padding:20px; margin:14px 0;
}
.reject-title { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:#ef4444; margin-bottom:8px; }
.reject-body  { font-size:13px; color:var(--muted); line-height:1.6; }
.reject-hint  { font-size:12px; color:var(--muted); margin-top:10px; padding-top:10px;
    border-top:1px solid rgba(239,68,68,.15); }

/* ── SECTION LABEL ── */
.sec-lbl { font-family:'Syne',sans-serif; font-size:10px; font-weight:700;
    color:var(--muted); text-transform:uppercase; letter-spacing:2.5px; margin:18px 0 10px; }

/* ── DIVIDER ── */
.sv-hr { height:1px; background:var(--border); margin:18px 0; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background:var(--surf) !important; border-radius:10px !important;
    padding:4px !important; gap:3px !important; border:1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important;
    border-radius:7px !important; font-family:'DM Sans',sans-serif !important;
    font-size:13px !important; font-weight:500 !important; padding:7px 15px !important; border:none !important; }
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,rgba(0,212,255,.12),rgba(124,58,237,.12)) !important;
    color:var(--accent) !important; border:1px solid rgba(0,212,255,.25) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:18px !important; }

/* ── INPUTS ── */
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea,
.stDateInput>div>div>input, .stTimeInput>div>div>input {
    background:var(--surf2) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:9px !important; font-family:'DM Sans',sans-serif !important; }
.stTextInput>div>div>input:focus { border-color:rgba(0,212,255,.35) !important;
    box-shadow:0 0 0 2px rgba(0,212,255,.08) !important; }

/* ── BUTTONS ── */
.stButton>button {
    background:linear-gradient(135deg,rgba(0,212,255,.1),rgba(124,58,237,.1)) !important;
    border:1px solid rgba(0,212,255,.25) !important; color:var(--accent) !important;
    border-radius:9px !important; font-family:'DM Sans',sans-serif !important;
    font-weight:500 !important; transition:all .2s !important; }
.stButton>button:hover {
    background:linear-gradient(135deg,rgba(0,212,255,.2),rgba(124,58,237,.2)) !important;
    border-color:var(--accent) !important; transform:translateY(-1px) !important;
    box-shadow:0 4px 16px rgba(0,212,255,.18) !important; }
.stDownloadButton>button {
    background:linear-gradient(135deg,#00d4ff,#7c3aed) !important;
    border:none !important; color:#fff !important; font-weight:600 !important; border-radius:9px !important; }

/* ── ALERTS ── */
.stSuccess>div { background:rgba(16,185,129,.08) !important; border:1px solid rgba(16,185,129,.25) !important;
    border-radius:9px !important; color:#10b981 !important; }
.stError>div   { background:rgba(239,68,68,.08) !important; border:1px solid rgba(239,68,68,.25) !important;
    border-radius:9px !important; color:#ef4444 !important; }
.stWarning>div { background:rgba(245,158,11,.08) !important; border:1px solid rgba(245,158,11,.25) !important;
    border-radius:9px !important; color:#f59e0b !important; }
.stInfo>div    { background:rgba(0,212,255,.06) !important; border:1px solid rgba(0,212,255,.2) !important;
    border-radius:9px !important; color:var(--accent) !important; }

/* ── FILE UPLOADER ── */
.stFileUploader>div { background:var(--surf) !important;
    border:2px dashed rgba(0,212,255,.18) !important; border-radius:12px !important; }

/* ── AUTH ── */
.auth-wrap { max-width:440px; margin:0 auto; padding:32px; }
.auth-title { font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
    background:linear-gradient(135deg,#fff 30%,#00d4ff); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:4px; }
.auth-sub { font-size:13px; color:var(--muted); margin-bottom:24px; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,.1); border-radius:2px; }

/* ── RADIO (nav) ── */
div[data-testid="stSidebarNav"] { display:none !important; }
.stRadio>div { gap:2px !important; }
.stRadio>div>label { padding:10px 14px !important; border-radius:9px !important;
    font-size:13px !important; font-weight:500 !important; color:var(--muted) !important;
    transition:all .18s !important; border:1px solid transparent !important; }
.stRadio>div>label:hover { background:rgba(0,212,255,.06) !important; color:var(--accent) !important; }
.stRadio>div>label[data-checked="true"] {
    background:rgba(0,212,255,.08) !important; color:var(--accent) !important;
    border-color:rgba(0,212,255,.2) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADERS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_retinal_model():
    return tf.keras.models.load_model("model.keras")

@st.cache_resource
def load_skin_model():
    return tf.keras.models.load_model("skin_model.keras")

# ─────────────────────────────────────────────
# PREDICT WRAPPERS (use loaded models)
# ─────────────────────────────────────────────
def run_retinal(img_raw, img_batch):
    return _predict_retinal(img_raw, img_batch, load_retinal_model())

def run_skin(img_raw, img_batch):
    return _predict_skin(img_raw, img_batch, load_skin_model())

# ─────────────────────────────────────────────
# PDF GENERATORS
# ─────────────────────────────────────────────
def pdf_retinal(patient_name, dr_label, heart, hyper):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(0, 180, 220)
    pdf.cell(200, 12, "SMARTVISION AI", ln=True, align='C')

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(140, 160, 190)
    pdf.cell(200, 7, "RETINAL ANALYSIS REPORT", ln=True, align='C')

    pdf.ln(3)
    pdf.set_draw_color(0, 180, 220)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.set_text_color(50, 60, 80)
    pdf.cell(200, 8, f"Patient      : {patient_name}", ln=True)
    pdf.cell(200, 8, f"Date         : {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True)
    pdf.cell(200, 8, f"Analysis     : Retinal / Diabetic Screening", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 160, 200)
    pdf.cell(200, 9, "DIAGNOSIS RESULTS", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.set_text_color(40, 50, 70)
    pdf.cell(200, 8, f"  Diabetic Retinopathy Stage : {dr_label}", ln=True)
    pdf.cell(200, 8, f"  Heart Disease Risk         : {heart}", ln=True)
    pdf.cell(200, 8, f"  Hypertension Risk          : {hyper}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(190, 70, 70)
    pdf.cell(200, 8, "DISCLAIMER", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 110, 130)
    pdf.multi_cell(0, 7,
        "This report is AI-generated for screening purposes only. "
        "It is NOT a substitute for professional medical advice. "
        "Please consult a qualified ophthalmologist or physician.")

    fname = f"Retinal_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(fname)
    return fname


def pdf_skin(patient_name, diagnosis, confidence):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(0, 180, 220)
    pdf.cell(200, 12, "SMARTVISION AI", ln=True, align='C')

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(140, 160, 190)
    pdf.cell(200, 7, "SKIN LESION ANALYSIS REPORT", ln=True, align='C')

    pdf.ln(3)
    pdf.set_draw_color(0, 180, 220)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.set_text_color(50, 60, 80)
    pdf.cell(200, 8, f"Patient      : {patient_name}", ln=True)
    pdf.cell(200, 8, f"Date         : {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True)
    pdf.cell(200, 8, f"Analysis     : Skin Lesion / Dermatology", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 160, 200)
    pdf.cell(200, 9, "DIAGNOSIS RESULTS", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.set_text_color(40, 50, 70)
    pdf.cell(200, 8, f"  Detected Condition : {diagnosis}", ln=True)
    pdf.cell(200, 8, f"  Confidence Score   : {confidence:.1f}%", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(190, 70, 70)
    pdf.cell(200, 8, "DISCLAIMER", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 110, 130)
    pdf.multi_cell(0, 7,
        "This report is AI-generated for screening purposes only. "
        "It is NOT a substitute for professional medical advice. "
        "Please consult a qualified dermatologist for diagnosis and treatment.")

    fname = f"Skin_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(fname)
    return fname


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sv-logo">
            <div class="sv-logo-title">⚕ SmartVision AI</div>
            <div class="sv-logo-sub">Medical Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.logged_in:
            choice = st.radio("", ["🔐  Login", "📝  Sign Up"], label_visibility="collapsed")
            return "login" if "Login" in choice else "signup"

        user = st.session_state.user
        role = user[1]
        role_badge = "badge-blue" if role == "patient" else "badge-purple"

        st.markdown(f"""
        <div class="sv-user-chip">
            <div class="sv-user-name">{user[2]}</div>
            <div class="sv-user-role"><span class="badge {role_badge}">{role}</span></div>
        </div>
        """, unsafe_allow_html=True)

        if role == "patient":
            opts = [
                "🏠  Dashboard",
                "📤  Upload & Analyse",
                "👨‍⚕️  Doctors",
                "💬  Chat",
                "📅  Meetings",
                "📋  Report History",
            ]
        else:
            opts = [
                "🏠  Dashboard",
                "📥  Patient Requests",
                "📊  Patient Reports",
                "📅  Meetings",
                "💬  Chat",
            ]

        nav = st.radio("", opts, label_visibility="collapsed")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button("🚪  Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user      = None
            st.rerun()

        return nav


# ─────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────
def page_login():
    st.markdown("""
    <div style="padding:48px 0 16px; text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:38px;font-weight:800;
            background:linear-gradient(135deg,#fff 30%,#00d4ff);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">SmartVision AI</div>
        <div style="font-size:13px;color:var(--muted);margin-top:6px;">Medical Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown('<div class="sv-card" style="padding:28px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:18px;font-weight:700;margin-bottom:18px;">Welcome back 👋</div>', unsafe_allow_html=True)

        email    = st.text_input("📧 Email", placeholder="you@example.com")
        password = st.text_input("🔒 Password", type="password", placeholder="••••••••")

        if st.button("Login →", use_container_width=True):
            user = login(email, password)
            if user:
                if user[1] == "doctor" and user[7] == 0:
                    st.error("Doctor account pending verification.")
                else:
                    st.session_state.user      = user
                    st.session_state.logged_in = True
                    st.success("✅ Login successful!")
                    st.rerun()
            else:
                st.error("❌ Invalid email or password.")

        st.markdown('</div>', unsafe_allow_html=True)


def page_signup():
    st.markdown("""
    <div style="padding:36px 0 16px; text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
            background:linear-gradient(135deg,#fff 30%,#00d4ff);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Create Account</div>
        <div style="font-size:13px;color:var(--muted);margin-top:4px;">Join SmartVision AI</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="sv-card" style="padding:28px;">', unsafe_allow_html=True)

        role = st.selectbox("👤 Role", ["patient", "doctor"],
                            format_func=lambda x: "🧑 Patient" if x=="patient" else "👨‍⚕️ Doctor")
        name     = st.text_input("Full Name",  placeholder="John Doe")
        email    = st.text_input("📧 Email",   placeholder="you@example.com")
        password = st.text_input("🔒 Password", type="password", placeholder="Min. 8 characters")

        spec  = ""
        reg   = ""
        if role == "doctor":
            spec = st.text_input("🏥 Specialisation", placeholder="e.g. Dermatology, Ophthalmology")
            reg  = st.text_input("🪪 Medical Reg. No.", placeholder="MCI / NMC Number")

        if st.button("Create Account →", use_container_width=True):
            if not name or not email or not password:
                st.error("Please fill all required fields.")
            elif role == "doctor" and not reg.strip():
                st.error("Registration number is required for doctors.")
            else:
                ok = signup(role, name, email, password, reg, "", spec)
                if ok:
                    st.success("✅ Account created! Please log in.")
                else:
                    st.error("❌ Email already registered.")

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PATIENT PAGES
# ─────────────────────────────────────────────
def page_patient_dashboard():
    user    = st.session_state.user
    reports = get_reports()
    doctors = get_doctors()
    conns   = get_connections(user[0], "patient")

    st.markdown(f"""
    <div class="page-hdr">
        <div class="page-title">Hello, {user[2]} 👋</div>
        <div class="page-sub">Your health overview</div>
    </div>
    <div class="main-wrap">
    <div class="stat-row">
        <div class="stat-card"><div class="stat-num">{len(reports)}</div><div class="stat-lbl">Total Reports</div></div>
        <div class="stat-card"><div class="stat-num">{len(doctors)}</div><div class="stat-lbl">Doctors Available</div></div>
        <div class="stat-card"><div class="stat-num">{len(conns)}</div><div class="stat-lbl">My Doctors</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.info("📤 Use **Upload & Analyse** in the sidebar to generate a new medical report.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_upload():
    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Upload & Analyse</div>
        <div class="page-sub">AI-powered medical image analysis with automatic validation</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    analysis = st.selectbox(
        "🔬 Analysis Type",
        ["Retinal Analysis (Diabetic Screening)", "Skin Lesion Analysis (Dermatology)"]
    )

    st.markdown('<div class="sv-hr"></div>', unsafe_allow_html=True)

    # Guidance for user
    if "Retinal" in analysis:
        st.info("📷 Please upload a **fundus / retinal photograph** — a circular eye image with orange/red tones on a dark background. (APTOS / Kaggle DR format)")
    else:
        st.info("📷 Please upload a **dermoscopy or skin lesion photograph** — a close-up image of a skin lesion with warm skin-tone colours. (HAM10000 / ISIC format)")

    file = st.file_uploader("📁 Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if file:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(file, caption="Uploaded Image", use_column_width=True)

        with c2:
            patient_name = st.text_input("🧑 Patient Name", placeholder="Full name")

            if st.button("🔍 Analyse Image", use_container_width=True):
                if not patient_name.strip():
                    st.warning("Please enter the patient name.")
                else:
                    with st.spinner("Validating and analysing..."):
                        file.seek(0)
                        img_raw, img_batch = preprocess_bytes(file.read())

                        if img_raw is None:
                            st.error("❌ Could not decode the image. Please upload a valid JPG or PNG.")

                        elif "Retinal" in analysis:
                            result = run_retinal(img_raw, img_batch)

                            if not result["valid"]:
                                st.markdown(f"""
                                <div class="reject-box">
                                    <div class="reject-title">⚠️ Image Rejected — Not a Retinal Image</div>
                                    <div class="reject-body">{result['reason']}</div>
                                    <div class="reject-hint">
                                        <b>Expected:</b> A fundus/retinal photograph (circular bright image on dark background, orange/red tones).<br>
                                        <b>Examples of invalid uploads:</b> skin lesion photos, X-rays, CT scans, regular selfies, food photos.
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            else:
                                dr     = result["dr_stage"]
                                drl    = result["dr_label"]
                                heart  = result["heart"]
                                hyper  = result["hyper"]
                                dr_col = ["badge-green","badge-amber","badge-amber","badge-red","badge-red"]

                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="result-title">🔬 Retinal Analysis Results</div>
                                    <div class="result-row">
                                        <span class="result-key">Image Validation</span>
                                        <span class="badge badge-green">✓ Valid Retinal Image</span>
                                    </div>
                                    <div class="result-row">
                                        <span class="result-key">Diabetic Retinopathy</span>
                                        <span class="badge {dr_col[dr]}">{drl}</span>
                                    </div>
                                    <div class="result-row">
                                        <span class="result-key">Heart Disease Risk</span>
                                        <span class="badge {'badge-red' if heart=='YES' else 'badge-green'}">{heart}</span>
                                    </div>
                                    <div class="result-row">
                                        <span class="result-key">Hypertension Risk</span>
                                        <span class="badge {'badge-red' if hyper=='YES' else 'badge-green'}">{hyper}</span>
                                    </div>
                                    <div class="result-row">
                                        <span class="result-key">Model Confidence</span>
                                        <span class="result-val">{result['confidence']:.1%}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                pdf_path = pdf_retinal(patient_name, drl, heart, hyper)
                                save_report(patient_name, f"DR:{drl}", heart, hyper, pdf_path)
                                with open(pdf_path, "rb") as f:
                                    st.download_button("📥 Download PDF Report", f, pdf_path,
                                                       mime="application/pdf", use_container_width=True)

                        else:  # Skin
                            result = run_skin(img_raw, img_batch)

                            if not result["valid"]:
                                st.markdown(f"""
                                <div class="reject-box">
                                    <div class="reject-title">⚠️ Image Rejected — Not a Skin Image</div>
                                    <div class="reject-body">{result['reason']}</div>
                                    <div class="reject-hint">
                                        <b>Expected:</b> A dermoscopy or skin lesion photograph (close-up, warm skin tones, no dark border).<br>
                                        <b>Examples of invalid uploads:</b> retinal fundus photos, X-rays, CT scans, regular photos.
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            else:
                                diag    = result["diagnosis"]
                                conf    = result["confidence"] * 100
                                cc      = "badge-green" if conf>80 else "badge-amber" if conf>60 else "badge-red"

                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="result-title">🔬 Skin Lesion Analysis Results</div>
                                    <div class="result-row">
                                        <span class="result-key">Image Validation</span>
                                        <span class="badge badge-green">✓ Valid Skin Image</span>
                                    </div>
                                    <div class="result-row">
                                        <span class="result-key">Detected Condition</span>
                                        <span class="result-val">{diag}</span>
                                    </div>
                                    <div class="result-row">
                                        <span class="result-key">Confidence Score</span>
                                        <span class="badge {cc}">{conf:.1f}%</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                pdf_path = pdf_skin(patient_name, diag, conf)
                                save_report(patient_name, f"Skin:{diag}", f"{conf:.1f}%", "N/A", pdf_path)
                                with open(pdf_path, "rb") as f:
                                    st.download_button("📥 Download PDF Report", f, pdf_path,
                                                       mime="application/pdf", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_patient_doctors():
    user = st.session_state.user

    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Find Doctors</div>
        <div class="page-sub">Connect with specialists on the platform</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    doctors = get_doctors()

    if not doctors:
        st.info("No doctors registered yet.")
    else:
        st.markdown('<div class="sec-lbl">Available Specialists</div>', unsafe_allow_html=True)
        for d in doctors:
            doc_id   = d[0]
            doc_name = d[1]
            doc_reg  = d[2]
            doc_spec = (d[3] if len(d) > 3 and d[3] else "General Practice")

            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"""
                <div class="sv-card">
                    <div class="doc-card">
                        <div class="doc-avatar">👨‍⚕️</div>
                        <div>
                            <div class="doc-name">Dr. {doc_name}</div>
                            <div class="doc-spec">🏥 {doc_spec} &nbsp;·&nbsp; 🪪 {doc_reg}</div>
                        </div>
                        <span class="badge badge-blue" style="margin-left:auto;">Verified</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if st.button("Send Request", key=f"req_{doc_id}"):
                    send_request(user[0], doc_id)
                    st.success(f"Request sent to Dr. {doc_name}!")

    st.markdown("</div>", unsafe_allow_html=True)


def page_patient_chat():
    user = st.session_state.user

    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Chat with Doctors</div>
        <div class="page-sub">Secure messaging with accepted doctors</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    conns = get_connections(user[0], "patient")

    if not conns:
        st.info("No connected doctors yet. Send a request from **Doctors** tab first.")
    else:
        for c in conns:
            doc_id, doc_name = c[0], c[1]
            st.markdown(f"""
            <div class="sv-card">
                <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:12px;">
                    💬 Dr. {doc_name}
                </div>
            """, unsafe_allow_html=True)

            msgs = get_messages(user[0], doc_id)
            chat_html = '<div class="chat-box">'
            if msgs:
                for m in msgs:
                    sender, text, ts = m[0], m[1], m[2]
                    is_me  = (sender == "patient")
                    cls    = "msg-p" if is_me else "msg-d"
                    bbl    = "bbl bbl-p" if is_me else "bbl bbl-d"
                    t_str  = str(ts)[-8:-3] if ts else ""
                    chat_html += f'<div class="{cls}"><div class="{bbl}">{text}</div><div class="msg-time">{t_str}</div></div>'
            else:
                chat_html += '<div style="color:var(--muted);font-size:13px;text-align:center;padding:20px 0;">No messages yet. Say hello!</div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            mc1, mc2 = st.columns([4, 1])
            with mc1:
                msg = st.text_input("", key=f"pm_{doc_id}", label_visibility="collapsed",
                                    placeholder=f"Message Dr. {doc_name}...")
            with mc2:
                if st.button("Send ➤", key=f"ps_{doc_id}"):
                    if msg.strip():
                        send_message(user[0], doc_id, "patient", msg)
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_patient_meetings():
    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Meetings</div>
        <div class="page-sub">Consultations scheduled by your doctors</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    reports  = get_reports()
    meetings = [r for r in reports if r[6] and r[6].strip()]

    if not meetings:
        st.info("📅 No meetings scheduled yet. Your doctor will add a meeting link here.")
    else:
        st.markdown('<div class="sec-lbl">Scheduled Consultations</div>', unsafe_allow_html=True)
        for r in meetings:
            raw_link = r[6]
            created  = str(r[7])[:16] if len(r) > 7 and r[7] else ""
            # Parse link | datetime format
            if " | " in raw_link:
                link, meet_dt = raw_link.split(" | ", 1)
            else:
                link, meet_dt = raw_link, ""

            st.markdown(f"""
            <div class="meet-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:15px;">👤 {r[1]}</div>
                        <div class="meet-time">📅 {meet_dt if meet_dt else created}</div>
                    </div>
                    <a href="{link}" target="_blank" style="text-decoration:none;">
                        <div style="background:linear-gradient(135deg,#00d4ff,#7c3aed);color:#fff;
                            padding:9px 18px;border-radius:9px;font-weight:600;font-size:13px;">
                            Join Meeting →
                        </div>
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_patient_history():
    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Report History</div>
        <div class="page-sub">All your past AI-generated medical reports</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    reports = get_reports()

    if not reports:
        st.info("No reports yet. Upload an image to generate your first report.")
    else:
        st.markdown('<div class="sec-lbl">All Reports</div>', unsafe_allow_html=True)
        for r in reports:
            rid, name, dr, heart, hyper, fpath = r[0], r[1], r[2], r[3], r[4], r[5]
            created = str(r[7])[:16] if len(r)>7 and r[7] else "N/A"

            heart_b = "badge-red" if heart == "YES" else "badge-green"
            hyper_b = "badge-red" if hyper == "YES" else "badge-green"

            st.markdown(f"""
            <div class="rep-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="rep-name">📋 {name}</div>
                        <div class="rep-meta">📅 {created}</div>
                    </div>
                    <span class="badge badge-grey">#{rid}</span>
                </div>
                <div class="rep-tags">
                    <span class="badge badge-blue">{dr}</span>
                    <span class="badge {heart_b}">Heart: {heart}</span>
                    <span class="badge {hyper_b}">Hyper: {hyper}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if fpath and os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    st.download_button(f"📥 Download Report #{rid}", f, fpath,
                                       mime="application/pdf", key=f"hist_{rid}")

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DOCTOR PAGES
# ─────────────────────────────────────────────
def page_doctor_dashboard():
    user    = st.session_state.user
    reports = get_reports()
    reqs    = get_requests(user[0])
    conns   = get_connections(user[0], "doctor")

    st.markdown(f"""
    <div class="page-hdr">
        <div class="page-title">Dr. {user[2]} 🩺</div>
        <div class="page-sub">Patient management overview</div>
    </div>
    <div class="main-wrap">
    <div class="stat-row">
        <div class="stat-card"><div class="stat-num">{len(reqs)}</div><div class="stat-lbl">Pending Requests</div></div>
        <div class="stat-card"><div class="stat-num">{len(conns)}</div><div class="stat-lbl">Active Patients</div></div>
        <div class="stat-card"><div class="stat-num">{len(reports)}</div><div class="stat-lbl">Total Reports</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Use the tabs in the sidebar to manage patients, review reports, and schedule meetings.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_doctor_requests():
    user = st.session_state.user

    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Patient Requests</div>
        <div class="page-sub">Accept incoming consultation requests</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    reqs = get_requests(user[0])

    if not reqs:
        st.info("No pending requests.")
    else:
        st.markdown(f'<div class="sec-lbl">{len(reqs)} Pending</div>', unsafe_allow_html=True)
        for r in reqs:
            rid, pname = r[0], r[1]
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"""
                <div class="sv-card">
                    <div class="doc-card">
                        <div class="doc-avatar">🧑</div>
                        <div>
                            <div class="doc-name">{pname}</div>
                            <div class="doc-spec">Requesting consultation</div>
                        </div>
                        <span class="badge badge-amber" style="margin-left:auto;">Pending</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if st.button("✅ Accept", key=f"acc_{rid}"):
                    accept_request(rid)
                    st.success(f"Accepted {pname}")
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def page_doctor_reports():
    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Patient Reports</div>
        <div class="page-sub">Review AI-generated analysis reports</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    reports = get_reports()

    if not reports:
        st.info("No reports available yet.")
    else:
        st.markdown('<div class="sec-lbl">All Reports</div>', unsafe_allow_html=True)
        for r in reports:
            rid, name, dr, heart, hyper, fpath = r[0], r[1], r[2], r[3], r[4], r[5]
            created = str(r[7])[:16] if len(r)>7 and r[7] else "N/A"
            heart_b = "badge-red" if heart=="YES" else "badge-green"
            hyper_b = "badge-red" if hyper=="YES" else "badge-green"

            st.markdown(f"""
            <div class="rep-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="rep-name">👤 {name}</div>
                        <div class="rep-meta">Generated: {created}</div>
                    </div>
                    <span class="badge badge-blue">#{rid}</span>
                </div>
                <div class="rep-tags">
                    <span class="badge badge-purple">{dr}</span>
                    <span class="badge {heart_b}">Heart: {heart}</span>
                    <span class="badge {hyper_b}">Hyper: {hyper}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if fpath and os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    st.download_button(f"📥 Download Report #{rid}", f, fpath,
                                       mime="application/pdf", key=f"drep_{rid}")

    st.markdown("</div>", unsafe_allow_html=True)


def page_doctor_meetings():
    user = st.session_state.user

    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Schedule Meetings</div>
        <div class="page-sub">Add consultation links and times for your patients</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    reports = get_reports()

    if not reports:
        st.info("No patient reports yet.")
    else:
        st.markdown('<div class="sec-lbl">Patient Consultations</div>', unsafe_allow_html=True)
        for r in reports:
            rid, name = r[0], r[1]
            raw       = r[6] if r[6] else ""
            cur_link  = raw.split(" | ")[0] if " | " in raw else raw
            cur_dt    = raw.split(" | ")[1] if " | " in raw else ""

            st.markdown(f"""
            <div class="sv-card">
                <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:12px;">👤 {name}</div>
            """, unsafe_allow_html=True)

            mc1, mc2 = st.columns([2, 1])
            with mc1:
                new_link = st.text_input(
                    "Meeting Link",
                    value=cur_link,
                    key=f"ml_{rid}",
                    placeholder="https://meet.google.com/xxx-xxxx-xxx"
                )
            with mc2:
                new_dt = st.text_input(
                    "Date & Time",
                    value=cur_dt,
                    key=f"mdt_{rid}",
                    placeholder="e.g. 2025-12-10 14:30"
                )

            if st.button("💾 Save Meeting", key=f"sm_{rid}"):
                combined = f"{new_link} | {new_dt}" if new_dt.strip() else new_link
                update_meeting(rid, combined)
                st.success("✅ Meeting saved!")
                st.rerun()

            if cur_link:
                st.markdown(f"""
                <div style="margin-top:6px;padding:9px 12px;background:rgba(0,212,255,.04);
                    border-radius:8px;border:1px solid rgba(0,212,255,.1);font-size:12px;">
                    Current: <a href="{cur_link}" target="_blank" style="color:var(--accent);">{cur_link}</a>
                    {f'&nbsp;·&nbsp;<span style="color:var(--muted);">📅 {cur_dt}</span>' if cur_dt else ''}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_doctor_chat():
    user = st.session_state.user

    st.markdown("""
    <div class="page-hdr">
        <div class="page-title">Patient Chat</div>
        <div class="page-sub">Communicate with your accepted patients</div>
    </div>
    <div class="main-wrap">
    """, unsafe_allow_html=True)

    conns = get_connections(user[0], "doctor")

    if not conns:
        st.info("No connected patients. Accept requests from **Patient Requests** first.")
    else:
        for c in conns:
            pat_id, pat_name = c[0], c[1]
            st.markdown(f"""
            <div class="sv-card">
                <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:12px;">
                    💬 {pat_name}
                </div>
            """, unsafe_allow_html=True)

            msgs = get_messages(pat_id, user[0])
            chat_html = '<div class="chat-box">'
            if msgs:
                for m in msgs:
                    sender, text, ts = m[0], m[1], m[2]
                    is_me  = (sender == "doctor")
                    cls    = "msg-d" if is_me else "msg-p"
                    bbl    = "bbl bbl-d" if is_me else "bbl bbl-p"
                    t_str  = str(ts)[-8:-3] if ts else ""
                    chat_html += f'<div class="{cls}"><div class="{bbl}">{text}</div><div class="msg-time">{t_str}</div></div>'
            else:
                chat_html += '<div style="color:var(--muted);font-size:13px;text-align:center;padding:20px 0;">No messages yet.</div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            dc1, dc2 = st.columns([4, 1])
            with dc1:
                msg = st.text_input("", key=f"dm_{pat_id}", label_visibility="collapsed",
                                    placeholder=f"Message {pat_name}...")
            with dc2:
                if st.button("Send ➤", key=f"ds_{pat_id}"):
                    if msg.strip():
                        send_message(pat_id, user[0], "doctor", msg)
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────
nav = sidebar()

if not st.session_state.logged_in:
    if nav == "login":
        page_login()
    else:
        page_signup()
else:
    role = st.session_state.user[1]

    if role == "patient":
        if   "Dashboard"   in nav: page_patient_dashboard()
        elif "Upload"      in nav: page_upload()
        elif "Doctors"     in nav: page_patient_doctors()
        elif "Chat"        in nav: page_patient_chat()
        elif "Meetings"    in nav: page_patient_meetings()
        elif "History"     in nav: page_patient_history()
        else:                      page_patient_dashboard()
    else:
        if   "Dashboard"   in nav: page_doctor_dashboard()
        elif "Requests"    in nav: page_doctor_requests()
        elif "Reports"     in nav: page_doctor_reports()
        elif "Meetings"    in nav: page_doctor_meetings()
        elif "Chat"        in nav: page_doctor_chat()
        else:                      page_doctor_dashboard()