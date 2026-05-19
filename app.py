"""
╔══════════════════════════════════════════════════════════════╗
║         GESTURE COMMAND CONTROL SYSTEM                       ║
║         FAANG-Level Portfolio Project                        ║
║         Author: [Thamarai Selvan B]                                  ║
╚══════════════════════════════════════════════════════════════╝

Run: streamlit run app.py
"""
import os
import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

from gesture_engine import GestureEngine
from command_executor import CommandExecutor
from ui_components import render_sidebar, render_gesture_legend, render_command_log

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GestureOS · Command Control",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── INJECT CUSTOM CSS ──────────────────────────────────────────────────────────
# Replace line 30 in app.py with this:
css_path = "assets/styles.css"
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# ─── SESSION STATE INIT ─────────────────────────────────────────────────────────
if "command_log" not in st.session_state:
    st.session_state.command_log = []
if "engine" not in st.session_state:
    st.session_state.engine = GestureEngine()
if "executor" not in st.session_state:
    st.session_state.executor = CommandExecutor()
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = None
if "last_command_time" not in st.session_state:
    st.session_state.last_command_time = 0
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "demo_text" not in st.session_state:
    st.session_state.demo_text = (
        "Hello! I'm the GestureOS demo text editor.\n\n"
        "Select this text and try gestures:\n"
        "✂️  Scissors → CUT\n"
        "📋 Open palm → PASTE\n"
        "👌 OK sign  → COPY\n"
        "✊ Fist      → BOLD\n"
        "☝️  One finger → ITALIC\n"
        "🤙 Call sign → UNDO"
    )

# ─── LAYOUT ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">
    <span class="gesture-icon">🖐️</span>
    <span>GestureOS</span>
  </div>
  <div class="hero-sub">Hand Gesture → System Command · Powered by TensorFlow + MediaPipe</div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
render_sidebar()

# ─── MAIN COLUMNS ───────────────────────────────────────────────────────────────
col_cam, col_mid, col_editor = st.columns([1.4, 0.15, 1.4])

# ── CAMERA FEED ─────────────────────────────────────────────────────────────────
with col_cam:
    st.markdown('<div class="panel-title">📷 Live Camera Feed</div>', unsafe_allow_html=True)

    cam_placeholder   = st.empty()
    status_placeholder = st.empty()

    ctrl_c1, ctrl_c2 = st.columns(2)
    with ctrl_c1:
        start_btn = st.button("▶  Start Camera", use_container_width=True, key="start_cam")
    with ctrl_c2:
        stop_btn  = st.button("⏹  Stop Camera",  use_container_width=True, key="stop_cam")

    if start_btn:
        st.session_state.camera_active = True
    if stop_btn:
        st.session_state.camera_active = False

    render_gesture_legend()

# ── CENTER DIVIDER ───────────────────────────────────────────────────────────────
with col_mid:
    st.markdown('<div class="center-divider"><span>⚡</span></div>', unsafe_allow_html=True)

# ── EDITOR PANEL ─────────────────────────────────────────────────────────────────
with col_editor:
    st.markdown('<div class="panel-title">📝 Text Editor  <span class="badge">Demo Workspace</span></div>',
                unsafe_allow_html=True)

    demo_text = st.text_area(
        label="editor",
        value=st.session_state.demo_text,
        height=260,
        label_visibility="collapsed",
        key="editor_area",
    )
    st.session_state.demo_text = demo_text

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        if st.button("**B**  Bold",   use_container_width=True): st.session_state.executor.simulate_bold()
    with col_e2:
        if st.button("*I*  Italic",   use_container_width=True): st.session_state.executor.simulate_italic()
    with col_e3:
        if st.button("↩  Undo",       use_container_width=True): st.session_state.executor.simulate_undo()

    col_e4, col_e5, col_e6 = st.columns(3)
    with col_e4:
        if st.button("✂️  Cut",        use_container_width=True): st.session_state.executor.simulate_cut()
    with col_e5:
        if st.button("📋  Copy",       use_container_width=True): st.session_state.executor.simulate_copy()
    with col_e6:
        if st.button("📌  Paste",      use_container_width=True): st.session_state.executor.simulate_paste()

    st.markdown("---")
    render_command_log()

# ─── CAMERA LOOP ────────────────────────────────────────────────────────────────
COOLDOWN_SEC = 1.5          # prevent gesture spam

if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    engine   = st.session_state.engine
    executor = st.session_state.executor

    FRAME_LIMIT = 300        # safety limit ≈ 10 s at 30 fps
    frame_count = 0

    while st.session_state.camera_active and frame_count < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("⚠️ Camera not accessible — check permissions.")
            break

        frame = cv2.flip(frame, 1)           # mirror
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run gesture detection ──────────────────────────────────────────────
        gesture, annotated_frame, confidence = engine.detect(rgb)

        # ── Overlay HUD ───────────────────────────────────────────────────────
        annotated_frame = engine.draw_hud(annotated_frame, gesture, confidence)

        # ── Execute command if cooldown passed ────────────────────────────────
        now = time.time()
        if (
            gesture
            and gesture != "none"
            and (now - st.session_state.last_command_time) > COOLDOWN_SEC
        ):
            result = executor.execute(gesture)
            if result:
                st.session_state.last_gesture     = gesture
                st.session_state.last_command_time = now
                log_entry = {
                    "time":    time.strftime("%H:%M:%S"),
                    "gesture": gesture,
                    "command": result["command"],
                    "keys":    result["keys"],
                    "icon":    result["icon"],
                }
                st.session_state.command_log.insert(0, log_entry)
                st.session_state.command_log = st.session_state.command_log[:20]

        # ── Render frame ──────────────────────────────────────────────────────
        cam_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

        # ── Status bar ────────────────────────────────────────────────────────
        if gesture and gesture != "none":
            status_placeholder.markdown(
                f'<div class="status-bar active">Detected: <b>{gesture.upper()}</b> '
                f'— Confidence {confidence:.0%}</div>',
                unsafe_allow_html=True,
            )
        else:
            status_placeholder.markdown(
                '<div class="status-bar idle">👋 Show your hand to the camera…</div>',
                unsafe_allow_html=True,
            )

        frame_count += 1
        time.sleep(0.033)    # ≈30 fps

    cap.release()
    st.session_state.camera_active = False
    cam_placeholder.markdown(
        '<div class="cam-idle">📷 Camera stopped — press ▶ Start to resume</div>',
        unsafe_allow_html=True,
    )
else:
    cam_placeholder.markdown(
        '<div class="cam-idle">📷 Press ▶ Start Camera to begin gesture detection</div>',
        unsafe_allow_html=True,
    )
