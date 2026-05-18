"""
ui_components.py
────────────────
Reusable Streamlit UI blocks:
  - render_sidebar()      – settings + project info
  - render_gesture_legend()  – gesture reference cards
  - render_command_log()  – live activity feed
"""

from __future__ import annotations

import streamlit as st
from command_executor import get_gesture_info


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
          <span class="sidebar-icon">🖐️</span>
          <span class="sidebar-title">GestureOS</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Settings ──────────────────────────────────────────────────────────
        st.markdown("### ⚙️  Settings")

        confidence = st.slider(
            "Detection Confidence",
            min_value=0.50, max_value=0.99,
            value=0.75, step=0.05,
            help="Minimum confidence score to accept a gesture."
        )
        st.session_state["confidence_thresh"] = confidence

        cooldown = st.slider(
            "Command Cooldown (s)",
            min_value=0.5, max_value=3.0,
            value=1.5, step=0.25,
            help="Minimum seconds between consecutive commands."
        )
        st.session_state["cooldown"] = cooldown

        smooth = st.slider(
            "Smoothing Window (frames)",
            min_value=1, max_value=10,
            value=5, step=1,
            help="Majority-vote window to reduce flicker."
        )
        st.session_state["smooth_window"] = smooth

        st.markdown("---")

        # ── Stack info ────────────────────────────────────────────────────────
        st.markdown("### 🧰  Tech Stack")
        stack = [
            ("🧠", "TensorFlow 2.x",   "Neural gesture classifier"),
            ("🤲", "MediaPipe Hands",   "21-point hand landmark tracking"),
            ("📷", "OpenCV",            "Real-time video capture"),
            ("🖥️", "Streamlit",         "Interactive web UI"),
            ("⌨️", "PyAutoGUI",         "OS keyboard simulation"),
            ("🐍", "Python 3.10+",      "Core runtime"),
        ]
        for icon, name, desc in stack:
            st.markdown(
                f'<div class="stack-item"><span>{icon} <b>{name}</b></span>'
                f'<span class="stack-desc">{desc}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── About ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="about-box">
          <b>GestureOS</b> · FAANG-Level Portfolio Project<br>
          Hand gestures → OS commands via ML pipeline.<br><br>
          <a href="https://github.com/your-username/gestureos" target="_blank">⭐ GitHub</a>
          &nbsp;|&nbsp;
          <a href="https://linkedin.com/in/your-profile" target="_blank">🔗 LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)


# ─── GESTURE LEGEND ──────────────────────────────────────────────────────────────

def render_gesture_legend():
    st.markdown("### 🗺️  Gesture Reference")
    gestures = get_gesture_info()

    # 2-column grid
    cols = st.columns(2)
    for idx, g in enumerate(gestures):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="gesture-card" style="border-color:{g['color']}22; background:{g['color']}11;">
                  <div class="gc-icon">{g['icon']}</div>
                  <div class="gc-info">
                    <div class="gc-name" style="color:{g['color']}">{g['command']}</div>
                    <div class="gc-gesture">{g['gesture'].replace('_', ' ')}</div>
                    <div class="gc-shortcut">{g['shortcut']}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── COMMAND LOG ─────────────────────────────────────────────────────────────────

def render_command_log():
    st.markdown("### 📟  Activity Log")

    log = st.session_state.get("command_log", [])

    if not log:
        st.markdown(
            '<div class="log-empty">No commands fired yet. Start camera and try a gesture!</div>',
            unsafe_allow_html=True,
        )
        return

    for entry in log[:8]:
        gesture_info = next(
            (g for g in get_gesture_info() if g["gesture"] == entry["gesture"]), {}
        )
        color = gesture_info.get("color", "#aaa")
        st.markdown(
            f"""
            <div class="log-entry">
              <span class="log-time">{entry['time']}</span>
              <span class="log-icon">{entry['icon']}</span>
              <span class="log-command" style="color:{color}">{entry['command']}</span>
              <span class="log-keys">({entry['keys'][0]}+{entry['keys'][1].upper()})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("🗑  Clear Log", use_container_width=True):
        st.session_state.command_log = []
        st.rerun()
