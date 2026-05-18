"""
gesture_engine.py  (v3 - Windows-safe imports)
───────────────────────────────────────────────
Fixes:
  - No top-level `import mediapipe as mp`  -> avoids bfloat16/protobuf crash
  - No top-level `import tensorflow`       -> lazy-loaded only if model exists
  - Uses mediapipe submodule imports which bypass the tasks bootstrap chain
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

# ── MediaPipe: bypass the tasks bootstrap that triggers the TF/protobuf crash ─
# Importing submodules directly skips mediapipe/__init__.py's
# `import mediapipe.tasks.python as tasks` line which is what explodes.
try:
    from mediapipe.python.solutions import hands          as _mp_hands_mod
    from mediapipe.python.solutions import drawing_utils  as _mp_draw_mod
    from mediapipe.python.solutions import drawing_styles as _mp_styles_mod
except ImportError:
    try:
        from mediapipe.solutions import hands          as _mp_hands_mod
        from mediapipe.solutions import drawing_utils  as _mp_draw_mod
        from mediapipe.solutions import drawing_styles as _mp_styles_mod
    except ImportError as exc:
        raise ImportError(
            f"\n[GestureOS] Cannot load MediaPipe: {exc}\n\n"
            "Run these commands and restart:\n"
            "  pip uninstall mediapipe protobuf tensorflow -y\n"
            "  pip install protobuf==3.20.3\n"
            "  pip install mediapipe==0.10.9\n"
        )

# ── TensorFlow: NEVER imported at module level ────────────────────────────────
# Loaded lazily inside __init__ only when a trained model file exists.
# This prevents the bfloat16 runtime crash from mismatched TF+protobuf.

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL_PATH        = "models/gesture_model.h5"
GESTURE_CLASSES   = ["scissors", "open_palm", "ok_sign", "fist", "one_finger", "call_sign"]
CONFIDENCE_THRESH = 0.75
SMOOTH_WINDOW     = 5

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


class GestureEngine:
    """Real-time hand gesture recognition."""

    def __init__(self):
        # ── MediaPipe ─────────────────────────────────────────────────────────
        self.mp_hands  = _mp_hands_mod
        self.mp_draw   = _mp_draw_mod
        self.mp_styles = _mp_styles_mod
        self.hands     = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        # ── TF model (lazy, optional) ─────────────────────────────────────────
        self.model = None
        if os.path.exists(MODEL_PATH):
            try:
                import tensorflow as tf          # lazy import - only here
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print("[GestureEngine] TF model loaded")
            except Exception as exc:
                print(f"[GestureEngine] TF model skipped ({exc}). Using heuristics.")
        else:
            print("[GestureEngine] No model file - using heuristic classifier.")

        self._buffer: deque[str] = deque(maxlen=SMOOTH_WINDOW)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def detect(
        self, rgb_frame: np.ndarray
    ) -> Tuple[Optional[str], np.ndarray, float]:
        result_frame = rgb_frame.copy()
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            self._buffer.clear()
            return None, result_frame, 0.0

        landmarks = results.multi_hand_landmarks[0]

        self.mp_draw.draw_landmarks(
            result_frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

        features = self._extract_features(landmarks)

        if self.model is not None:
            raw_gesture, confidence = self._classify_tf(features)
        else:
            raw_gesture, confidence = self._classify_heuristic(landmarks)

        self._buffer.append(raw_gesture if confidence >= CONFIDENCE_THRESH else "none")
        gesture = self._majority_vote()
        return gesture, result_frame, confidence

    def draw_hud(
        self, frame: np.ndarray, gesture: Optional[str], confidence: float
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        label = gesture.upper().replace("_", " ") if gesture and gesture != "none" else "NO GESTURE"
        color = self._gesture_color(gesture)

        cv2.putText(frame, label, (15, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{confidence:.0%}", (w - 80, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)

        bar_w = int((w - 20) * min(confidence, 1.0))
        cv2.rectangle(frame, (10, h - 14), (w - 10, h - 6), (40, 40, 60), -1)
        cv2.rectangle(frame, (10, h - 14), (10 + bar_w, h - 6), color, -1)
        return frame

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _extract_features(self, landmarks) -> np.ndarray:
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
        )
        coords -= coords[0]
        scale = np.max(np.abs(coords)) + 1e-6
        coords /= scale
        return coords.flatten()

    def _classify_tf(self, features: np.ndarray) -> Tuple[str, float]:
        import tensorflow as tf
        inp   = features[np.newaxis, :]
        probs = self.model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(probs))
        return GESTURE_CLASSES[idx], float(probs[idx])

    def _classify_heuristic(self, landmarks) -> Tuple[str, float]:
        tips = FINGER_TIPS
        pips = FINGER_PIPS
        lm   = landmarks.landmark

        def tip_above_pip(finger_idx: int) -> bool:
            return lm[tips[finger_idx]].y < lm[pips[finger_idx]].y

        def thumb_extended() -> bool:
            return abs(lm[4].x - lm[3].x) > 0.04

        t, i, m, r, p = (
            thumb_extended(),
            tip_above_pip(1),
            tip_above_pip(2),
            tip_above_pip(3),
            tip_above_pip(4),
        )

        if i and m and not r and not p:
            spread = abs(lm[8].x - lm[12].x)
            return "scissors", min(0.5 + spread * 4, 0.99)

        if t and i and m and r and p:
            return "open_palm", 0.92

        if not i and not m and not r and not p:
            return "fist", 0.90

        if i and not m and not r and not p:
            return "one_finger", 0.88

        if t and p and not i and not m and not r:
            return "call_sign", 0.88

        dist_ok = np.sqrt((lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2)
        if dist_ok < 0.06 and m and r and p:
            return "ok_sign", min(0.5 + (0.06 - dist_ok) * 10, 0.99)

        return "none", 0.0

    def _majority_vote(self) -> Optional[str]:
        if not self._buffer:
            return None
        counts: dict[str, int] = {}
        for g in self._buffer:
            counts[g] = counts.get(g, 0) + 1
        winner = max(counts, key=counts.__getitem__)
        return winner if winner != "none" else None

    @staticmethod
    def _gesture_color(gesture: Optional[str]) -> Tuple[int, int, int]:
        palette = {
            "scissors":   (255, 80,  80),
            "open_palm":  (80,  220, 100),
            "ok_sign":    (80,  180, 255),
            "fist":       (255, 160, 50),
            "one_finger": (220, 80,  255),
            "call_sign":  (80,  240, 220),
        }
        return palette.get(gesture, (180, 180, 180))
