"""
collect_data.py
───────────────
Interactive data collection tool to build the gesture training dataset.

Usage
─────
  python collect_data.py

Controls
────────
  SPACE  → capture current frame landmarks
  N      → move to next gesture
  Q      → quit / save

Saves to:  data/gesture_dataset.csv
"""

import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp

DATA_PATH     = "data/gesture_dataset.csv"
SAMPLES_EACH  = 200    # target samples per gesture
GESTURE_QUEUE = [
    ("scissors",   "✂️  Make a scissors shape  (index + middle extended, spread apart)"),
    ("open_palm",  "🖐  Open palm              (all 5 fingers fully extended)"),
    ("ok_sign",    "👌  OK sign                (index tip touching thumb tip)"),
    ("fist",       "✊  Closed fist            (all fingers curled)"),
    ("one_finger", "☝️   One finger             (only index finger extended up)"),
    ("call_sign",  "🤙  Call sign              (thumb + pinky extended, others curled)"),
]

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ─── FEATURE EXTRACTION (must match gesture_engine.py) ────────────────────────────

def extract_features(landmarks) -> np.ndarray:
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
    )
    wrist   = coords[0]
    coords -= wrist
    scale   = np.max(np.abs(coords)) + 1e-6
    coords /= scale
    return coords.flatten()


# ─── MAIN COLLECTION LOOP ─────────────────────────────────────────────────────────

def collect():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load existing data count
    existing_rows = 0
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            existing_rows = sum(1 for _ in f) - 1   # minus header
        print(f"[Collect] Resuming — {existing_rows} existing samples found.")

    csv_file = open(DATA_PATH, "a", newline="")
    writer   = csv.writer(csv_file)

    # Write header if new file
    if existing_rows == 0:
        header = [f"f{i}" for i in range(63)] + ["label"]
        writer.writerow(header)

    for gesture_name, instruction in GESTURE_QUEUE:
        count = 0
        print(f"\n── Next gesture: {gesture_name} ─────────────────────")
        print(f"   {instruction}")
        print(f"   Press SPACE to capture · N to skip · Q to quit")

        while count < SAMPLES_EACH:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            # Overlay instruction text
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 90), (15, 15, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, gesture_name.upper(), (15, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (80, 220, 100), 2)
            cv2.putText(frame, f"Samples: {count}/{SAMPLES_EACH}", (15, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.putText(frame, "SPACE=capture  N=skip  Q=quit", (320, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 160), 1)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("GestureOS — Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[Collect] Quitting early.")
                cap.release()
                csv_file.close()
                cv2.destroyAllWindows()
                return

            elif key == ord("n"):
                print(f"[Collect] Skipping {gesture_name}")
                break

            elif key == ord(" ") and res.multi_hand_landmarks:
                features = extract_features(res.multi_hand_landmarks[0])
                row = list(features) + [gesture_name]
                writer.writerow(row)
                csv_file.flush()
                count += 1

                # Flash green feedback
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (640, 480), (50, 220, 80), 8)
                cv2.imshow("GestureOS — Data Collector", flash)
                cv2.waitKey(80)

        print(f"[Collect] ✓  {gesture_name}: {count} samples captured")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[Collect] Dataset saved → {DATA_PATH}")
    print("[Collect] Next step:  python train_model.py")


if __name__ == "__main__":
    collect()
