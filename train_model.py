"""
train_model.py
──────────────
Train a lightweight TensorFlow gesture classifier from collected data.

Usage
─────
  # Step 1: collect samples
  python collect_data.py

  # Step 2: train
  python train_model.py

  # Step 3: run app
  streamlit run app.py

Architecture
────────────
  Input  → 63-d vector (21 landmarks × x/y/z, normalised)
  Dense(256, relu) → Dropout(0.4)
  Dense(128, relu) → Dropout(0.3)
  Dense(64,  relu) → Dropout(0.2)
  Dense(6,   softmax)           ← 6 gesture classes

Training details
────────────────
  - Stratified 80/20 split
  - Adam, lr=1e-3, cosine-decay
  - Label smoothing 0.1
  - EarlyStopping (patience=15)
  - ReduceLROnPlateau (patience=7)
  - Saves best weights to models/gesture_model.h5
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("models", exist_ok=True)

# ── Try TF import ──────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, regularizers
    print(f"[Train] TensorFlow {tf.__version__} ✓")
except ImportError:
    raise SystemExit("TensorFlow not installed. Run: pip install tensorflow")


GESTURE_CLASSES = ["scissors", "open_palm", "ok_sign", "fist", "one_finger", "call_sign"]
DATA_PATH       = "data/gesture_dataset.csv"
MODEL_PATH      = "models/gesture_model.h5"
HISTORY_PATH    = "models/training_history.json"
EPOCHS          = 150
BATCH_SIZE      = 32


# ─── LOAD DATA ────────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Run:  python collect_data.py  to collect samples first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"[Train] Loaded {len(df)} samples across {df['label'].nunique()} classes")
    print(df["label"].value_counts().to_string())

    X = df.drop(columns=["label"]).values.astype(np.float32)   # (N, 63)
    y = df["label"].values

    le = LabelEncoder()
    le.fit(GESTURE_CLASSES)
    y_enc = le.transform(y)
    return X, y_enc, le


# ─── BUILD MODEL ─────────────────────────────────────────────────────────────────

def build_model(input_dim: int = 63, num_classes: int = 6) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,), name="landmarks")

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inp, out, name="GestureClassifier")
    model.summary()
    return model


# ─── TRAINING ────────────────────────────────────────────────────────────────────

def train():
    X, y, le = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[Train] Train: {len(X_train)}  Val: {len(X_val)}")

    model = build_model()

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=EPOCHS * (len(X_train) // BATCH_SIZE),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    cb = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1,
    )

    # ── Save history ──────────────────────────────────────────────────────────
    with open(HISTORY_PATH, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()}, f
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#080C18")

    for ax in axes:
        ax.set_facecolor("#0D1428")

    # Loss curve
    axes[0].plot(history.history["loss"],     color="#4A90D9", label="Train Loss")
    axes[0].plot(history.history["val_loss"], color="#DC50FF", label="Val Loss",  linestyle="--")
    axes[0].set_title("Loss Curves",     color="#C8D6F0")
    axes[0].set_xlabel("Epoch",          color="#6888B0")
    axes[0].set_ylabel("Loss",           color="#6888B0")
    axes[0].legend()
    axes[0].tick_params(colors="#6888B0")

    # Confusion matrix
    sns.heatmap(
        cm, annot=True, fmt="d", ax=axes[1],
        xticklabels=le.classes_, yticklabels=le.classes_,
        cmap="Blues", linewidths=0.5,
    )
    axes[1].set_title("Confusion Matrix", color="#C8D6F0")
    axes[1].tick_params(colors="#6888B0")

    plt.tight_layout()
    plt.savefig("models/training_results.png", dpi=150, facecolor="#080C18")
    print("[Train] Saved plot → models/training_results.png")
    print(f"[Train] Model saved → {MODEL_PATH} ✓")


if __name__ == "__main__":
    train()
