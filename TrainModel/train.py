# =========================================================
# SER FINAL FIXED (BALANCE + MFCC + BETTER VOTING)
# =========================================================

import os
import glob
import random
import json
from datetime import datetime

import numpy as np
import librosa
import joblib
import tensorflow as tf

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical

# =========================================================
# CONFIG
# =========================================================
DATASET_PATH = r"E:\KHMT\N4K2\DATN\DATASET_LABELED"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

SAMPLE_RATE = 16000
WINDOW = 1.5
STEP = 0.75

N_MELS = 64
MAX_LEN = 128

BATCH_SIZE = 32
EPOCHS = 50
LR = 2e-4

EMOTION_MAP = {
    "ANG": 0,
    "ANX": 1,
    "HAP": 2,
    "NEU": 3,
    "SAD": 4
}

NUM_CLASSES = len(EMOTION_MAP)
EMOTION_LABELS = list(EMOTION_MAP.keys())

# =========================================================
# AUDIO
# =========================================================
def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y)
    return y

def split_audio(y):
    win_len = int(WINDOW * SAMPLE_RATE)
    step_len = int(STEP * SAMPLE_RATE)

    if len(y) < win_len:
        y = np.pad(y, (0, win_len - len(y)))
        return [y]

    segments = []
    for i in range(0, len(y) - win_len + 1, step_len):
        segments.append(y[i:i+win_len])

    return segments

# =========================================================
# FEATURE (MEL + MFCC)
# =========================================================
def extract_features(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel = librosa.power_to_db(mel)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)

    feat = np.concatenate([mel, mfcc], axis=0).T

    if feat.shape[0] > MAX_LEN:
        feat = feat[:MAX_LEN]
    else:
        pad = MAX_LEN - feat.shape[0]
        feat = np.pad(feat, ((0, pad), (0, 0)))

    return feat

# =========================================================
# AUGMENT
# =========================================================
def augment(y):
    if random.random() < 0.7:
        y += 0.02 * np.random.randn(len(y))

    if random.random() < 0.7:
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=random.uniform(-3, 3))

    if random.random() < 0.7:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))

    return y

# =========================================================
# LOAD DATA
# =========================================================
def load_data():
    X, y, groups = [], [], []

    for emo, idx in EMOTION_MAP.items():
        files = glob.glob(os.path.join(DATASET_PATH, emo, "*.wav"))
        print(f"{emo}: {len(files)} files")

        for f in files:
            audio = load_audio(f)
            segments = split_audio(audio)

            for seg in segments:
                feat = extract_features(seg)

                X.append(feat)
                y.append(idx)
                groups.append(f)

                if random.random() < 0.5:
                    seg_aug = augment(seg)
                    feat_aug = extract_features(seg_aug)

                    X.append(feat_aug)
                    y.append(idx)
                    groups.append(f)

    return np.array(X), np.array(y), np.array(groups)

# =========================================================
# BALANCE DATA (KEY FIX)
# =========================================================
def balance_dataset(X, y, groups):
    class_data = defaultdict(list)

    for xi, yi, gi in zip(X, y, groups):
        class_data[yi].append((xi, yi, gi))

    max_count = max(len(v) for v in class_data.values())

    X_new, y_new, g_new = [], [], []

    for cls in class_data:
        samples = class_data[cls]

        while len(samples) < max_count:
            samples.append(random.choice(samples))

        for xi, yi, gi in samples:
            X_new.append(xi)
            y_new.append(yi)
            g_new.append(gi)

    return np.array(X_new), np.array(y_new), np.array(g_new)

# =========================================================
# MODEL (STRONGER)
# =========================================================
def build_model(input_shape):
    inp = Input(shape=input_shape)

    x = Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=Adam(LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading data...")
    X, y, groups = load_data()

    print("Balancing data...")
    X, y, groups = balance_dataset(X, y, groups)

    print(f"Total samples: {len(X)}")

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    T, F = X_train.shape[1], X_train.shape[2]

    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(-1, T, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(-1, T, F)

    y_train_oh = to_categorical(y_train)

    print("Building model...")
    model = build_model((T, F))

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=3)
    ]

    print("Training...")
    model.fit(
        X_train, y_train_oh,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Predicting...")
    probs = model.predict(X_test)

    # =====================================================
    # VOTING (AVERAGE PROBABILITY)
    # =====================================================
    file_probs = defaultdict(list)

    for p, g in zip(probs, g_test):
        file_probs[g].append(p)

    final_preds = []
    final_labels = []

    for f in file_probs:
        avg_prob = np.mean(file_probs[f], axis=0)
        pred = np.argmax(avg_prob)

        label = EMOTION_MAP[os.path.basename(os.path.dirname(f))]

        final_preds.append(pred)
        final_labels.append(label)

    report = classification_report(final_labels, final_preds, target_names=EMOTION_LABELS)
    print("\n🔥 FINAL RESULT:")
    print(report)

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"\n💾 Saving results to {RESULT_DIR}...")

    # 1. Model
    model_path = os.path.join(RESULT_DIR, "ser_model.keras")
    model.save(model_path)
    print(f"  ✅ Model saved: {model_path}")

    # 2. Scaler
    scaler_path = os.path.join(RESULT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Scaler saved: {scaler_path}")

    # 3. Classification report
    report_path = os.path.join(RESULT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ Report saved: {report_path}")

    # 4. Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    cm_path = os.path.join(RESULT_DIR, "confusion_matrix.npy")
    np.save(cm_path, cm)
    print(f"  ✅ Confusion matrix saved: {cm_path}")

    # 5. Training config
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": DATASET_PATH,
        "sample_rate": SAMPLE_RATE,
        "window": WINDOW,
        "step": STEP,
        "n_mels": N_MELS,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "num_classes": NUM_CLASSES,
        "emotion_labels": EMOTION_LABELS,
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    config_path = os.path.join(RESULT_DIR, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Config saved: {config_path}")

    print("\n🎉 All results saved!")


# =========================================================
if __name__ == "__main__":
    main()