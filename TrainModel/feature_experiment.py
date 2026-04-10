# =========================================================
# FEATURE ABLATION STUDY - Test từng feature riêng lẻ
# =========================================================
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import glob
import random
import json
import time
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import librosa
import tensorflow as tf
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ================= GPU =================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ================= CONFIG =================
DATASET_PATH = r"../DATASET_LABELED"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

SAMPLE_RATE = 16000
WINDOW = 1.5
STEP = 0.75

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 30          # Giảm epoch để chạy nhanh hơn khi thí nghiệm
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
# ĐỊNH NGHĨA CÁC FEATURE CẦN THÍ NGHIỆM
# =========================================================
FEATURE_CONFIGS = {
    # --- Feature đơn lẻ ---
    "mel_64": {
        "desc": "Mel Spectrogram (64 bands)",
        "func": lambda y: librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
        ).T
    },
    "mel_128": {
        "desc": "Mel Spectrogram (128 bands)",
        "func": lambda y: librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
        ).T
    },
    "mfcc_13": {
        "desc": "MFCC (13 coefficients)",
        "func": lambda y: librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13).T
    },
    "mfcc_20": {
        "desc": "MFCC (20 coefficients)",
        "func": lambda y: librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20).T
    },
    "mfcc_40": {
        "desc": "MFCC (40 coefficients)",
        "func": lambda y: librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40).T
    },
    "delta_mfcc": {
        "desc": "Delta MFCC (20 coefficients)",
        "func": lambda y: librosa.feature.delta(
            librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
        ).T
    },
    "delta2_mfcc": {
        "desc": "Delta-Delta MFCC (20 coefficients)",
        "func": lambda y: librosa.feature.delta(
            librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20), order=2
        ).T
    },
    "chroma": {
        "desc": "Chroma (12 pitch classes)",
        "func": lambda y: librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE).T
    },
    "spectral_contrast": {
        "desc": "Spectral Contrast (7 bands)",
        "func": lambda y: librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE).T
    },
    "zcr": {
        "desc": "Zero Crossing Rate",
        "func": lambda y: librosa.feature.zero_crossing_rate(y).T
    },
    "rms": {
        "desc": "RMS Energy",
        "func": lambda y: librosa.feature.rms(y=y).T
    },

    # --- Tổ hợp (đang dùng trong train.py) ---
    "mel64_mfcc20": {
        "desc": "[CURRENT] Mel(64) + MFCC(20) — đang dùng trong train.py",
        "func": lambda y: np.concatenate([
            librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
            ),
            librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
        ], axis=0).T
    },

    # --- Tổ hợp mở rộng ---
    "mfcc20_delta_delta2": {
        "desc": "MFCC(20) + Delta + Delta2",
        "func": lambda y: _mfcc_delta_delta2(y)
    },
    "mel64_mfcc20_delta": {
        "desc": "Mel(64) + MFCC(20) + Delta MFCC",
        "func": lambda y: _mel_mfcc_delta(y)
    },
    "all_features": {
        "desc": "ALL: Mel(64) + MFCC(20) + Delta + Delta2 + Chroma + Contrast + ZCR + RMS",
        "func": lambda y: _all_features(y)
    },

    # --- Tổ hợp mới để so sánh ---
    "mel128_mfcc20": {
        "desc": "Mel(128) + MFCC(20)",
        "func": lambda y: _mel128_mfcc20(y)
    },
    "mfcc20_chroma": {
        "desc": "MFCC(20) + Chroma(12)",
        "func": lambda y: _mfcc20_chroma(y)
    },
    "mel64_chroma_contrast": {
        "desc": "Mel(64) + Chroma(12) + Spectral Contrast(7)",
        "func": lambda y: _mel64_chroma_contrast(y)
    },
    "mfcc20_delta_chroma": {
        "desc": "MFCC(20) + Delta + Chroma(12)",
        "func": lambda y: _mfcc20_delta_chroma(y)
    },
    "mel128_mfcc20_delta_delta2": {
        "desc": "Mel(128) + MFCC(20) + Delta + Delta2",
        "func": lambda y: _mel128_mfcc20_delta_delta2(y)
    },
    "mel64_contrast_zcr_rms": {
        "desc": "Mel(64) + Spectral Contrast + ZCR + RMS",
        "func": lambda y: _mel64_contrast_zcr_rms(y)
    },
    "mfcc40_delta_delta2": {
        "desc": "MFCC(40) + Delta + Delta2",
        "func": lambda y: _mfcc40_delta_delta2(y)
    },
    "mel128_mfcc40": {
        "desc": "Mel(128) + MFCC(40) — Heavy combo",
        "func": lambda y: _mel128_mfcc40(y)
    },
    "chroma_contrast_zcr_rms": {
        "desc": "Chroma + Contrast + ZCR + RMS — Prosody/Tonal",
        "func": lambda y: _chroma_contrast_zcr_rms(y)
    },
    "mel64_mfcc20_delta_delta2": {
        "desc": "Mel(64) + MFCC(20) + Delta + Delta2 — Full MFCC pipeline",
        "func": lambda y: _mel64_mfcc20_delta_delta2(y)
    },
}


# =========================================================
# HÀM TRÍCH XUẤT TỔ HỢP FEATURE
# =========================================================
def _mfcc_delta_delta2(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta, delta2], axis=0).T


def _mel_mfcc_delta(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    )
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([mel, mfcc, delta], axis=0).T


def _all_features(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    )
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    min_len = min(
        mel.shape[1], mfcc.shape[1], delta.shape[1], delta2.shape[1],
        chroma.shape[1], contrast.shape[1], zcr.shape[1], rms.shape[1]
    )

    return np.concatenate([
        mel[:, :min_len],
        mfcc[:, :min_len],
        delta[:, :min_len],
        delta2[:, :min_len],
        chroma[:, :min_len],
        contrast[:, :min_len],
        zcr[:, :min_len],
        rms[:, :min_len],
    ], axis=0).T


def _mel128_mfcc20(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
    )
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    min_len = min(mel.shape[1], mfcc.shape[1])
    return np.concatenate([mel[:, :min_len], mfcc[:, :min_len]], axis=0).T


def _mfcc20_chroma(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    min_len = min(mfcc.shape[1], chroma.shape[1])
    return np.concatenate([mfcc[:, :min_len], chroma[:, :min_len]], axis=0).T


def _mel64_chroma_contrast(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    )
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    min_len = min(mel.shape[1], chroma.shape[1], contrast.shape[1])
    return np.concatenate([
        mel[:, :min_len], chroma[:, :min_len], contrast[:, :min_len]
    ], axis=0).T


def _mfcc20_delta_chroma(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    min_len = min(mfcc.shape[1], delta.shape[1], chroma.shape[1])
    return np.concatenate([
        mfcc[:, :min_len], delta[:, :min_len], chroma[:, :min_len]
    ], axis=0).T


def _mel128_mfcc20_delta_delta2(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
    )
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    min_len = min(mel.shape[1], mfcc.shape[1], delta.shape[1], delta2.shape[1])
    return np.concatenate([
        mel[:, :min_len], mfcc[:, :min_len],
        delta[:, :min_len], delta2[:, :min_len]
    ], axis=0).T


def _mel64_contrast_zcr_rms(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    )
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    min_len = min(mel.shape[1], contrast.shape[1], zcr.shape[1], rms.shape[1])
    return np.concatenate([
        mel[:, :min_len], contrast[:, :min_len],
        zcr[:, :min_len], rms[:, :min_len]
    ], axis=0).T


def _mfcc40_delta_delta2(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta, delta2], axis=0).T


def _mel128_mfcc40(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
    )
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40)
    min_len = min(mel.shape[1], mfcc.shape[1])
    return np.concatenate([mel[:, :min_len], mfcc[:, :min_len]], axis=0).T


def _chroma_contrast_zcr_rms(y):
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    min_len = min(chroma.shape[1], contrast.shape[1], zcr.shape[1], rms.shape[1])
    return np.concatenate([
        chroma[:, :min_len], contrast[:, :min_len],
        zcr[:, :min_len], rms[:, :min_len]
    ], axis=0).T


def _mel64_mfcc20_delta_delta2(y):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    )
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    min_len = min(mel.shape[1], mfcc.shape[1], delta.shape[1], delta2.shape[1])
    return np.concatenate([
        mel[:, :min_len], mfcc[:, :min_len],
        delta[:, :min_len], delta2[:, :min_len]
    ], axis=0).T


# =========================================================
# AUDIO UTILS
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
        segments.append(y[i:i + win_len])

    return segments


def pad_or_truncate(feat, max_len=MAX_LEN):
    """Pad hoặc truncate feature về chiều dài cố định."""
    if feat.shape[0] > max_len:
        feat = feat[:max_len]
    elif feat.shape[0] < max_len:
        pad_width = max_len - feat.shape[0]
        feat = np.pad(feat, ((0, pad_width), (0, 0)))
    return feat


# =========================================================
# LOAD DATA CHO 1 FEATURE CONFIG
# =========================================================
def load_data_for_feature(files, feature_func):
    """Load audio và trích xuất feature cho danh sách file."""
    X, y, groups = [], [], []

    for f in files:
        label = EMOTION_MAP[os.path.basename(os.path.dirname(f))]
        audio = load_audio(f)
        segments = split_audio(audio)

        for seg in segments:
            try:
                feat = feature_func(seg)
                feat = pad_or_truncate(feat)
                X.append(feat)
                y.append(label)
                groups.append(f)
            except Exception as e:
                print(f"  ⚠️ Skip segment from {os.path.basename(f)}: {e}")
                continue

    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)


# =========================================================
# BUILD MODEL (tự động theo input shape)
# =========================================================
def build_model(input_shape):
    """CNN + BiLSTM model, tự động điều chỉnh theo feature dimension."""
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
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =========================================================
# CHẠY THÍ NGHIỆM CHO 1 FEATURE
# =========================================================
def run_experiment(feature_name, feature_config, train_files, test_files):
    """Chạy train + evaluate cho 1 feature config."""
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT: {feature_name}")
    print(f"   {feature_config['desc']}")
    print(f"{'='*60}")

    feature_func = feature_config["func"]

    # Load data
    print("  📂 Loading training data...")
    X_train, y_train, g_train = load_data_for_feature(train_files, feature_func)

    print("  📂 Loading test data...")
    X_test, y_test, g_test = load_data_for_feature(test_files, feature_func)

    if len(X_train) == 0 or len(X_test) == 0:
        print("  ❌ No data loaded, skipping...")
        return None

    feature_dim = X_train.shape[2]
    print(f"  📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"  📐 Feature shape: ({MAX_LEN}, {feature_dim})")

    # Normalize
    scaler = StandardScaler()
    T, F = X_train.shape[1], X_train.shape[2]
    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(-1, T, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(-1, T, F)

    y_train_oh = to_categorical(y_train, NUM_CLASSES)

    # Build & Train
    model = build_model((T, F))
    total_params = model.count_params()
    print(f"  🏗️ Model params: {total_params:,}")

    start_time = time.time()

    history = model.fit(
        X_train, y_train_oh,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(patience=6, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=3, verbose=0)
        ],
        verbose=0
    )

    train_time = time.time() - start_time

    # Best val accuracy from history
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    best_train_acc = max(history.history.get('accuracy', [0]))
    actual_epochs = len(history.history['loss'])

    # Predict with voting
    probs = model.predict(X_test, verbose=0)

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

    # Metrics
    acc = accuracy_score(final_labels, final_preds)
    report = classification_report(
        final_labels, final_preds,
        target_names=EMOTION_LABELS,
        output_dict=True
    )

    report_text = classification_report(
        final_labels, final_preds,
        target_names=EMOTION_LABELS
    )

    print(f"\n  📈 Results:")
    print(f"     Train Acc:   {best_train_acc:.4f}")
    print(f"     Val Acc:     {best_val_acc:.4f}")
    print(f"     Test Acc:    {acc:.4f}")
    print(f"     Epochs:      {actual_epochs}/{EPOCHS}")
    print(f"     Time:        {train_time:.1f}s")
    print(f"\n{report_text}")

    # Clean up
    del model, X_train, X_test
    tf.keras.backend.clear_session()

    return {
        "feature": feature_name,
        "desc": feature_config["desc"],
        "feature_dim": feature_dim,
        "train_samples": int(len(y_train)),
        "test_files": len(file_probs),
        "train_acc": round(float(best_train_acc), 4),
        "val_acc": round(float(best_val_acc), 4),
        "test_acc": round(float(acc), 4),
        "per_class": {
            emo: round(report[emo]["f1-score"], 4)
            for emo in EMOTION_LABELS if emo in report
        },
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "epochs": actual_epochs,
        "train_time_s": round(train_time, 1),
        "params": total_params,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 60)
    print("🔬 FEATURE ABLATION STUDY — Speech Emotion Recognition")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Collect all files
    all_files = []
    for emo in EMOTION_MAP:
        files = glob.glob(os.path.join(DATASET_PATH, emo, "*.wav"))
        all_files += files
        print(f"  {emo}: {len(files)} files")
    print(f"  Total: {len(all_files)} files\n")

    # Split train/test (file-level, stratified)
    labels = [EMOTION_MAP[os.path.basename(os.path.dirname(f))] for f in all_files]
    train_files, test_files = train_test_split(
        all_files, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"  Train files: {len(train_files)} | Test files: {len(test_files)}\n")

    # Chọn feature để test (có thể comment bớt để chạy nhanh)
    features_to_test = [
        # --- Feature đơn lẻ ---
        "mel_64",
        "mel_128",
        "mfcc_13",
        "mfcc_20",
        "mfcc_40",
        "delta_mfcc",
        "delta2_mfcc",
        "chroma",
        "spectral_contrast",
        "zcr",
        "rms",
        # --- Tổ hợp cũ ---
        "mel64_mfcc20",          # <- đang dùng trong train.py
        "mfcc20_delta_delta2",
        "mel64_mfcc20_delta",
        "all_features",
        # --- Tổ hợp mới ---
        "mel128_mfcc20",
        "mfcc20_chroma",
        "mel64_chroma_contrast",
        "mfcc20_delta_chroma",
        "mel128_mfcc20_delta_delta2",
        "mel64_contrast_zcr_rms",
        "mfcc40_delta_delta2",
        "mel128_mfcc40",
        "chroma_contrast_zcr_rms",
        "mel64_mfcc20_delta_delta2",
    ]

    results = []

    for feat_name in features_to_test:
        if feat_name not in FEATURE_CONFIGS:
            print(f"⚠️ Feature '{feat_name}' not found, skipping.")
            continue

        try:
            result = run_experiment(
                feat_name, FEATURE_CONFIGS[feat_name],
                train_files, test_files
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ❌ ERROR in {feat_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # =========================================================
    # TỔNG KẾT
    # =========================================================
    if not results:
        print("\n❌ No results to show.")
        return

    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT SO SÁNH FEATURE")
    print("=" * 80)

    # Sort by test accuracy
    results.sort(key=lambda x: x["test_acc"], reverse=True)

    # Header
    print(f"\n{'Rank':<5} {'Feature':<25} {'Dim':<5} {'Test Acc':<10} "
          f"{'Val Acc':<10} {'W-F1':<8} {'M-F1':<8} {'Time':<8}")
    print("-" * 80)

    for i, r in enumerate(results, 1):
        marker = " ⭐" if r["feature"] == "mel64_mfcc20" else ""
        print(f"{i:<5} {r['feature']:<25} {r['feature_dim']:<5} "
              f"{r['test_acc']:<10.4f} {r['val_acc']:<10.4f} "
              f"{r['weighted_f1']:<8.4f} {r['macro_f1']:<8.4f} "
              f"{r['train_time_s']:<8.1f}{marker}")

    # Per-class breakdown
    print(f"\n{'Feature':<25}", end="")
    for emo in EMOTION_LABELS:
        print(f" {emo:<8}", end="")
    print()
    print("-" * 70)

    for r in results:
        print(f"{r['feature']:<25}", end="")
        for emo in EMOTION_LABELS:
            f1 = r["per_class"].get(emo, 0)
            print(f" {f1:<8.4f}", end="")
        print()

    # Save results
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_DIR, "feature_experiment_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "sample_rate": SAMPLE_RATE,
                "window": WINDOW,
                "step": STEP,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved: {result_path}")

    # Best recommendation
    best = results[0]
    current = next((r for r in results if r["feature"] == "mel64_mfcc20"), None)

    print(f"\n🏆 BEST FEATURE: {best['feature']} — Test Acc: {best['test_acc']:.4f}")
    if current:
        print(f"📍 CURRENT (mel64_mfcc20): Test Acc: {current['test_acc']:.4f}")
        diff = best['test_acc'] - current['test_acc']
        if diff > 0:
            print(f"   → Cải thiện tiềm năng: +{diff:.4f} ({diff*100:.2f}%)")
        else:
            print(f"   → Feature hiện tại đã là tốt nhất!")

    print("\n🎉 Feature experiment completed!")


if __name__ == "__main__":
    main()
