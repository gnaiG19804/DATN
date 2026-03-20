
import os
import glob
import numpy as np
import random
import librosa
import joblib
import tensorflow as tf

# Fix GPU memory — chỉ dùng đúng lượng VRAM cần thiết
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization,
    Bidirectional, GlobalAveragePooling1D, GaussianNoise
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =============================================================================
# CONFIG
# =============================================================================
DATASET_PATH = "../DATASET_LABELED"

SAMPLE_RATE = 16000
DURATION = 4.0
HOP_LENGTH = 512

BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4

EMOTION_MAP = {
    "ANG": 0,
    "ANX": 1,
    "HAP": 2,
    "NEU": 3,
    "SAD": 4
}

NUM_CLASSES = len(EMOTION_MAP)
EMOTION_LABELS = list(EMOTION_MAP.keys())

# Thư mục lưu kết quả
RESULTS_DIR = "results"


# =============================================================================
# AUDIO
# =============================================================================
def pad_audio(y):
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) > target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=25)
    return pad_audio(y)


# =============================================================================
# FEATURE EXTRACTION (FIXED)
# =============================================================================
def extract_features(y):
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=40, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel)

    # FIX: đảm bảo cùng hop_length
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0)

    if np.std(f0) > 0:
        f0 = (f0 - np.mean(f0)) / (np.std(f0) + 1e-6)

    f0 = f0.reshape(1, -1)

    # đảm bảo cùng time length
    min_len = min(
        zcr.shape[1], rms.shape[1], mfcc.shape[1],
        mel_db.shape[1], f0.shape[1]
    )

    zcr = zcr[:, :min_len]
    rms = rms[:, :min_len]
    mfcc = mfcc[:, :min_len]
    delta = delta[:, :min_len]
    delta2 = delta2[:, :min_len]
    mel_db = mel_db[:, :min_len]
    f0 = f0[:, :min_len]

    features = np.concatenate([
        zcr, rms,
        mfcc, delta, delta2,
        mel_db,
        f0
    ], axis=0)

    return features.T


# =============================================================================
# AUGMENTATION
# =============================================================================
def augment(y):
    # Noise injection
    if random.random() < 0.5:
        noise_level = np.random.uniform(0.005, 0.025)
        y = y + noise_level * np.random.randn(len(y))

    # Time stretch
    if random.random() < 0.5:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)

    # Time shift
    if random.random() < 0.4:
        shift = int(np.random.uniform(-0.1, 0.1) * len(y))
        y = np.roll(y, shift)

    # Pitch shift
    if random.random() < 0.3:
        n_steps = np.random.uniform(-1.5, 1.5)
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)

    # Volume change
    if random.random() < 0.4:
        gain = np.random.uniform(0.8, 1.2)
        y = y * gain

    return pad_audio(y)


# =============================================================================
# LOAD DATA (FIXED)
# =============================================================================
def load_data():
    X, y, files = [], [], []

    abs_path = os.path.abspath(DATASET_PATH)
    print(f"  Dataset path: {abs_path}")
    print(f"  Exists: {os.path.isdir(abs_path)}")

    for emo, idx in EMOTION_MAP.items():
        folder = os.path.join(DATASET_PATH, emo)
        wav_files = glob.glob(folder + "/*.wav")
        print(f"  [{emo}] {folder} → {len(wav_files)} files")

        for file in wav_files:
            try:
                audio = load_audio(file)
                feat = extract_features(audio)

                X.append(feat)
                y.append(idx)
                files.append(file)

            except Exception as e:
                print(f"[WARN] {file}: {e}")

    return np.array(X), np.array(y, dtype=int), files


# =============================================================================
# BALANCE (FIXED)
# =============================================================================
def balance_data(X, y, files):
    max_count = max(np.bincount(y))
    target = int(max_count * 0.7)

    X_new, y_new = [], []

    for c in range(NUM_CLASSES):
        idxs = np.where(y == c)[0]

        # keep original
        for i in idxs:
            X_new.append(X[i])
            y_new.append(c)

        # augment
        needed = max(0, target - len(idxs))

        for _ in range(needed):
            i = random.choice(idxs)
            audio = load_audio(files[i])
            audio = augment(audio)
            feat = extract_features(audio)

            X_new.append(feat)
            y_new.append(c)

    return np.array(X_new), np.array(y_new)


# =============================================================================
# MODEL
# =============================================================================
def build_model(input_shape):
    inp = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    out = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=Adam(LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=['accuracy']
    )

    return model


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_training_history(history, results_dir):
    """Lưu biểu đồ Loss và Accuracy theo epoch."""

    # --- Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved loss_curve.png")

    # --- Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved accuracy_curve.png")


def save_confusion_matrix(y_true, y_pred, results_dir):
    """Lưu confusion matrix dạng heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved confusion_matrix.png")

    # Normalized confusion matrix (%)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt='.1f', cmap='Oranges',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS
    )
    plt.title('Confusion Matrix (Normalized %)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_normalized.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved confusion_matrix_normalized.png")


def save_classification_report(y_true, y_pred, results_dir):
    """Lưu classification report ra file text."""
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS)

    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"SER v5.1 — Classification Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)
        f.write(f"\n{'='*60}\n")

    print(f"  ✓ Saved classification_report.txt")
    return report


def save_training_summary(history, y_true, y_pred, results_dir):
    """Lưu tổng hợp kết quả training."""
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True)

    summary_path = os.path.join(results_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"SER v5.1 — Training Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        # Config
        f.write(f"[CONFIG]\n")
        f.write(f"  Sample Rate    : {SAMPLE_RATE}\n")
        f.write(f"  Duration       : {DURATION}s\n")
        f.write(f"  Batch Size     : {BATCH_SIZE}\n")
        f.write(f"  Max Epochs     : {EPOCHS}\n")
        f.write(f"  Learning Rate  : {LR}\n")
        f.write(f"  Emotions       : {EMOTION_LABELS}\n\n")

        # Training info
        actual_epochs = len(history.history['loss'])
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_accuracy'])
        final_train_acc = history.history['accuracy'][-1]
        final_train_loss = history.history['loss'][-1]

        f.write(f"[TRAINING]\n")
        f.write(f"  Actual Epochs  : {actual_epochs}\n")
        f.write(f"  Final Train Loss     : {final_train_loss:.4f}\n")
        f.write(f"  Final Train Accuracy : {final_train_acc:.4f} ({final_train_acc*100:.1f}%)\n")
        f.write(f"  Best Val Loss        : {best_val_loss:.4f}\n")
        f.write(f"  Best Val Accuracy    : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)\n\n")

        # Test results
        test_acc = report['accuracy']
        f.write(f"[TEST RESULTS]\n")
        f.write(f"  Test Accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)\n\n")

        # Per-class results
        f.write(f"[PER-CLASS RESULTS]\n")
        f.write(f"  {'Emotion':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write(f"  {'-'*50}\n")
        for emo in EMOTION_LABELS:
            p = report[emo]['precision']
            r = report[emo]['recall']
            f1 = report[emo]['f1-score']
            sup = report[emo]['support']
            f.write(f"  {emo:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {sup:>10.0f}\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"Files saved in: {os.path.abspath(results_dir)}\n")

    print(f"  ✓ Saved training_summary.txt")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Tạo thư mục results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}")

    print("Loading data...")
    X, y, files = load_data()

    print(f"  Total samples: {len(y)}")
    if len(y) == 0:
        print("ERROR: No data loaded! Check DATASET_PATH.")
        return

    print("Balancing data...")
    X, y = balance_data(X, y, files)

    print("Split...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Normalize...")
    scaler = StandardScaler()
    T, F = X_tr.shape[1], X_tr.shape[2]

    X_tr = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(-1, T, F)
    X_te = scaler.transform(X_te.reshape(-1, F)).reshape(-1, T, F)

    joblib.dump(scaler, "scaler.pkl")

    y_tr_oh = to_categorical(y_tr)
    y_te_oh = to_categorical(y_te)

    # class weight
    weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights = dict(enumerate(weights))

    print("Build model...")
    model = build_model((T, F))

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    print("Training...")
    history = model.fit(
        X_tr, y_tr_oh,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save("final_model.keras")

    print("Evaluate...")
    preds = np.argmax(model.predict(X_te), axis=1)

    # In ra console
    report_text = classification_report(y_te, preds, target_names=EMOTION_LABELS)
    print(report_text)

    # =========================================================================
    # LƯU KẾT QUẢ
    # =========================================================================
    print(f"\nSaving results to '{RESULTS_DIR}/'...")

    save_training_history(history, RESULTS_DIR)
    save_confusion_matrix(y_te, preds, RESULTS_DIR)
    save_classification_report(y_te, preds, RESULTS_DIR)
    save_training_summary(history, y_te, preds, RESULTS_DIR)

    print(f"\n✅ All results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print(f"   - loss_curve.png")
    print(f"   - accuracy_curve.png")
    print(f"   - confusion_matrix.png")
    print(f"   - confusion_matrix_normalized.png")
    print(f"   - classification_report.txt")
    print(f"   - training_summary.txt")


if __name__ == "__main__":
    main()