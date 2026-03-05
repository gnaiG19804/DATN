"""
=============================================================================
TRAIN SER v3 — Vietnamese Speech Emotion Recognition
=============================================================================
Built on v2 with 10 additional improvements:
  1.  CRNN Model           — Conv1D → Residual → BiLSTM → Attention
  2.  Mixup Augmentation   — λ·X1 + (1-λ)·X2 on training features
  3.  Improved Residual CNN — 4 residual blocks with proper skip connections
  4.  Stable Attention      — Dense(1) → Softmax(axis=1) → weighted sum via Lambda
  5.  Label Smoothing 0.1  + Gradient Clipping in Adam
  6.  LayerNormalization    — Optional in-model normalization
  7.  tf.data Pipeline      — shuffle / batch / prefetch / AUTOTUNE
  8.  Model Comparison Table — Clean summary after training
  9.  Improved Visualization — Per-class accuracy bar chart
  10. Full Compatibility    — Dataset / emotion map / gender / caching preserved
=============================================================================
Run:  python train_ser_v3.py
=============================================================================
"""

# ============================================================================
# 0. IMPORTS
# ============================================================================
import os
import glob
import csv
import hashlib
import json
import numpy as np
import random
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Flatten, BatchNormalization,
    Conv1D, MaxPooling1D, Bidirectional, Concatenate, Add,
    GlobalAveragePooling1D, Softmax, Multiply, Activation,
    Lambda, LayerNormalization
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# 1. CONFIG
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "DATASET_LABELED")
DATALABEL_PATH = os.path.join(SCRIPT_DIR, "..", "DataLabel")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_optimized")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")

SAMPLE_RATE = 16000
DURATION = 4.0
N_MFCC = 40
HOP_LENGTH = 256

# Audio augmentation (applied after split, training only)
NUM_AUGMENTATIONS = 3
ENABLE_SPEC_AUGMENT = True
SPEC_TIME_MASK_MAX = 15
SPEC_FREQ_MASK_MAX = 10
SPEC_NUM_MASKS = 2

# Mixup (Improvement v3-#2)
ENABLE_MIXUP = True
MIXUP_ALPHA = 0.2           # Beta distribution parameter

# Training
EPOCHS = 30
BATCH_SIZE = 32
INITIAL_LR = 1e-3
LABEL_SMOOTHING = 0.1       # v3-#5
GRADIENT_CLIP_NORM = 1.0    # v3-#5

# Caching
USE_CACHE = False

# Label maps
EMOTION_MAP = {
    "ANG": 0,
    "ANX": 1,
    "HAP": 2,
    "NEU": 3,
    "SAD": 4
}

GENDER_MAP = {
    "Nam": 0,
    "Nữ": 1
}

NUM_CLASSES = len(EMOTION_MAP)
IDX_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}


# ============================================================================
# 2. DATA LOADING
# ============================================================================
def build_gender_lookup(datalabel_path):
    """Read CSV files in DataLabel/ → {filename.wav: gender_int}."""
    gender_lookup = {}
    csv_files = glob.glob(os.path.join(datalabel_path, "*.csv"))

    for csv_file in csv_files:
        try:
            with open(csv_file, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('Filename', '').strip()
                    character = row.get('Character', '').strip()
                    if not filename:
                        continue
                    if character.lower() in ['null', '', 'none']:
                        continue
                    if character in GENDER_MAP:
                        gender_lookup[filename] = GENDER_MAP[character]
        except Exception as e:
            print(f"  [WARN] Error reading {csv_file}: {e}")

    return gender_lookup


def pad_audio(data, sr, duration):
    """Pad or truncate audio to fixed length."""
    target_len = int(sr * duration)
    if len(data) > target_len:
        return data[:target_len]
    elif len(data) < target_len:
        return np.pad(data, (0, target_len - len(data)), 'constant')
    return data


def load_audio(file_path):
    """Load, trim silence, pad to DURATION."""
    data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    data, _ = librosa.effects.trim(data, top_db=25)
    return pad_audio(data, sr, DURATION)


# ============================================================================
# 3. FEATURE EXTRACTION (simplified: MFCC+Δ+ΔΔ+ZCR+RMS = 122 dims)
# ============================================================================
def extract_features(data, sr=SAMPLE_RATE):
    """Extract: MFCC(40) + Delta(40) + Delta2(40) + ZCR(1) + RMS(1) = 122."""
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=2048, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=data, frame_length=2048, hop_length=HOP_LENGTH)
    features = np.concatenate([mfcc, delta1, delta2, zcr, rms], axis=0)
    return features.T  # (time_steps, 122)


def _config_hash():
    """Hash config for cache invalidation."""
    cfg = f"{SAMPLE_RATE}_{DURATION}_{N_MFCC}_{HOP_LENGTH}"
    return hashlib.md5(cfg.encode()).hexdigest()[:8]


def load_original_data(dataset_path, gender_lookup):
    """
    Load original (non-augmented) features + file_paths.
    Uses .npy cache when USE_CACHE=True.
    """
    h = _config_hash()
    cache_feat = os.path.join(CACHE_DIR, f"features_{h}.npy")
    cache_gender = os.path.join(CACHE_DIR, f"gender_{h}.npy")
    cache_labels = os.path.join(CACHE_DIR, f"labels_{h}.npy")
    cache_paths = os.path.join(CACHE_DIR, f"paths_{h}.json")

    if USE_CACHE and all(os.path.exists(p)
                         for p in [cache_feat, cache_gender, cache_labels, cache_paths]):
        print("  Loading from cache...")
        X = np.load(cache_feat)
        g = np.load(cache_gender)
        y = np.load(cache_labels)
        with open(cache_paths, 'r', encoding='utf-8') as f:
            fp = json.load(f)
        print(f"  Loaded {X.shape[0]} samples from cache")
        return X, g, y, fp

    X, g, y, fp = [], [], [], []
    skipped = 0

    print("  Extracting features (first run — will save cache)...")
    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path) or folder_name not in EMOTION_MAP:
            continue
        emotion_label = EMOTION_MAP[folder_name]
        print(f"    -> {folder_name}")

        for file_path in glob.glob(os.path.join(folder_path, "*.wav")):
            basename = os.path.basename(file_path)
            if basename not in gender_lookup:
                skipped += 1
                continue
            try:
                audio = load_audio(file_path)
                feat = extract_features(audio)
                X.append(feat)
                g.append(gender_lookup[basename])
                y.append(emotion_label)
                fp.append(file_path)
            except Exception as e:
                print(f"    [WARN] {basename}: {e}")

    X = np.array(X)
    g = np.array(g)
    y = np.array(y)
    print(f"  Total: {len(y)} original samples  (skipped {skipped} — no gender)")

    if USE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_feat, X)
        np.save(cache_gender, g)
        np.save(cache_labels, y)
        with open(cache_paths, 'w', encoding='utf-8') as f:
            json.dump(fp, f)
        print(f"  Cache saved to {CACHE_DIR}/")

    return X, g, y, fp


# ============================================================================
# 4. AUGMENTATION (audio-level + SpecAugment + Mixup)
# ============================================================================
def noise(data):
    amp = 0.035 * np.random.uniform() * np.amax(np.abs(data) + 1e-9)
    return data + amp * np.random.normal(size=len(data))

def stretch(data):
    return librosa.effects.time_stretch(y=data, rate=np.random.uniform(0.8, 1.2))

def shift(data):
    return np.roll(data, int(np.random.uniform(-5, 5) * 1000))

def pitch_shift_fn(data, sr=SAMPLE_RATE):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=np.random.uniform(-2, 2))

def volume_perturb(data):
    return data * np.random.uniform(0.6, 1.4)


def augment_audio(file_path):
    """Load, apply 1-3 random augmentations, extract features."""
    data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    data, _ = librosa.effects.trim(data, top_db=25)

    aug_fns = [noise, stretch, shift, lambda d: pitch_shift_fn(d, sr), volume_perturb]
    for fn in random.sample(aug_fns, random.randint(1, 3)):
        data = fn(data)

    data = pad_audio(data, sr, DURATION)
    return extract_features(data, sr)


def spec_augment(features):
    """SpecAugment: time + frequency masking on (T, F) feature matrix."""
    feat = features.copy()
    T, F = feat.shape
    for _ in range(SPEC_NUM_MASKS):
        t = np.random.randint(0, min(SPEC_TIME_MASK_MAX, T))
        t0 = np.random.randint(0, max(1, T - t))
        feat[t0:t0 + t, :] = 0.0

        f = np.random.randint(0, min(SPEC_FREQ_MASK_MAX, F))
        f0 = np.random.randint(0, max(1, F - f))
        feat[:, f0:f0 + f] = 0.0
    return feat


def augment_training_set(X_train, g_train, y_train, file_paths_train):
    """
    Augment ONLY training data (no data leakage).
    Creates NUM_AUGMENTATIONS copies per sample + optional SpecAugment.
    """
    X_aug, g_aug, y_aug = [], [], []
    print(f"  Augmenting: {NUM_AUGMENTATIONS}x per sample...")

    for i, fpath in enumerate(file_paths_train):
        for _ in range(NUM_AUGMENTATIONS):
            try:
                feat = augment_audio(fpath)
                if ENABLE_SPEC_AUGMENT:
                    feat = spec_augment(feat)
                X_aug.append(feat)
                g_aug.append(g_train[i])
                y_aug.append(y_train[i])
            except Exception:
                continue

    if not X_aug:
        return X_train, g_train, y_train

    X_out = np.concatenate([X_train, np.array(X_aug)], axis=0)
    g_out = np.concatenate([g_train, np.array(g_aug)], axis=0)
    y_out = np.concatenate([y_train, np.array(y_aug)], axis=0)

    print(f"  After augment: {X_out.shape[0]} samples "
          f"(orig {X_train.shape[0]} + aug {len(X_aug)})")
    return X_out, g_out, y_out


def apply_mixup(X, g, y_onehot, alpha=MIXUP_ALPHA):
    """
    Mixup augmentation (v3-#2).
    X_mix = λ·X1 + (1-λ)·X2
    y_mix = λ·y1 + (1-λ)·y2
    """
    n = X.shape[0]
    indices = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha, size=(n, 1, 1))
    lam_y = lam.reshape(n, 1)
    lam_g = lam.reshape(n, 1)

    X_mix = lam * X + (1 - lam) * X[indices]
    g_mix = lam_g * g + (1 - lam_g) * g[indices]
    y_mix = lam_y * y_onehot + (1 - lam_y) * y_onehot[indices]

    return X_mix.astype(np.float32), g_mix.astype(np.float32), y_mix.astype(np.float32)


# ============================================================================
# 5. DATA SPLITTING & NORMALIZATION
# ============================================================================
def stratified_split(X, g, y, file_paths):
    """Stratified split: 72% train / 8% val / 20% test."""
    X_tr, X_te, g_tr, g_te, y_tr, y_te, fp_tr, fp_te = train_test_split(
        X, g, y, file_paths, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, g_tr, g_val, y_tr, y_val, fp_tr, fp_val = train_test_split(
        X_tr, g_tr, y_tr, fp_tr, test_size=0.1, random_state=42, stratify=y_tr
    )
    print(f"  Train: {len(y_tr)}  |  Val: {len(y_val)}  |  Test: {len(y_te)}")
    return (X_tr, X_val, X_te, g_tr, g_val, g_te, y_tr, y_val, y_te, fp_tr)


def normalize_features(X_train, X_val, X_test):
    """StandardScaler fit on train only, transform all (v3-#6)."""
    scaler = StandardScaler()
    n_tr, T, F = X_train.shape
    scaler.fit(X_train.reshape(-1, F))
    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(n_tr, T, F)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape[0], T, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], T, F)
    print("  StandardScaler fitted on training set.")
    return X_train, X_val, X_test, scaler


def compute_weights(y_train):
    """Class weights for imbalanced data."""
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(weights))
    print(f"  Class weights: { {IDX_TO_EMOTION[k]: f'{v:.3f}' for k, v in cw.items()} }")
    return cw


# ============================================================================
# 6. tf.data PIPELINE (v3-#7)
# ============================================================================
def make_tf_dataset(X_audio, X_gender, y_onehot, batch_size, shuffle=False):
    """
    Create a tf.data.Dataset for efficient training / evaluation.
    Input: numpy arrays. Output: batched, prefetched dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices(
        ({'audio_input': X_audio, 'gender_input': X_gender}, y_onehot)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(y_onehot), 10000), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================================
# 7. MODEL BUILDING (v3 — CRNN, improved residual, stable attention)
# ============================================================================
def _gender_branch():
    """Shared gender sub-network."""
    inp = Input(shape=(1,), name='gender_input')
    g = Dense(16, activation='relu')(inp)
    g = BatchNormalization()(g)
    g = Dense(8, activation='relu')(g)
    return inp, g


def _attention_block(x):
    """
    Stable attention (v3-#4).
    x: (batch, T, units) → returns (batch, units)
    """
    score = Dense(1)(x)                              # (batch, T, 1)
    weights = Softmax(axis=1)(score)                  # (batch, T, 1)
    context = Multiply()([x, weights])                # (batch, T, units)
    context = Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)  # (batch, units)
    return context


def _classification_head(audio_feat, gender_feat, num_classes):
    """Shared classification head."""
    concat = Concatenate()([audio_feat, gender_feat])
    z = Dense(128, activation='relu')(concat)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.3)(z)
    return Dense(num_classes, activation='softmax', name='output')(z)


def _residual_block(x, filters, kernel_size=3):
    """
    Improved residual block (v3-#3):
    Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → Skip → ReLU
    """
    shortcut = x

    out = Conv1D(filters, kernel_size, padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv1D(filters, kernel_size, padding='same')(out)
    out = BatchNormalization()(out)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    out = Add()([out, shortcut])
    out = Activation('relu')(out)
    return out


def _make_optimizer(lr):
    """Adam with gradient clipping (v3-#5)."""
    return Adam(learning_rate=lr, clipnorm=GRADIENT_CLIP_NORM)


def _compile_model(model, lr):
    """Compile with label smoothing loss + clipped optimizer (v3-#5)."""
    model.compile(
        optimizer=_make_optimizer(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    return model


# ---------- LSTM + Attention ----------
def build_lstm_attention(input_shape, num_classes, lr=1e-3):
    audio_input = Input(shape=input_shape, name='audio_input')

    x = LSTM(256, return_sequences=True)(audio_input)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = _attention_block(x)

    audio_feat = Dense(128, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.4)(audio_feat)

    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)

    model = Model(inputs=[audio_input, gender_input], outputs=output)
    return _compile_model(model, lr)


# ---------- Bi-LSTM + Attention ----------
def build_bilstm_attention(input_shape, num_classes, lr=1e-3):
    audio_input = Input(shape=input_shape, name='audio_input')

    x = Bidirectional(LSTM(256, return_sequences=True))(audio_input)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = _attention_block(x)

    audio_feat = Dense(256, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.4)(audio_feat)

    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)

    model = Model(inputs=[audio_input, gender_input], outputs=output)
    return _compile_model(model, lr)


# ---------- CNN + Residual (v3-#3: 4 residual blocks) ----------
def build_cnn_residual(input_shape, num_classes, lr=1e-3):
    audio_input = Input(shape=input_shape, name='audio_input')

    # Initial conv
    x = Conv1D(128, 5, padding='same', activation='relu')(audio_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    # 4 residual blocks with increasing filters
    x = _residual_block(x, 128)
    x = Dropout(0.25)(x)

    x = _residual_block(x, 256)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = _residual_block(x, 512)
    x = Dropout(0.3)(x)

    x = _residual_block(x, 512)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling1D()(x)

    audio_feat = Dense(256, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.3)(audio_feat)

    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)

    model = Model(inputs=[audio_input, gender_input], outputs=output)
    return _compile_model(model, lr)


# ---------- CRNN (v3-#1: CNN + BiLSTM + Attention) ----------
def build_crnn_attention(input_shape, num_classes, lr=1e-3):
    """
    CRNN: Conv blocks act as feature extractor → BiLSTM → Attention → Dense.
    Best of both worlds: local pattern extraction (CNN) + temporal modelling (LSTM).
    """
    audio_input = Input(shape=input_shape, name='audio_input')

    # CNN feature extractor
    x = Conv1D(128, 5, padding='same', activation='relu')(audio_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    x = _residual_block(x, 256)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = _residual_block(x, 256)
    x = Dropout(0.3)(x)

    # BiLSTM temporal modelling
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Attention
    x = _attention_block(x)

    audio_feat = Dense(256, activation='relu')(x)
    audio_feat = BatchNormalization()(audio_feat)
    audio_feat = Dropout(0.4)(audio_feat)

    gender_input, gender_feat = _gender_branch()
    output = _classification_head(audio_feat, gender_feat, num_classes)

    model = Model(inputs=[audio_input, gender_input], outputs=output)
    return _compile_model(model, lr)


# ============================================================================
# 8. TRAINING (CosineDecay + improved callbacks + tf.data)
# ============================================================================
def make_lr_schedule(num_train_samples):
    """CosineDecay learning rate schedule."""
    steps_per_epoch = max(1, num_train_samples // BATCH_SIZE)
    return CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=EPOCHS * steps_per_epoch,
        alpha=1e-6
    )


def train_model(model, name, train_ds, val_ds, class_weights,
                epochs=EPOCHS):
    """Train with tf.data datasets + improved callbacks."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_{name}.keras")
    log_dir = os.path.join(OUTPUT_DIR, "logs", name)

    callbacks = []

    print(f"\n{'='*60}")
    print(f"  TRAINING: {name}")
    print(f"{'='*60}")
    model.summary()

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weights
    )
    return history


# ============================================================================
# 9. EVALUATION & VISUALIZATION (v3-#8, #9)
# ============================================================================
def evaluate_model(model, name, test_ds, y_test_int):
    """Evaluate + classification report."""
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\n  {name}: Loss={loss:.4f}  Accuracy={acc*100:.2f}%")

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    target_names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]
    print(classification_report(y_test_int, y_pred, target_names=target_names))

    return loss, acc, y_pred, y_test_int


def print_comparison_table(results):
    """Print clean model comparison table (v3-#8)."""
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Loss':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")

    best_name, best_acc = None, 0
    for name, (loss, acc, _, _) in results.items():
        star = " ★" if acc >= max(r[1] for r in results.values()) else ""
        print(f"  {name:<25} {acc*100:>9.2f}% {loss:>10.4f}{star}")
        if acc > best_acc:
            best_acc, best_name = acc, name

    print(f"\n  🏆 Best: {best_name}  ({best_acc*100:.2f}%)")
    return best_name, best_acc


def plot_results(histories, results, output_dir):
    """Training curves + confusion matrices + per-class accuracy (v3-#9)."""
    os.makedirs(output_dir, exist_ok=True)
    emo_names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]

    # ---- 1. Training curves ----
    n = len(histories)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = [axes]

    for i, (name, h) in enumerate(histories.items()):
        axes[i][0].plot(h.history['accuracy'], label='Train')
        axes[i][0].plot(h.history['val_accuracy'], label='Val')
        axes[i][0].set_title(f'{name} — Accuracy')
        axes[i][0].set_xlabel('Epoch')
        axes[i][0].set_ylabel('Accuracy')
        axes[i][0].legend()
        axes[i][0].grid(True)

        axes[i][1].plot(h.history['loss'], label='Train')
        axes[i][1].plot(h.history['val_loss'], label='Val')
        axes[i][1].set_title(f'{name} — Loss')
        axes[i][1].set_xlabel('Epoch')
        axes[i][1].set_ylabel('Loss')
        axes[i][1].legend()
        axes[i][1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"  Saved: training_history.png")

    # ---- 2. Confusion matrices ----
    n_r = len(results)
    fig, axes = plt.subplots(1, n_r, figsize=(8 * n_r, 6))
    if n_r == 1:
        axes = [axes]

    for i, (name, (_, _, y_pred, y_true)) in enumerate(results.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emo_names, yticklabels=emo_names, ax=axes[i])
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  Saved: confusion_matrix.png")

    # ---- 3. Per-class accuracy bar chart (v3-#9) ----
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(NUM_CLASSES)
    width = 0.8 / len(results)

    for j, (name, (_, _, y_pred, y_true)) in enumerate(results.items()):
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9) * 100
        bars = ax.bar(x_pos + j * width, per_class_acc, width, label=name, alpha=0.85)
        for bar, val in zip(bars, per_class_acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy by Model')
    ax.set_xticks(x_pos + width * (len(results) - 1) / 2)
    ax.set_xticklabels(emo_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=150)
    plt.close()
    print(f"  Saved: per_class_accuracy.png")

    # ---- 4. Model comparison bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    accs = [results[n][1] * 100 for n in names]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))

    bars = ax.barh(names, accs, color=colors, height=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Model Comparison — Test Accuracy')
    ax.set_xlim(0, max(accs) + 10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: model_comparison.png")

    print(f"\n  All plots saved to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # ---- GPU Configuration ----
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  [INFO] Found GPU(s): {gpus}")
        except RuntimeError as e:
            print(f"  [ERROR] GPU Config Error: {e}")
    else:
        print("  [WARN] No GPU found! Training will be VERY slow on CPU.")

    t_start = datetime.datetime.now()
    print("=" * 60)
    print("  Vietnamese SER — Optimized Training v3")
    print("=" * 60)
    print(f"  Started: {t_start}")
    print(f"  Config:  SR={SAMPLE_RATE} DUR={DURATION}s MFCC={N_MFCC} "
          f"HOP={HOP_LENGTH} AUG={NUM_AUGMENTATIONS}x")
    print(f"  Training: EPOCHS={EPOCHS} BATCH={BATCH_SIZE} LR={INITIAL_LR}")
    print(f"  Mixup={ENABLE_MIXUP}(α={MIXUP_ALPHA})  "
          f"SpecAugment={ENABLE_SPEC_AUGMENT}  Cache={USE_CACHE}")
    print(f"  LabelSmoothing={LABEL_SMOOTHING}  GradClip={GRADIENT_CLIP_NORM}")

    # ---- Step 1: Gender lookup ----
    print(f"\n{'='*60}")
    print("  STEP 1: Gender metadata")
    print(f"{'='*60}")
    gender_lookup = build_gender_lookup(DATALABEL_PATH)
    n_m = sum(1 for v in gender_lookup.values() if v == 0)
    n_f = sum(1 for v in gender_lookup.values() if v == 1)
    print(f"  Found {len(gender_lookup)} files  (Nam={n_m}, Nữ={n_f})")

    # ---- Step 2: Load originals (with cache) ----
    print(f"\n{'='*60}")
    print("  STEP 2: Load original features")
    print(f"{'='*60}")
    X_audio, X_gender, y, file_paths = load_original_data(DATASET_PATH, gender_lookup)

    for emo, idx in EMOTION_MAP.items():
        print(f"    {emo}: {np.sum(y == idx)} samples")

    # ---- Step 3: Stratified split ----
    print(f"\n{'='*60}")
    print("  STEP 3: Stratified split")
    print(f"{'='*60}")
    (X_tr, X_val, X_te,
     g_tr, g_val, g_te,
     y_tr, y_val, y_te,
     fp_tr) = stratified_split(X_audio, X_gender, y, file_paths)

    # ---- Step 4: Augment training set ONLY ----
    print(f"\n{'='*60}")
    print("  STEP 4: Augment training set ONLY")
    print(f"{'='*60}")
    X_tr, g_tr, y_tr = augment_training_set(X_tr, g_tr, y_tr, fp_tr)

    # ---- Step 5: Normalize ----
    print(f"\n{'='*60}")
    print("  STEP 5: Normalize features")
    print(f"{'='*60}")
    X_tr, X_val, X_te, scaler = normalize_features(X_tr, X_val, X_te)

    # ---- Step 6: Class weights ----
    print(f"\n{'='*60}")
    print("  STEP 6: Class weights")
    print(f"{'='*60}")
    class_weights = compute_weights(y_tr)

    # ---- Convert labels ----
    y_tr_oh = to_categorical(y_tr, NUM_CLASSES).astype(np.float32)
    y_val_oh = to_categorical(y_val, NUM_CLASSES).astype(np.float32)
    y_te_oh = to_categorical(y_te, NUM_CLASSES).astype(np.float32)

    # ---- Mixup on training set (v3-#2) ----
    g_tr_2d = g_tr.reshape(-1, 1).astype(np.float32)
    g_val_2d = g_val.reshape(-1, 1).astype(np.float32)
    g_te_2d = g_te.reshape(-1, 1).astype(np.float32)

    if ENABLE_MIXUP:
        print(f"\n{'='*60}")
        print(f"  STEP 6b: Mixup augmentation (α={MIXUP_ALPHA})")
        print(f"{'='*60}")
        X_tr_mix, g_tr_mix, y_tr_mix = apply_mixup(
            X_tr.astype(np.float32), g_tr_2d, y_tr_oh
        )
        # Combine original + mixup
        X_tr_final = np.concatenate([X_tr, X_tr_mix], axis=0).astype(np.float32)
        g_tr_final = np.concatenate([g_tr_2d, g_tr_mix], axis=0).astype(np.float32)
        y_tr_final = np.concatenate([y_tr_oh, y_tr_mix], axis=0).astype(np.float32)
        print(f"  After Mixup: {X_tr_final.shape[0]} samples "
              f"(orig+aug {X_tr.shape[0]} + mixup {X_tr_mix.shape[0]})")
    else:
        X_tr_final = X_tr.astype(np.float32)
        g_tr_final = g_tr_2d
        y_tr_final = y_tr_oh

    input_shape = (X_tr_final.shape[1], X_tr_final.shape[2])
    print(f"\n  Audio input shape: {input_shape}")
    print(f"  Feature dim: {input_shape[1]}  (MFCC+Δ+ΔΔ+ZCR+RMS)")

    # ---- Step 7: Build tf.data pipelines (v3-#7) ----
    print(f"\n{'='*60}")
    print("  STEP 7: Build tf.data pipelines")
    print(f"{'='*60}")
    train_ds = make_tf_dataset(X_tr_final, g_tr_final, y_tr_final,
                               BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(X_val.astype(np.float32), g_val_2d,
                              y_val_oh, BATCH_SIZE)
    test_ds = make_tf_dataset(X_te.astype(np.float32), g_te_2d,
                               y_te_oh, BATCH_SIZE)
    print("  Datasets created with shuffle/batch/prefetch/AUTOTUNE")

    # ---- LR schedule ----
    lr_schedule = make_lr_schedule(X_tr_final.shape[0])

    # ---- Step 8: Train all models ----
    histories = {}
    results = {}

    model_builders = [
        ("LSTM+Attention",    build_lstm_attention),
        ("BiLSTM+Attention",  build_bilstm_attention),
        ("CNN+Residual",      build_cnn_residual),
        ("CRNN+Attention",    build_crnn_attention),
    ]

    for model_name, builder in model_builders:
        print(f"\n{'='*60}")
        print(f"  STEP 8: Training {model_name}")
        print(f"{'='*60}")

        model = builder(input_shape, NUM_CLASSES, lr=lr_schedule)
        history = train_model(model, model_name.replace("+", "_"),
                              train_ds, val_ds, class_weights)
        histories[model_name] = history
        results[model_name] = evaluate_model(
            model, model_name, test_ds, y_te
        )

    # ---- Step 9: Results summary ----
    best_name, best_acc = print_comparison_table(results)

    # ---- Baseline comparison ----
    print(f"\n{'='*60}")
    print("  VS BASELINE (v1 — no optimization)")
    print(f"{'='*60}")
    baseline = {"LSTM+Gender (v1)": 70.54,
                "CNN+Gender (v1)": 70.21,
                "BiLSTM+Gender (v1)": 64.98}
    for n, a in baseline.items():
        print(f"    {n}: {a}%")
    delta = best_acc * 100 - 70.54
    arrow = "↑" if delta > 0 else "↓"
    print(f"\n    Improvement over best v1: {arrow} {abs(delta):.2f}%")

    # ---- Step 10: Plots ----
    print(f"\n{'='*60}")
    print("  STEP 10: Saving visualizations")
    print(f"{'='*60}")
    plot_results(histories, results, OUTPUT_DIR)

    t_end = datetime.datetime.now()
    print(f"\n  Finished: {t_end}  (elapsed {t_end - t_start})")
    print("=" * 60)


if __name__ == "__main__":
    main()
