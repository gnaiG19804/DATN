# =========================================================
# SER V3 - FIX OVERFITTING + FOCAL LOSS + SPECAUGMENT
# =========================================================

import os, glob, random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Fix MKL memory error
import numpy as np
import librosa
import tensorflow as tf

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# ================= GPU FIX =================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ================= CONFIG =================
DATASET_PATH = r"../DATASET_LABELED"

SAMPLE_RATE = 16000
WINDOW = 2.5
STEP = 1.25

N_MELS = 128
MAX_LEN = 150
MAX_SEGMENTS = 10

BATCH_SIZE = 8  # Giảm để tránh tràn RAM khi training
EPOCHS = 80
LR = 3e-4

EMOTION_MAP = {
    "ANG": 0,
    "ANX": 1,
    "HAP": 2,
    "NEU": 3,
    "SAD": 4
}

EMOTION_LABELS = list(EMOTION_MAP.keys())

# ================= FOCAL LOSS =================
def focal_loss(gamma=2.0, alpha=None):
    """Focal Loss - giúp model tập trung vào những mẫu khó phân loại"""
    def focal_loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        weight = y_true * K.pow(1.0 - y_pred, gamma)
        fl = weight * ce
        return K.sum(fl, axis=-1)
    return focal_loss_fn

# ================= AUDIO =================
def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y)
    return y

def split_audio(y):
    win = int(WINDOW * SAMPLE_RATE)
    step = int(STEP * SAMPLE_RATE)

    if len(y) < win:
        return [np.pad(y, (0, win - len(y)))]

    return [y[i:i+win] for i in range(0, len(y)-win+1, step)]

# ================= FEATURE =================
def extract_features(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel = librosa.power_to_db(mel)

    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch = np.nan_to_num(pitch)

    energy = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    min_len = min(mel.shape[1], delta.shape[1], delta2.shape[1],
                  len(pitch), len(energy), len(zcr))

    feat = np.concatenate([
        mel[:, :min_len],
        delta[:, :min_len],
        delta2[:, :min_len],
        pitch[:min_len][None, :],
        energy[:min_len][None, :],
        zcr[:min_len][None, :]
    ], axis=0).T

    if feat.shape[0] > MAX_LEN:
        feat = feat[:MAX_LEN]
    else:
        feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)))

    return feat

# ================= SPECAUGMENT =================
def spec_augment(feat, freq_mask=15, time_mask=20):
    """SpecAugment: mask ngẫu nhiên trên frequency và time axis"""
    feat = feat.copy()
    T, F = feat.shape

    # Frequency masking
    if random.random() < 0.5:
        f = random.randint(1, freq_mask)
        f0 = random.randint(0, max(0, F - f))
        feat[:, f0:f0+f] = 0

    # Time masking
    if random.random() < 0.5:
        t = random.randint(1, time_mask)
        t0 = random.randint(0, max(0, T - t))
        feat[t0:t0+t, :] = 0

    return feat

# ================= AUGMENT =================
def augment(y):
    """Audio augmentation nhẹ nhàng"""
    if random.random() < 0.5:
        y += np.random.uniform(0.003, 0.008) * np.random.randn(len(y))

    if random.random() < 0.4:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))

    if random.random() < 0.4:
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=random.uniform(-1.5, 1.5))

    return y

# ================= LOAD =================
def load_data(files, is_train=True):
    X, y, groups = [], [], []

    for f in files:
        label = EMOTION_MAP[os.path.basename(os.path.dirname(f))]

        # Giảm NEU vừa phải
        if is_train and label == EMOTION_MAP["NEU"] and random.random() < 0.35:
            continue

        audio = load_audio(f)
        segments = split_audio(audio)

        if len(segments) > MAX_SEGMENTS:
            segments = random.sample(segments, MAX_SEGMENTS)

        for seg in segments:
            feat = extract_features(seg)
            X.append(feat)
            y.append(label)
            groups.append(f)

            if is_train:
                # Augmentation repeat theo mức thiếu
                if label == EMOTION_MAP["ANX"]:
                    repeat = 3
                elif label in [EMOTION_MAP["HAP"], EMOTION_MAP["SAD"]]:
                    repeat = 2
                else:
                    repeat = 1

                for _ in range(repeat):
                    if random.random() < 0.6:
                        aug_audio = augment(seg)
                        aug_feat = extract_features(aug_audio)
                    else:
                        aug_feat = feat.copy()

                    # Luôn áp dụng SpecAugment
                    aug_feat = spec_augment(aug_feat)
                    X.append(aug_feat)
                    y.append(label)
                    groups.append(f)

    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)

# ================= MODEL =================
def build_model(input_shape):
    inp = Input(shape=(input_shape[0], input_shape[1], 1))

    # Block 1
    x = Conv2D(32, 3, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.2)(x)

    # Block 3
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    # Block 4
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    # GIẢM CHIỀU trước khi vào LSTM (tránh 6144 params)
    # Reshape: (batch, time_steps, features)
    shape = x.shape
    x = Reshape((shape[1], shape[2] * shape[3]))(x)

    # Dense projection để giảm chiều từ ~3000+ xuống 128
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.3)(x)

    # LSTM nhẹ hơn nhiều
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    out = Dense(len(EMOTION_MAP), activation='softmax')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=Adam(LR),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy']
    )

    return model

# ================= MIXUP (IN-PLACE, MEMORY-EFFICIENT) =================
def mixup_data_inplace(X, y, alpha=0.2):
    """Mixup in-place: trộn 2 mẫu trực tiếp trên X để tiết kiệm RAM"""
    batch_size = len(X)
    indices = np.random.permutation(batch_size)
    lam = np.random.beta(alpha, alpha, batch_size).astype(np.float32)
    lam_x = lam.reshape(-1, 1, 1, 1)
    lam_y = lam.reshape(-1, 1)

    # In-place: X = lam * X + (1-lam) * X[indices]
    for i in range(batch_size):
        X[i] = lam_x[i] * X[i] + (1 - lam_x[i]) * X[indices[i]]

    y[:] = lam_y * y + (1 - lam_y) * y[indices]

    return X, y

# ================= MAIN =================
def main():
    all_files = []
    for emo in EMOTION_MAP:
        all_files += glob.glob(os.path.join(DATASET_PATH, emo, "*.wav"))

    print(f"Total files: {len(all_files)}")
    for emo in EMOTION_MAP:
        count = sum(1 for f in all_files if os.path.basename(os.path.dirname(f)) == emo)
        print(f"  {emo}: {count}")

    labels = [EMOTION_MAP[os.path.basename(os.path.dirname(f))] for f in all_files]

    train_files, test_files = train_test_split(all_files, test_size=0.2, stratify=labels, random_state=42)
    train_labels = [EMOTION_MAP[os.path.basename(os.path.dirname(f))] for f in train_files]
    train_files, val_files = train_test_split(train_files, test_size=0.2, stratify=train_labels, random_state=42)

    print("\nLoading data...")
    X_train, y_train, g_train = load_data(train_files, True)
    X_val, y_val, g_val = load_data(val_files, False)
    X_test, y_test, g_test = load_data(test_files, False)

    print(f"\nTrain samples: {len(y_train)}")
    for i, emo in enumerate(EMOTION_LABELS):
        print(f"  {emo}: {np.sum(y_train == i)}")

    # Class weight balanced
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"\nClass weights: {class_weights}")

    # NORMALIZE
    scaler = StandardScaler()
    T, F = X_train.shape[1], X_train.shape[2]

    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(-1, T, F).astype(np.float32)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(-1, T, F).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(-1, T, F).astype(np.float32)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Mixup in-place trên training data (không tạo bản copy)
    y_train_cat = to_categorical(y_train).astype(np.float32)
    X_train, y_train_cat = mixup_data_inplace(X_train, y_train_cat, alpha=0.2)

    # Shuffle
    idx = np.random.permutation(len(X_train))
    X_train_final = X_train[idx]
    y_train_final = y_train_cat[idx]

    model = build_model((T, F))
    model.summary()

    print(f"\nTotal train (with mixup): {len(X_train_final)}")

    model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, to_categorical(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, min_lr=1e-7)
        ]
    )

    # ===== TEST =====
    probs = model.predict(X_test)

    file_probs = defaultdict(list)
    for p, g in zip(probs, g_test):
        file_probs[g].append(p)

    final_preds, final_labels = [], []

    for f in file_probs:
        p = np.array(file_probs[f])
        w = np.max(p, axis=1) ** 2
        pred = np.argmax(np.average(p, axis=0, weights=w))

        final_preds.append(pred)
        final_labels.append(EMOTION_MAP[os.path.basename(os.path.dirname(f))])

    print("\n🔥 FINAL RESULT:")
    print(classification_report(final_labels, final_preds, target_names=EMOTION_LABELS))
    print("\nConfusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))


if __name__ == "__main__":
    main()
