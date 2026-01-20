import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tensorflow.keras.utils import to_categorical 

# ===== CẤU HÌNH (QUAN TRỌNG NHẤT) =====

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "DATASET_LABELED"

print(f"🔍 Đang tìm dữ liệu tại: {DATASET_DIR}")

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def extract_mfcc(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """Hàm đọc file audio và biến nó thành các con số MFCC"""
    try:
        # Load audio
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Xử lý độ dài (Padding/Truncating)
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            padding = int(SAMPLES_PER_TRACK - len(signal)) # Ép kiểu int cho chắc
            signal = np.pad(signal, (0, padding), mode='constant')

        # Trích xuất MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Transpose để có shape (Time, n_mfcc)
        mfcc = mfcc.T 
        return mfcc
    except Exception as e:
        print(f"⚠️ Lỗi file {os.path.basename(file_path)}: {e}")
        return None

# ===== CHƯƠNG TRÌNH CHÍNH =====
print(f"🔍 Đang tìm dữ liệu tại: {DATASET_DIR}")

if not os.path.exists(DATASET_DIR):
    print("❌ LỖI TO: Không tìm thấy thư mục DATASET_LABELED!")
    print("👉 Bạn hãy kiểm tra lại đường dẫn trong phần CẤU HÌNH.")
    exit()

data = []
labels = []
files_count = 0

# Duyệt qua từng folder
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_DIR)):
    if dirpath == DATASET_DIR:
        continue
        
    label = os.path.basename(dirpath)
    # Bỏ qua các folder rác
    if label in ["TRASH", "Unlabeled"]: 
        continue

    print(f"📂 Đang xử lý nhãn: {label}...")

    for f in filenames:
        if f.lower().endswith(".wav"): # Kiểm tra đuôi file (chữ thường)
            file_path = os.path.join(dirpath, f)
            
            mfcc_features = extract_mfcc(file_path)
            
            if mfcc_features is not None:
                data.append(mfcc_features)
                labels.append(label)
                files_count += 1

# ===== KIỂM TRA DỮ LIỆU TRƯỚC KHI XỬ LÝ =====
if files_count == 0:
    print("\n❌ LỖI: Không tìm thấy bất kỳ file .wav nào!")
    print("👉 Hãy kiểm tra xem trong folder DATASET_LABELED có các folder con (ANG, SAD...) chưa?")
    exit()

# Chuyển sang dạng Numpy Array
X = np.array(data)
y = np.array(labels)

print("\n--- KẾT QUẢ ---")
print(f"✅ Tổng số mẫu dữ liệu: {len(X)}")
# Thêm kiểm tra len(X) > 0 để tránh lỗi Index Out of Bounds
if len(X) > 0:
    print(f"Kích thước 1 mẫu: {X[0].shape}") 

    # ===== LƯU DỮ LIỆU =====
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    np.save("classes.npy", le.classes_)
    np.save("X_data.npy", X)
    np.save("y_data.npy", y_onehot)

    print("💾 Đã lưu xong: X_data.npy, y_data.npy, classes.npy")
    print("🚀 Bạn đã sẵn sàng để Train Model!")
else:
    print("❌ Có lỗi xảy ra, mảng dữ liệu bị rỗng.")