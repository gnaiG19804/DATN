import os
import math
from pydub import AudioSegment
from pydub.silence import split_on_silence

# ===== CẤU HÌNH =====
INPUT_FOLDER = "raw_cuts_with_time_HSCS"     
 # Folder hiện tại của bạn
OUTPUT_FOLDER = "final_dataset_3s_5s_HSCS"    # Folder chứa file thành phẩm

MIN_DURATION = 3000  # 3 giây
MAX_DURATION = 5000  # 5 giây
TARGET_SPLIT = 4000  

# ===== SETUP =====
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

files = sorted(os.listdir(INPUT_FOLDER))
print(f"Đang xử lý {len(files)} file...")

count_ok = 0
count_split = 0
count_short = 0

for filename in files:
    if not filename.endswith(".wav"):
        continue
        
    filepath = os.path.join(INPUT_FOLDER, filename)
    audio = AudioSegment.from_file(filepath)
    duration = len(audio)

    # TRƯỜNG HỢP 1: QUÁ NGẮN (< 3s) -> Bỏ qua
    if duration < MIN_DURATION:
        count_short += 1
        continue

    # TRƯỜNG HỢP 2: CHUẨN (3s - 5s) -> Copy sang luôn
    elif MIN_DURATION <= duration <= MAX_DURATION:
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        audio.export(out_path, format="wav")
        count_ok += 1

    # TRƯỜNG HỢP 3: QUÁ DÀI (> 5s) -> Cắt nhỏ tiếp
    else:
        # Ví dụ: File dài 13s -> Cần cắt thành các đoạn ~4s
        # Chiến thuật: Cắt thô theo thời gian (Fixed Slicing)
        # Vì nếu dùng split_on_silence ở đây có thể nó không tìm được chỗ cắt
        
        num_chunks = math.ceil(duration / TARGET_SPLIT) # 13s / 4s = 4 đoạn
        
        for i in range(num_chunks):
            start = i * TARGET_SPLIT
            end = min((i + 1) * TARGET_SPLIT, duration)
            
            # Kiểm tra đoạn cắt ra có quá ngắn không?
            # Ví dụ đoạn cuối cùng chỉ còn 1 giây -> Bỏ hoặc gộp (ở đây ta chọn bỏ cho sạch)
            chunk_len = end - start
            if chunk_len >= 2500: # Chỉ giữ lại nếu đoạn cắt ra > 2.5s
                chunk = audio[start:end]
                
                # Đặt tên mới: FileGoc_part1.wav
                new_filename = f"{filename[:-4]}_part{i+1}.wav"
                out_path = os.path.join(OUTPUT_FOLDER, new_filename)
                
                chunk.export(out_path, format="wav")
                count_split += 1

print("--- TỔNG KẾT ---")
print(f"✅ Giữ nguyên (Chuẩn): {count_ok} file")
print(f"✂️ Đã cắt nhỏ (Dài): {count_split} file mới")
print(f"🗑️ Đã loại bỏ (Ngắn): {count_short} file")
print(f"📂 Hãy vào folder '{OUTPUT_FOLDER}' để lấy dữ liệu chuẩn.")