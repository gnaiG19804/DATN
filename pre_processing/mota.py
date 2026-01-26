import os
import csv
import datetime
from pydub import AudioSegment

# ===== CẤU HÌNH =====
current_dir = os.path.dirname(os.path.abspath(__file__))
# Moves up to the parent directory (e.g., E:\KHMT\N4K2\DATN)
parent_dir = os.path.dirname(current_dir)

AUDIO_FOLDER = os.path.join(parent_dir, "data_final", "final_dataset_3s_5s_BDD")
OUTPUT_CSV = "dataset_metadata_full_BDD.csv"

# Thông tin cứng
MOVIE_NAME = "Biệt dược đen"
Link = "https://www.youtube.com/watch?v=XiPycGLPPGQ"

# ===== HÀM HỖ TRỢ XỬ LÝ THỜI GIAN =====
def parse_filename_time(filename):
    """
    Chuyển tên file '0h05m23s_seg...' thành số giây (float)
    Ví dụ: '0h01m30s' -> 90.0 giây
    """
    try:
        # Lấy phần thời gian: "0h05m23s"
        time_part = filename.split("_")[0]
        
        hours = int(time_part.split("h")[0])
        minutes = int(time_part.split("h")[1].split("m")[0])
        seconds = int(time_part.split("m")[1].split("s")[0])
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except:
        return 0.0

def format_seconds_to_hms(seconds):
    """Chuyển số giây thành dạng 00:05:23"""
    return str(datetime.timedelta(seconds=int(seconds)))

# ===== XỬ LÝ CHÍNH =====
headers = [
    "Filename", "Movie", "Link", 
    "Start Time", "End Time", "Duration (s)", 
    "Character", "Emotion"
]

data_rows = []

print("Đang quét và tính toán thời gian (sẽ mất vài giây)...")

files = sorted(os.listdir(AUDIO_FOLDER))

for i, filename in enumerate(files):
    if filename.endswith(".wav"):
        filepath = os.path.join(AUDIO_FOLDER, filename)
        
        # 1. Lấy Start Time từ tên file
        start_seconds = parse_filename_time(filename)
        
        # 2. Lấy độ dài (Duration) bằng cách đọc file audio
        try:
            audio = AudioSegment.from_file(filepath)
            duration_seconds = audio.duration_seconds
        except:
            duration_seconds = 0
            
        # 3. Tính End Time
        end_seconds = start_seconds + duration_seconds
        
        # 4. Format lại thành dạng 00:00:00 để dễ đọc
        start_str = format_seconds_to_hms(start_seconds)
        end_str = format_seconds_to_hms(end_seconds)
        
        # Tạo dòng dữ liệu
        row = [
            filename,
            MOVIE_NAME,
            Link,
            start_str,          # Cột Start
            end_str,            # Cột End (Mới thêm)
            round(duration_seconds, 2), # Cột độ dài (Mới thêm)
            "", "", ""          # Các cột trống để điền tay
        ]
        data_rows.append(row)
        
        # In tiến độ chạy cho đỡ sốt ruột
        if i % 50 == 0:
            print(f"Đã xử lý {i} file...")

# ===== GHI RA FILE CSV =====
with open(OUTPUT_CSV, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data_rows)

print(f"✅ Xong! File '{OUTPUT_CSV}' đã có đầy đủ Start và End time.")