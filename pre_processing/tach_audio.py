from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import datetime

# ===== CẤU HÌNH =====
# Gets the directory where the script is located (e.g., E:\KHMT\N4K2\DATN\pre_processing)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Moves up to the parent directory (e.g., E:\KHMT\N4K2\DATN)
parent_dir = os.path.dirname(current_dir)
# Constructs the absolute path to the file
FILE_PATH = os.path.join(parent_dir, "separated", "htdemucs", "QBBp3", "vocals.wav")
OUTPUT_DIR = "raw_cuts_with_time_QBBp3"

MIN_SILENCE_LEN = 700   # (ms) Độ dài im lặng tối thiểu để ngắt câu
KEEP_SILENCE = 100      # (ms) Giữ lại một chút đầu đuôi cho tự nhiên
MIN_LENGTH = 2000       # (ms) Chỉ lấy file dài hơn 2 giây

# ===== SETUP FFMPEG =====
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

# ===== TẠO FOLDER =====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== LOAD AUDIO =====
print("Đang tải file âm thanh...")
sound = AudioSegment.from_file(FILE_PATH) 
sound = sound.set_channels(1).set_frame_rate(16000)

# ===== TÍNH TOÁN NGƯỠNG TỰ ĐỘNG =====
avg_db = sound.dBFS
SILENCE_THRESH = -45 # Bạn có thể giữ cứng hoặc để avg_db - 14 tùy ý
print(f"Độ to trung bình: {avg_db} dB | Ngưỡng cắt: {SILENCE_THRESH} dB")

# ===== BƯỚC QUAN TRỌNG: PHÁT HIỆN THỜI GIAN =====
print("Đang quét vị trí các câu thoại (Detecting)...")

# detect_nonsilent trả về một list các khoảng thời gian: [[start, end], [start, end], ...]
nonsilent_ranges = detect_nonsilent(
    sound,
    min_silence_len=MIN_SILENCE_LEN,
    silence_thresh=SILENCE_THRESH
)

print(f"Đã tìm thấy {len(nonsilent_ranges)} đoạn tiềm năng. Đang cắt và xuất file...")

count = 0
for i, (start_i, end_i) in enumerate(nonsilent_ranges):
    
    # 1. Mở rộng thêm khoảng lặng (Keep Silence) thủ công
    # Cần đảm bảo không bị âm (nhỏ hơn 0) hoặc vượt quá độ dài file
    start_time = max(0, start_i - KEEP_SILENCE)
    end_time = min(len(sound), end_i + KEEP_SILENCE)
    
    # 2. Tính độ dài
    duration = end_time - start_time
    
    # 3. Chỉ lấy file dài hơn mức quy định (2 giây)
    if duration >= MIN_LENGTH:
        chunk = sound[start_time:end_time]
        
        # 4. Tạo tên file chứa Timestamp
        # Đổi mili-giây sang định dạng Giờ-Phút-Giây
        time_str = str(datetime.timedelta(milliseconds=start_time))
        # time_str sẽ có dạng "0:05:23.400000". Làm gọn lại:
        clean_time = time_str.split(".")[0].replace(":", "h", 1).replace(":", "m") # -> 0h05m23
        
        filename = f"{clean_time}s_seg{i:04d}.wav" 
        # Ví dụ kết quả: 0h05m23s_seg0012.wav
        
        out_file = os.path.join(OUTPUT_DIR, filename)
        chunk.export(out_file, format="wav")
        count += 1

print(f"Hoàn tất! Đã xuất {count} file vào thư mục '{OUTPUT_DIR}'.")