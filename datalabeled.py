import os
import csv
import shutil

# ===== CẤU HÌNH (Bạn chỉnh lại cho đúng tên file/thư mục của bạn) =====
CSV_PATH = "DataLabel/PHD.csv"       # Tên file CSV bạn vừa sửa xong
SOURCE_FOLDER = "n8n-custom/data/PHD/final_3to5s"   # Thư mục chứa tất cả các file âm thanh lộn xộn
OUTPUT_DIR = "SORTED_BY_PHD"          # Thư mục đích (Code sẽ tự tạo)

# ===== XỬ LÝ =====
# 1. Tạo thư mục đích
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Đang đọc file '{CSV_PATH}' và phân loại...")

count_success = 0
count_missing = 0
count_skipped = 0

with open(CSV_PATH, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        filename = row["Filename"]
        emotion = row["Emotion"]
        
        # Kiểm tra dữ liệu đầu vào
        if not filename:
            continue
            
        # Nếu chưa gán nhãn (ô Emotion để trống), bỏ qua hoặc cho vào folder riêng
        if not emotion or emotion.strip() == "":
            emotion = "Unlabeled" 
            # count_skipped += 1
            # continue # Nếu muốn bỏ qua hẳn thì bỏ comment dòng này
            
        # Chuẩn hóa tên folder (bỏ khoảng trắng thừa, viết hoa chữ cái đầu)
        # Ví dụ: " anger " -> "Anger"
        emotion_folder_name = emotion.strip().capitalize()
        
        # Đường dẫn file gốc
        src_path = os.path.join(SOURCE_FOLDER, filename)
        
        # Đường dẫn đích (Tự tạo folder con theo tên cảm xúc)
        dest_folder = os.path.join(OUTPUT_DIR, emotion_folder_name)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
            
        dest_path = os.path.join(dest_folder, filename)
        
        # Thực hiện Copy
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path) # copy2 giữ nguyên ngày tháng tạo file
            count_success += 1
            # In ra cho vui mắt (tùy chọn)
            # print(f"✅ {filename} -> {emotion_folder_name}")
        else:
            print(f"⚠️ Lỗi: Không tìm thấy file gốc '{filename}'")
            count_missing += 1

print("-" * 30)
print(f"🎉 HOÀN TẤT!")
print(f"✅ Đã copy thành công: {count_success} file")
print(f"❌ Không tìm thấy file gốc: {count_missing} file")
print(f"📂 Dữ liệu đã được chia vào thư mục: '{OUTPUT_DIR}'")