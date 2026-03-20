# 🎙️ Speech Emotion Recognition — Demo

Demo web nhận dạng cảm xúc giọng nói tiếng Việt sử dụng model đã train.

## Cách sử dụng

### 1. Chạy demo

```bash
# Tự động tìm model tốt nhất trong output_optimized/
cd /mnt/g/doan/giang/DATN/TrainModel
conda activate ser
python demo.py


# Đổi port (mặc định: 5000)
python demo.py --port 8080
```

### 2. Mở trình duyệt

```
http://localhost:5000
```

### 3. Sử dụng giao diện

1. **Upload file** — Kéo thả hoặc nhấn chọn file âm thanh (.wav, .mp3, .flac, .ogg, .m4a)
3. **Nhấn "Nhận dạng cảm xúc"** — Xem kết quả

### 4. Kết quả trả về

- **Cảm xúc dự đoán** với emoji tương ứng
- **Độ tin cậy** (confidence %)
- **Biểu đồ xác suất** cho tất cả 5 lớp cảm xúc:

| Mã | Cảm xúc | Emoji |
|----|---------|-------|
| ANG | Tức giận | 😠 |
| ANX | Lo lắng | 😰 |
| HAP | Vui vẻ | 😄 |
| NEU | Bình thường | 😐 |
| SAD | Buồn | 😢 |

