from datasets import load_dataset
import soundfile as sf
import os

OUTPUT_DIR = "Bud500_1h"
TARGET_HOURS = 1

dataset = load_dataset(
    "linhtran92/viet_bud500",
    split="train",
    streaming=True
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_duration = 0
file_idx = 0

print("🚀 Start streaming...")

for i, sample in enumerate(dataset):
    if i % 10 == 0:
        print(f"Processing {i} | {total_duration/3600:.2f}h")

    # 🔥 decode tại đây (lazy)
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    duration = len(audio) / sr

    if total_duration >= TARGET_HOURS * 3600:
        break

    sf.write(f"{OUTPUT_DIR}/{file_idx}.wav", audio, sr)

    total_duration += duration
    file_idx += 1

print(f"\n✅ Done: {file_idx} files | {total_duration/3600:.2f} hours")