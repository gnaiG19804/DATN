import os
import csv
from collections import defaultdict

# 1. Count files in DATASET_LABELED
print("=" * 60)
print("1. DATASET_LABELED distribution")
print("=" * 60)
base = "DATASET_LABELED"
total = 0
for emo in ["ANG", "ANX", "HAP", "NEU", "SAD"]:
    p = os.path.join(base, emo)
    if os.path.exists(p):
        count = len([f for f in os.listdir(p) if f.endswith(".wav")])
        total += count
        print(f"  {emo}: {count} files")
    else:
        print(f"  {emo}: FOLDER NOT FOUND")
print(f"  TOTAL: {total} files")

# 2. Count source audio files
print("\n" + "=" * 60)
print("2. Source audio files (audio_convert/)")
print("=" * 60)
ac = "audio_convert"
if os.path.exists(ac):
    wavs = [f for f in os.listdir(ac) if f.endswith(".wav")]
    print(f"  Total source files: {len(wavs)}")
    total_size = sum(os.path.getsize(os.path.join(ac, f)) for f in wavs)
    print(f"  Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    # List unique movie names
    print("\n  Source files:")
    for f in sorted(wavs):
        size_mb = os.path.getsize(os.path.join(ac, f)) / 1024 / 1024
        print(f"    {f[:60]:60s} {size_mb:.1f} MB")

# 3. Count data_final folders
print("\n" + "=" * 60)
print("3. Processed audio folders (data_final/)")
print("=" * 60)
df = "data_final"
if os.path.exists(df):
    folders = os.listdir(df)
    print(f"  Total folders: {len(folders)}")
    total_processed = 0
    for folder in sorted(folders):
        fp = os.path.join(df, folder)
        if os.path.isdir(fp):
            cnt = len([f for f in os.listdir(fp) if f.endswith(".wav")])
            total_processed += cnt
    print(f"  Total processed audio segments: {total_processed}")

# 4. Count SORTED folders
print("\n" + "=" * 60)
print("4. SORTED folders")
print("=" * 60)
sf = "SORTED"
if os.path.exists(sf):
    folders = os.listdir(sf)
    print(f"  Total SORTED folders: {len(folders)}")

# 5. Analyze CSV metadata
print("\n" + "=" * 60)
print("5. DataLabel CSV analysis")
print("=" * 60)
dl = "DataLabel"
if os.path.exists(dl):
    csvs = [f for f in os.listdir(dl) if f.endswith(".csv")]
    print(f"  Total CSV files: {len(csvs)}")
    
    emotion_counts = defaultdict(int)
    total_rows = 0
    movies = set()
    
    for cf in csvs:
        try:
            with open(os.path.join(dl, cf), encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    total_rows += 1
                    emo = row.get("Emotion", "").strip().lower()
                    if emo:
                        emotion_counts[emo] += 1
                    movie = row.get("Movie", "").strip()
                    if movie:
                        movies.add(movie[:50])
        except Exception as e:
            pass
    
    print(f"  Total labeled rows: {total_rows}")
    print(f"  Unique movies: {len(movies)}")
    print(f"\n  Emotion distribution in CSVs:")
    for emo, cnt in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"    {emo:20s}: {cnt}")

# 6. metadata folder
print("\n" + "=" * 60)
print("6. Metadata folder analysis")
print("=" * 60)
md = "metadata"
if os.path.exists(md):
    csvs = [f for f in os.listdir(md) if f.endswith(".csv")]
    print(f"  Total metadata CSV files: {len(csvs)}")
    
    total_meta_rows = 0
    for cf in csvs:
        try:
            with open(os.path.join(md, cf), encoding="utf-8-sig") as fh:
                reader = csv.reader(fh)
                rows = sum(1 for _ in reader) - 1  # minus header
                total_meta_rows += rows
        except:
            pass
    print(f"  Total metadata rows (before labeling): {total_meta_rows}")

# 7. DATASET_LABELED file duration stats
print("\n" + "=" * 60)
print("7. Sample filenames from DATASET_LABELED")
print("=" * 60)
for emo in ["ANG", "ANX", "HAP", "NEU", "SAD"]:
    p = os.path.join(base, emo)
    if os.path.exists(p):
        files = sorted([f for f in os.listdir(p) if f.endswith(".wav")])[:3]
        print(f"  {emo}: {files}")

print("\nDONE")
