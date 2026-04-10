import os, csv, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHAN TICH CHI TIET NGUON DU LIEU")
print("=" * 80)

# =========================================
# 1. PHAN TICH DataLabel CSV - NGUON PHIM TU THU THAP
# =========================================
print("\n\n" + "=" * 80)
print("1. PHAN TICH CHI TIET DataLabel/ CSV")
print("=" * 80)

dl = "DataLabel"
csvs = sorted([f for f in os.listdir(dl) if f.endswith(".csv")])

movie_stats = {}  # movie_name -> {emotions, count, characters, durations}
all_emotions = defaultdict(int)
all_chars = defaultdict(int)
total_dur = 0
total_rows = 0

for cf in csvs:
    try:
        with open(os.path.join(dl, cf), encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames
            for row in reader:
                total_rows += 1
                movie = row.get("Movie", row.get("movie", "")).strip()
                if not movie:
                    movie = cf.replace("metadata - dataset_metadata_", "").replace("metadata - ", "").replace(".csv", "")
                
                emo = row.get("Emotion", row.get("emotion", "")).strip().lower()
                char = row.get("Character", row.get("character", "")).strip()
                dur_str = row.get("Duration", row.get("Duration (s)", "0"))
                try:
                    dur = float(str(dur_str).replace(",", "."))
                except:
                    dur = 0
                
                if movie not in movie_stats:
                    movie_stats[movie] = {"emotions": defaultdict(int), "count": 0, "chars": defaultdict(int), "total_dur": 0, "csv_file": cf}
                
                movie_stats[movie]["count"] += 1
                if emo:
                    movie_stats[movie]["emotions"][emo] += 1
                    all_emotions[emo] += 1
                if char:
                    movie_stats[movie]["chars"][char] += 1
                    all_chars[char] += 1
                movie_stats[movie]["total_dur"] += dur
                total_dur += dur
    except Exception as e:
        print(f"  Loi doc {cf}: {e}")

# In thong ke theo phim
print(f"\nTong so CSV files: {len(csvs)}")
print(f"Tong so dong da gan nhan: {total_rows}")
print(f"Tong thoi luong (uoc tinh): {total_dur/60:.1f} phut ({total_dur/3600:.2f} gio)")
print(f"So phim/nguon duy nhat: {len(movie_stats)}")

print(f"\n--- THONG KE THEO PHIM (sap xep theo so mau) ---")
print(f"{'STT':>3} {'Ten phim (rut gon)':50s} {'So mau':>7} {'Dur(s)':>8} {'ANG':>5} {'ANX':>5} {'HAP':>5} {'NEU':>5} {'SAD':>5} {'Nam':>4} {'Nu':>4}")
print("-" * 120)

sorted_movies = sorted(movie_stats.items(), key=lambda x: -x[1]["count"])
for i, (movie, stats) in enumerate(sorted_movies, 1):
    name = movie[:50]
    e = stats["emotions"]
    ang = e.get("anger", 0) + e.get("ang", 0)
    anx = e.get("anxiety", 0) + e.get("anx", 0) + e.get("fear", 0)
    hap = e.get("happiness", 0) + e.get("happy", 0) + e.get("hap", 0)
    neu = e.get("neutral", 0) + e.get("neu", 0)
    sad = e.get("sadness", 0) + e.get("sad", 0)
    nam = stats["chars"].get("Nam", 0) + stats["chars"].get("nam", 0)
    nu = stats["chars"].get("Nữ", 0) + stats["chars"].get("nữ", 0) + stats["chars"].get("Nu", 0)
    print(f"{i:3d} {name:50s} {stats['count']:7d} {stats['total_dur']:8.1f} {ang:5d} {anx:5d} {hap:5d} {neu:5d} {sad:5d} {nam:4d} {nu:4d}")

print(f"\n--- TONG THE LOAI CAM XUC TRONG CSV ---")
for emo, cnt in sorted(all_emotions.items(), key=lambda x: -x[1]):
    print(f"  {emo:20s}: {cnt}")

print(f"\n--- GIOI TINH ---")
for ch, cnt in sorted(all_chars.items(), key=lambda x: -x[1]):
    print(f"  {ch:20s}: {cnt}")

# =========================================
# 2. PHAN TICH metadata/ - DU LIEU TRUOC GAN NHAN
# =========================================
print("\n\n" + "=" * 80)
print("2. PHAN TICH metadata/ (truoc gan nhan)")
print("=" * 80)

md = "metadata"
md_csvs = sorted([f for f in os.listdir(md) if f.endswith(".csv")])

md_total = 0
md_movies = {}

for cf in md_csvs:
    try:
        with open(os.path.join(md, cf), encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            rows = list(reader)
            cnt = len(rows)
            md_total += cnt
            
            name = cf.replace("dataset_metadata_full_", "").replace("dataset_metadata_", "").replace(".csv", "")
            md_movies[name] = cnt
    except:
        pass

print(f"Tong CSV metadata: {len(md_csvs)}")
print(f"Tong so doan truoc gan nhan: {md_total}")

print(f"\n{'Ten':30s} {'Tong doan':>10} {'Da chon (DataLabel)':>20} {'Ty le':>8}")
print("-" * 75)
for name, cnt in sorted(md_movies.items(), key=lambda x: -x[1]):
    # Tim so mau da gan nhan tuong ung
    labeled = 0
    for mn, ms in movie_stats.items():
        mn_short = mn[:20].upper().replace(" ", "")
        if name.upper().replace("P2","").replace("P3","") in mn_short or name.upper() in ms["csv_file"].upper():
            labeled += ms["count"]
    
    pct = f"{labeled/cnt*100:.1f}%" if cnt > 0 and labeled > 0 else "-"
    print(f"{name:30s} {cnt:10d} {labeled:20d} {pct:>8}")

# =========================================
# 3. PHAN TICH ViSEC
# =========================================
print("\n\n" + "=" * 80)
print("3. PHAN TICH ViSEC")
print("=" * 80)

visec_path = os.path.join(md, "dataset_metadata_ViSEC.csv")
if os.path.exists(visec_path):
    visec_emo = defaultdict(int)
    visec_count = 0
    visec_durs = []
    with open(visec_path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            visec_count += 1
            emo = row.get("Emotion", "").strip()
            visec_emo[emo] += 1
            try:
                dur = float(row.get("Duration (s)", "0"))
                visec_durs.append(dur)
            except:
                pass
    
    print(f"Tong so mau ViSEC: {visec_count}")
    print(f"Thoi luong trung binh: {sum(visec_durs)/len(visec_durs):.2f}s" if visec_durs else "")
    print(f"Thoi luong min: {min(visec_durs):.2f}s - max: {max(visec_durs):.2f}s" if visec_durs else "")
    print(f"Tong thoi luong: {sum(visec_durs)/60:.1f} phut")
    print(f"\nPhan bo nhan:")
    for emo, cnt in sorted(visec_emo.items(), key=lambda x: -x[1]):
        print(f"  {emo:10s}: {cnt:5d} ({cnt/visec_count*100:.1f}%)")

# =========================================
# 4. PHAN TICH VNEMOS
# =========================================
print("\n\n" + "=" * 80)
print("4. PHAN TICH VNEMOS")
print("=" * 80)

vnemos_path = os.path.join(md, "dataset_metadata_VNEMOS.csv")
if os.path.exists(vnemos_path):
    vnemos_emo = defaultdict(int)
    vnemos_count = 0
    vnemos_durs = []
    vnemos_sources = defaultdict(int)
    with open(vnemos_path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vnemos_count += 1
            emo = row.get("Emotion", "").strip()
            vnemos_emo[emo] += 1
            folder = row.get("Folder", "").strip()
            vnemos_sources[folder] += 1
            try:
                dur = float(row.get("Duration (s)", "0"))
                vnemos_durs.append(dur)
            except:
                pass
    
    print(f"Tong so mau VNEMOS: {vnemos_count}")
    print(f"Thoi luong trung binh: {sum(vnemos_durs)/len(vnemos_durs):.2f}s" if vnemos_durs else "")
    print(f"Thoi luong min: {min(vnemos_durs):.2f}s - max: {max(vnemos_durs):.2f}s" if vnemos_durs else "")
    print(f"Tong thoi luong: {sum(vnemos_durs)/60:.1f} phut")
    print(f"\nPhan bo nhan:")
    for emo, cnt in sorted(vnemos_emo.items(), key=lambda x: -x[1]):
        print(f"  {emo:15s}: {cnt:5d} ({cnt/vnemos_count*100:.1f}%)")
    print(f"\nPhan bo theo Folder/Nguon goc:")
    for src, cnt in sorted(vnemos_sources.items(), key=lambda x: -x[1]):
        print(f"  {src:15s}: {cnt}")

# =========================================
# 5. PHAN TICH DATASET_LABELED cuoi cung
# =========================================
print("\n\n" + "=" * 80)
print("5. DATASET_LABELED cuoi cung")
print("=" * 80)

base = "DATASET_LABELED"
for emo in ["ANG", "ANX", "HAP", "NEU", "SAD"]:
    p = os.path.join(base, emo)
    if os.path.exists(p):
        files = [f for f in os.listdir(p) if f.endswith(".wav")]
        sizes = [os.path.getsize(os.path.join(p, f)) for f in files]
        total_size_mb = sum(sizes) / 1024 / 1024
        avg_size_kb = (sum(sizes) / len(sizes)) / 1024 if sizes else 0
        print(f"  {emo}: {len(files):5d} files | {total_size_mb:.1f} MB | avg {avg_size_kb:.1f} KB/file")

# =========================================
# 6. PHAN TICH data_final - tong doan truoc loc
# =========================================
print("\n\n" + "=" * 80)
print("6. data_final/ - chi tiet so doan moi phim")
print("=" * 80)

df = "data_final"
if os.path.exists(df):
    df_total = 0
    for folder in sorted(os.listdir(df)):
        fp = os.path.join(df, folder)
        if os.path.isdir(fp):
            cnt = len([f for f in os.listdir(fp) if f.endswith(".wav")])
            df_total += cnt
            print(f"  {folder:45s}: {cnt:5d} doan")
    print(f"\n  TONG: {df_total} doan (truoc gan nhan)")

# =========================================
# 7. PHAN TICH audio_convert - audio goc
# =========================================
print("\n\n" + "=" * 80)
print("7. audio_convert/ - chi tiet file audio goc")
print("=" * 80)

ac = "audio_convert"
if os.path.exists(ac):
    wavs = sorted([f for f in os.listdir(ac) if f.endswith(".wav")])
    total_gb = 0
    for f in wavs:
        size_mb = os.path.getsize(os.path.join(ac, f)) / 1024 / 1024
        total_gb += size_mb
        # Uoc tinh thoi luong: 16kHz mono 16-bit = 32000 bytes/s
        dur_min = os.path.getsize(os.path.join(ac, f)) / 32000 / 60
        print(f"  {f[:55]:55s} {size_mb:8.1f} MB  ~{dur_min:.1f} phut")
    print(f"\n  TONG: {len(wavs)} files | {total_gb/1024:.2f} GB | ~{total_gb/32000*1024*1024/60:.0f} phut")

print("\n\nDONE!")
