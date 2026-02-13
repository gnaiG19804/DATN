from flask import Flask, request, send_file
import subprocess, shutil, json, re, csv, math
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import datetime
import torch

app = Flask(__name__)

# Use local 'data' directory relative to this script
DATA = Path(__file__).parent / "data"
TMP = DATA / "_demucs_tmp"
DATA.mkdir(parents=True, exist_ok=True)

# Detect Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Worker running on device: {DEVICE} ---")

# =========================
# UTIL
# =========================

def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:120]

def parse_filename_time(filename):
    try:
        t = filename.split("_")[0]
        h = int(t.split("h")[0])
        m = int(t.split("h")[1].split("m")[0])
        s = int(t.split("m")[1].split("s")[0])
        return h*3600 + m*60 + s
    except:
        return 0

def fmt_hms(sec):
    return str(datetime.timedelta(seconds=int(sec)))

# =========================
# CHECK EXISTS
# =========================

@app.route("/check_exists", methods=["POST"])
def check_exists():
    title = request.json.get("title")
    if not title:
        return {"status":"error","reason":"missing title"}

    return {
        "exists": (DATA/title).exists(),
        "path": str(DATA/title)
    }


# =========================
# DOWNLOAD
# =========================

@app.route("/download_audio", methods=["POST"])
def download_audio():

    url = request.json.get("url")
    if not url:
        return {"error": "missing url"}, 400

    base_cmd = [
        "yt-dlp",
        "--remote-components", "ejs:github",
        "--extractor-args", "youtube:player_client=android",
        "--cookies", "youtube_cookies.txt",
    ]

    # ---- lấy metadata an toàn ----
    meta_proc = subprocess.run(
        base_cmd + ["--dump-json", url],
        capture_output=True,
        text=True
    )

    if meta_proc.returncode != 0:
        return {
            "error": "yt-dlp meta failed",
            "detail": meta_proc.stderr
        }, 400

    info = json.loads(meta_proc.stdout)

    title = safe_name(info["title"])
    link = info["webpage_url"]

    wav_path = DATA / f"{title}.wav"

    if wav_path.exists():
        return {"status": "skip", "title": title, "link": link, "path": str(wav_path)}

    # ---- download audio ----
    dl_proc = subprocess.run(
        base_cmd + [
            "-x",
            "--audio-format", "wav",
            "-o", str(wav_path),
            url
        ],
        capture_output=True,
        text=True
    )

    if dl_proc.returncode != 0:
        return {
            "error": "yt-dlp download failed",
            "detail": dl_proc.stderr
        }, 400

    return {"status": "ok", "title": title, "link": link, "path": str(wav_path)}


# =========================
# DEMUCS FULL FILE
# =========================
@app.route("/separate_vocals", methods=["POST"])
def separate_vocals():
    title = request.json.get("title")
    if not title:
        return {"status": "error", "reason": "missing title"}, 400

    wav_path = DATA / f"{title}.wav"
    out_dir = DATA / title
    vocals_out = out_dir / "vocals.wav"

    # Kiểm tra đầu vào
    if not wav_path.exists():
        return {"status": "error", "reason": f"source file not found: {wav_path}"}, 404

    if vocals_out.exists():
        return {"status": "skip", "title": title, "vocals_path": str(vocals_out)}

    # Dọn dẹp thư mục tạm trước khi chạy để tránh lấy nhầm file cũ
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Demucs for: {title} on {DEVICE} ---")
    
    # Chạy Demucs
    try:
        # Lưu ý: Client cần set timeout rất dài hoặc dùng cơ chế async worker
        r = subprocess.run([
            "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs",
            "--device", DEVICE, 
            "-o", str(TMP),
            str(wav_path)
        ], check=True) # check=True sẽ raise lỗi nếu Demucs fail
    except subprocess.CalledProcessError as e:
        return {"status": "error", "reason": "demucs process failed", "detail": str(e)}, 500

    # Xác định chính xác file output dựa trên tên file gốc
    # Demucs output structure: /TMP/htdemucs/{original_filename_without_ext}/vocals.wav
    demucs_song_folder_name = wav_path.stem  # Tên file không đuôi .wav
    generated_vocals = TMP / "htdemucs" / demucs_song_folder_name / "vocals.wav"

    if not generated_vocals.exists():
        # Fallback: Thử tìm bằng glob nếu tên file bị Demucs thay đổi ký tự lạ
        found = list(TMP.glob("**/vocals.wav"))
        if found:
            generated_vocals = found[0]
        else:
            return {"status": "error", "reason": "Demucs finished but vocals file not found"}, 500

    # Di chuyển file
    shutil.move(str(generated_vocals), str(vocals_out))
    
    # Dọn dẹp
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"--- Finished Demucs: {vocals_out} ---")

    return {"status": "ok", "title": title, "vocals_path": str(vocals_out)}

# =========================
# SPLIT ON VOCALS (Đã sửa)
# =========================
@app.route("/split_segments", methods=["POST"])
def split_segments():
    title = request.json.get("title")
    if not title:
        return {"status": "error", "reason": "missing title"}, 400

    vocals_path = DATA / title / "vocals.wav"
    seg_dir = DATA / title / "segments"

    # KIỂM TRA QUAN TRỌNG: File có tồn tại không?
    if not vocals_path.exists():
        return {
            "status": "error", 
            "reason": "vocals.wav not found. Did you run /separate_vocals and wait for it to finish?"
        }, 404

    if seg_dir.exists():
         # Nếu muốn chạy lại thì có thể xóa seg_dir cũ ở đây
         return {"status": "skip", "title": title, "segments_dir": str(seg_dir)}

    seg_dir.mkdir(parents=True, exist_ok=True)

    try:
        sound = AudioSegment.from_file(vocals_path)
        sound = sound.set_channels(1).set_frame_rate(16000)

        # ... (giữ nguyên logic cắt file của bạn) ...
        
        ranges = detect_nonsilent(
            sound,
            min_silence_len=400,
            silence_thresh=sound.dBFS - 16
        )
        
        count = 0
        for i,(s,e) in enumerate(ranges):
            if e-s < 2000: continue
            chunk = sound[s:e]
            t = str(datetime.timedelta(milliseconds=s))
            clean = t.split(".")[0].replace(":", "h",1).replace(":", "m")
            chunk.export(seg_dir/f"{clean}s_seg{i:04d}.wav","wav")
            count += 1
            
        return {"status": "ok", "segments": count, "title": title, "segments_dir": str(seg_dir)}

    except Exception as e:
        return {"status": "error", "reason": str(e)}, 500

# =========================
# RESIZE
# =========================

@app.route("/resize_segments", methods=["POST"])
def resize_segments():

    title = request.json.get("title")
    
    seg_dir = DATA / title / "segments"
    out_dir = DATA / title / "final_3to5s"

    if out_dir.exists():
        return {"status":"skip","title":title, "output_dir": str(out_dir)}

    out_dir.mkdir(exist_ok=True)

    for f in seg_dir.glob("*.wav"):

        a = AudioSegment.from_file(f)
        d = len(a)

        if d < 3000:
            continue

        elif d <= 5000:
            a.export(out_dir/f.name,"wav")

        else:
            for i in range(math.ceil(d/4000)):
                s=i*4000
                e=min((i+1)*4000,d)
                if e-s>=2500:
                    a[s:e].export(out_dir/f"{f.stem}_part{i+1}.wav","wav")

    return {"status":"ok","title":title, "output_dir": str(out_dir)}

# =========================
# METADATA
# =========================

@app.route("/build_metadata", methods=["POST"])
def build_metadata():
    title = request.json.get("title")
    link = request.json.get("link", "")

    audio_dir = DATA / title / "final_3to5s"
    csv_path = DATA / title / "metadata.csv"

    rows = []

    for f in audio_dir.glob("*.wav"):
        s = parse_filename_time(f.name)
        dur = AudioSegment.from_file(f).duration_seconds

        rows.append([
            f.name,
            title,
            link,
            fmt_hms(s),
            fmt_hms(s + dur),
            round(dur, 2),
            "",
            ""
        ])

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "Filename","Movie","Link",
            "Start","End","Duration",
            "Character","Emotion"
        ])
        w.writerows(rows)

    # ✅ trả file csv
    return send_file(
        csv_path,
        as_attachment=True,
        download_name=f"{title}_metadata.csv",
        mimetype="text/csv"
    )

# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
