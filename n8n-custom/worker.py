from flask import Flask, request, send_file
import subprocess, shutil, json, re, csv, math
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import datetime

app = Flask(__name__)

DATA = Path("/data")
TMP = DATA / "_demucs_tmp"
DATA.mkdir(exist_ok=True)

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
        "exists": (DATA/title).exists()
    }


# =========================
# DOWNLOAD
# =========================

@app.route("/download_audio", methods=["POST"])
def download_audio():

    url = request.json.get("url")

    meta = subprocess.check_output(["yt-dlp","--dump-json",url])
    info = json.loads(meta.decode())

    title = safe_name(info["title"])
    link = info["webpage_url"]

    wav_path = DATA / f"{title}.wav"

    if wav_path.exists():
        return {"status":"skip","title":title,"link":link}

    subprocess.run([
        "yt-dlp",
        "-x","--audio-format","wav",
        "-o", str(wav_path),
        url
    ])

    return {"status":"ok","title":title,"link":link}

# =========================
# DEMUCS FULL FILE
# =========================
@app.route("/separate_vocals", methods=["POST"])
def separate_vocals():

    title = request.json.get("title")

    wav_path = DATA / f"{title}.wav"
    out_dir = DATA / title
    vocals_out = out_dir / "vocals.wav"

    if vocals_out.exists():
        return {"status":"skip","title":title}

    out_dir.mkdir(parents=True, exist_ok=True)

    r = subprocess.run([
        "demucs",
        "--two-stems=vocals",
        "-n","htdemucs",
        "--device","cpu",
        "-o", str(TMP),
        str(wav_path)
    ])

    if r.returncode != 0:
        return {"status":"error","reason":"demucs failed"}

    model_dir = TMP / "htdemucs"
    vocals = list(model_dir.glob("**/vocals.wav"))

    if not vocals:
        return {"status":"error","reason":"no vocals found"}

    shutil.move(str(vocals[0]), str(vocals_out))
    shutil.rmtree(TMP, ignore_errors=True)

    return {"status":"ok","title":title}

# =========================
# SPLIT ON VOCALS
# =========================

@app.route("/split_segments", methods=["POST"])
def split_segments():

    title = request.json.get("title")

    vocals_path = DATA / title / "vocals.wav"
    seg_dir = DATA / title / "segments"


    if seg_dir.exists():
        return {"status":"skip","title":title}

    seg_dir.mkdir(exist_ok=True)

    sound = AudioSegment.from_file(vocals_path)
    sound = sound.set_channels(1).set_frame_rate(16000)

    ranges = detect_nonsilent(
        sound,
        min_silence_len=400,
        silence_thresh=sound.dBFS - 16
    )

    count = 0

    for i,(s,e) in enumerate(ranges):

        if e-s < 2000:
            continue

        chunk = sound[s:e]

        t = str(datetime.timedelta(milliseconds=s))
        clean = t.split(".")[0].replace(":", "h",1).replace(":", "m")

        chunk.export(seg_dir/f"{clean}s_seg{i:04d}.wav","wav")
        count += 1

    return {"status":"ok","segments":count,"title":title}

# =========================
# RESIZE
# =========================

@app.route("/resize_segments", methods=["POST"])
def resize_segments():

    title = request.json.get("title")

    seg_dir = DATA / title / "segments"
    out_dir = DATA / title / "final_3to5s"

    if out_dir.exists():
        return {"status":"skip","title":title}

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

    return {"status":"ok","title":title}

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

app.run(host="0.0.0.0", port=5000)
