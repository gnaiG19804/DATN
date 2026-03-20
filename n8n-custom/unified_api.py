import os
import shutil
import json
import re
import csv
import math
import datetime
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import numpy as np
import webrtcvad
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from speechbrain.inference.speaker import EncoderClassifier
from transformers import pipeline as hf_pipeline

# ================================
# CONFIG & HARDWARE AUTO-DETECT
# ================================

app = FastAPI(title="Unified Audio Worker API")

# Setup Data Directory (Local)
BASE_DIR = Path(__file__).parent.absolute()
DATA = BASE_DIR / "data"
TMP = DATA / "_demucs_tmp"
DATA.mkdir(parents=True, exist_ok=True)

# Detect Device for Demucs & Torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on device: {DEVICE} ---")

torch.set_grad_enabled(False)

# Path to local emotion model
WAVLM_MODEL_PATH = BASE_DIR.parent / "TrainModel" / "model" / "saved_wavlm_emotion_model"

# ================================
# MODELS (LAZY LOAD)
# ================================

spk_model = None
emotion_model = None
vad = webrtcvad.Vad(2)

def get_spk_model():
    global spk_model
    if spk_model is None:
        print("Loading speaker model...")
        spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE}
        )
        print("Speaker model ready")
    return spk_model

def get_emotion_model():
    global emotion_model
    if emotion_model is None:
        print(f"Loading emotion model from {WAVLM_MODEL_PATH}...")
        emotion_model = hf_pipeline(
            "audio-classification",
            model=str(WAVLM_MODEL_PATH),
            device=0 if DEVICE == "cuda" else -1,
            top_k=5,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        print("Emotion model ready")
    return emotion_model

# ================================
# UTILS
# ================================

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

def to_mono_16k(wav, sr):
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav

def trim_silence(wav, thr=0.01):
    e = wav.abs()
    mask = e > thr
    if mask.any():
        idx = mask.nonzero()
        return wav[:, idx[0,1]:idx[-1,1]]
    return wav

# ================================
# SCHEMAS (Pydantic)
# ================================

class UrlRequest(BaseModel):
    url: str

class TitleRequest(BaseModel):
    title: str

class CheckRequest(BaseModel):
    title: str

class AudioPathRequest(BaseModel):
    path: str

class BuildMetaRequest(BaseModel):
    title: str
    link: Optional[str] = ""

# ================================
# WORKER ENDPOINTS (Originally Flask)
# ================================

@app.post("/check_exists")
async def check_exists(req: CheckRequest):
    p = DATA / req.title
    return {
        "exists": p.exists(),
        "path": str(p)
    }

@app.post("/download_audio")
async def download_audio(req: UrlRequest):
    url = req.url
    base_cmd = [
        "yt-dlp",
        "--remote-components", "ejs:github",
        "--extractor-args", "youtube:player_client=android",
        "--cookies", "youtube_cookies.txt",
    ]

    # Meta
    meta_proc = subprocess.run(
        base_cmd + ["--dump-json", url],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR) # run where youtube_cookies.txt is
    )

    if meta_proc.returncode != 0:
        print(f"ERROR: yt-dlp meta failed for URL: {url}")
        print(f"Stderr: {meta_proc.stderr}")
        raise HTTPException(status_code=400, detail=f"yt-dlp meta failed: {meta_proc.stderr}")

    info = json.loads(meta_proc.stdout)
    title = safe_name(info["title"])
    link = info["webpage_url"]

    wav_path = DATA / f"{title}.wav"

    if wav_path.exists():
        return {"status": "skip", "title": title, "link": link, "path": str(wav_path)}

    # Download
    dl_proc = subprocess.run(
        base_cmd + [
            "-x",
            "--audio-format", "wav",
            "-o", str(wav_path),
            url
        ],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )

    if dl_proc.returncode != 0:
        raise HTTPException(status_code=400, detail=f"yt-dlp download failed: {dl_proc.stderr}")

    return {"status": "ok", "title": title, "link": link, "path": str(wav_path)}

CHUNK_DURATION_SEC = 600   # 10 phút mỗi chunk
OVERLAP_SEC = 5            # 5 giây overlap giữa các chunk để tránh mất tiếng ở biên

def _get_audio_duration(wav_path: Path) -> float:
    """Lấy duration (giây) bằng ffprobe, không cần load toàn bộ file."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
            capture_output=True, text=True, check=True
        )
        return float(r.stdout.strip())
    except Exception:
        # Fallback: dùng torchaudio metadata
        info = torchaudio.info(str(wav_path))
        return info.num_frames / info.sample_rate

def _split_audio_ffmpeg(wav_path: Path, chunk_dir: Path,
                        chunk_sec: int, overlap_sec: int) -> list:
    """Cắt file audio thành các chunk nhỏ bằng ffmpeg (nhanh, không load RAM)."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    duration = _get_audio_duration(wav_path)

    chunks = []
    start = 0
    idx = 0
    step = chunk_sec - overlap_sec

    while start < duration:
        out_file = chunk_dir / f"chunk_{idx:04d}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ss", str(start),
            "-t", str(chunk_sec),
            "-ar", "44100",       # giữ SR gốc cho Demucs
            "-acodec", "pcm_s16le",
            str(out_file)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunks.append((out_file, start))
        start += step
        idx += 1

    return chunks

def _run_demucs_on_chunk(chunk_path: Path, demucs_out: Path, device: str):
    """Chạy Demucs trên 1 chunk, fallback CPU nếu GPU OOM."""
    demucs_cmd = [
        "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "--device", device,
        "--segment", "7",
        "--jobs", "1",
        "-o", str(demucs_out),
        str(chunk_path)
    ]
    try:
        subprocess.run(demucs_cmd, check=True)
    except subprocess.CalledProcessError:
        if device == "cuda":
            print(f"--- GPU OOM on {chunk_path.name}. Falling back to CPU ---")
            cpu_cmd = demucs_cmd.copy()
            try:
                cpu_cmd[cpu_cmd.index("cuda")] = "cpu"
            except ValueError:
                pass
            subprocess.run(cpu_cmd, check=True)
        else:
            raise

def _concat_vocals_with_crossfade(vocal_files: list, output_path: Path,
                                   overlap_sec: int):
    """Ghép các file vocals lại, crossfade ở vùng overlap."""
    from pydub import AudioSegment as AS

    overlap_ms = overlap_sec * 1000

    combined = AS.from_file(str(vocal_files[0]))

    for vf in vocal_files[1:]:
        next_seg = AS.from_file(str(vf))
        # crossfade ở vùng overlap
        fade = min(overlap_ms, len(combined), len(next_seg))
        combined = combined.append(next_seg, crossfade=fade)

    combined.export(str(output_path), format="wav")

@app.post("/separate_vocals")
async def separate_vocals(req: TitleRequest):
    title = req.title
    wav_path = DATA / f"{title}.wav"
    out_dir = DATA / title
    vocals_out = out_dir / "vocals.wav"

    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"source file not found: {wav_path}")

    if vocals_out.exists():
        return {"status": "skip", "title": title, "vocals_path": str(vocals_out)}

    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = _get_audio_duration(wav_path)
    print(f"--- Starting Demucs for: {title} on {DEVICE} (duration={duration:.0f}s) ---")

    try:
        if duration <= CHUNK_DURATION_SEC:
            # ---- File ngắn: xử lý trực tiếp như cũ ----
            _run_demucs_on_chunk(wav_path, TMP, DEVICE)

            demucs_song_folder_name = wav_path.stem
            generated_vocals = TMP / "htdemucs" / demucs_song_folder_name / "vocals.wav"

            if not generated_vocals.exists():
                found = list(TMP.glob("**/vocals.wav"))
                if found:
                    generated_vocals = found[0]
                else:
                    raise HTTPException(status_code=500,
                                        detail="Demucs finished but vocals file not found")

            shutil.move(str(generated_vocals), str(vocals_out))

        else:
            # ---- File dài: chia chunk → xử lý từng chunk → ghép lại ----
            chunk_dir = TMP / "_chunks"
            chunks = _split_audio_ffmpeg(wav_path, chunk_dir,
                                         CHUNK_DURATION_SEC, OVERLAP_SEC)
            print(f"--- Split into {len(chunks)} chunks ---")

            vocal_parts = []
            for i, (chunk_path, _start) in enumerate(chunks):
                print(f"--- Processing chunk {i+1}/{len(chunks)}: {chunk_path.name} ---")
                demucs_chunk_out = TMP / f"_demucs_chunk_{i:04d}"
                demucs_chunk_out.mkdir(parents=True, exist_ok=True)

                _run_demucs_on_chunk(chunk_path, demucs_chunk_out, DEVICE)

                # Tìm vocals output
                vocal_file = demucs_chunk_out / "htdemucs" / chunk_path.stem / "vocals.wav"
                if not vocal_file.exists():
                    found = list(demucs_chunk_out.glob("**/vocals.wav"))
                    if found:
                        vocal_file = found[0]
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Demucs chunk {i} finished but vocals not found")

                vocal_parts.append(vocal_file)

                # Xoá chunk input để tiết kiệm ổ đĩa
                chunk_path.unlink(missing_ok=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Ghép tất cả vocal parts lại
            print(f"--- Concatenating {len(vocal_parts)} vocal parts ---")
            _concat_vocals_with_crossfade(vocal_parts, vocals_out, OVERLAP_SEC)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"demucs process failed: {str(e)}")

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"--- Finished Demucs: {vocals_out} ---")

    return {"status": "ok", "title": title, "vocals_path": str(vocals_out)}

@app.post("/split_segments")
async def split_segments(req: TitleRequest):
    title = req.title
    vocals_path = DATA / title / "vocals.wav"
    seg_dir = DATA / title / "segments"

    if not vocals_path.exists():
        raise HTTPException(status_code=404, detail="vocals.wav not found")

    if seg_dir.exists():
        return {"status": "skip", "title": title, "segments_dir": str(seg_dir)}

    seg_dir.mkdir(parents=True, exist_ok=True)

    try:
        sound = AudioSegment.from_file(vocals_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        
        ranges = detect_nonsilent(
            sound,
            min_silence_len=400,
            silence_thresh=sound.dBFS - 16
        )
        
        count = 0
        for i, (s, e) in enumerate(ranges):
            if e - s < 2000: continue
            chunk = sound[s:e]
            t = str(datetime.timedelta(milliseconds=s))
            clean = t.split(".")[0].replace(":", "h", 1).replace(":", "m")
            chunk.export(seg_dir / f"{clean}s_seg{i:04d}.wav", "wav")
            count += 1
            
        return {"status": "ok", "segments": count, "title": title, "segments_dir": str(seg_dir)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resize_segments")
async def resize_segments(req: TitleRequest):
    title = req.title
    seg_dir = DATA / title / "segments"
    out_dir = DATA / title / "final_3to5s"

    if out_dir.exists():
        return {"status": "skip", "title": title, "output_dir": str(out_dir)}

    out_dir.mkdir(exist_ok=True)

    for f in seg_dir.glob("*.wav"):
        try:
            a = AudioSegment.from_file(f)
            d = len(a)

            if d < 3000:
                continue
            elif d <= 5000:
                a.export(out_dir / f.name, "wav")
            else:
                for i in range(math.ceil(d / 4000)):
                    s = i * 4000
                    e = min((i + 1) * 4000, d)
                    if e - s >= 2500:
                        a[s:e].export(out_dir / f"{f.stem}_part{i+1}.wav", "wav")
        except Exception as e:
            print(f"Error checking file {f}: {e}")

    return {"status": "ok", "title": title, "output_dir": str(out_dir)}

@app.post("/build_metadata")
async def build_metadata(req: BuildMetaRequest):
    title = req.title
    link = req.link
    audio_dir = DATA / title / "final_3to5s"
    csv_path = DATA / title / "metadata.csv"

    rows = []
    if audio_dir.exists():
        for f in audio_dir.glob("*.wav"):
            try:
                s = parse_filename_time(f.name)
                dur = AudioSegment.from_file(f).duration_seconds
                rows.append([
                    f.name, title, link,
                    fmt_hms(s), fmt_hms(s + dur),
                    round(dur, 2), "", ""
                ])
            except Exception as e:
                print(f"Error processing {f}: {e}")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "Filename", "Movie", "Link",
            "Start", "End", "Duration",
            "Character", "Emotion"
        ])
        w.writerows(rows)

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"{title}.csv"
    )

# ================================
# SPEAKER & EMOTION ANALYSIS (Originally from speaker_pass_api.py)
# ================================

LABEL_MAP = {
    "ANG": "anger",
    "ANX": "anxiety",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness"
}

CALIB = {
    "ANG": 1.0,
    "ANX": 1.8,
    "HAP": 1.8,
    "NEU": 0.7,
    "SAD": 1.0
}

def speaker_similarity(wav):
    model = get_spk_model()
    L = wav.shape[1]
    if L < 32000:
        return 1.0
    
    # Move tensor to device if necessary (speechbrain usually handles this inside encode_batch if model is on device)
    if DEVICE == "cuda":
        wav = wav.to(DEVICE)

    a = model.encode_batch(wav[:, :int(L*0.7)])
    b = model.encode_batch(wav[:, int(L*0.3):])
    
    sim = torch.nn.functional.cosine_similarity(
        a.squeeze(),
        b.squeeze(),
        dim=0
    )
    return float(sim)

def vad_ratio(wav):
    # VAD requires CPU numpy/bytes
    x = (wav.cpu().squeeze().numpy()*32768).astype(np.int16)
    sr = 16000
    frame = int(sr*0.03)

    s = t = 0
    for i in range(0, len(x)-frame, frame):
        if vad.is_speech(x[i:i+frame].tobytes(), sr):
            s += 1
        t += 1

    return s/t if t else 0

def energy_var(wav):
    x = wav.cpu().squeeze().numpy()
    frame = 400
    hop = 200
    e = [np.mean(np.abs(x[i:i+frame])) for i in range(0,len(x)-frame,hop)]
    return float(np.std(e)) if len(e)>1 else 0

def speaker_pass_decision(wav):
    sim = speaker_similarity(wav)
    vr = vad_ratio(wav)
    ev = energy_var(wav)

    if sim < 0.58:
        return False, sim, vr, ev, 0

    score = 0
    if sim > .80: score += 4
    elif sim > .70: score += 3
    elif sim > .62: score += 2
    else: score += 1

    if vr > .6: score += 1
    if ev < .075: score += 1

    return score >= 4, sim, vr, ev, score

def calibrated_emotion(path):
    clf = get_emotion_model()
    # Pipeline handles file reading, but if we need manual control we can load audio.
    # Pipeline 'audio-classification' accepts path string.
    raw = clf(path)

    adj = []
    for r in raw:
        w = CALIB.get(r["label"], 1)
        adj.append({"label": r["label"], "score": r["score"] * w})

    tot = sum(r["score"] for r in adj)
    for r in adj:
        r["score"] /= tot

    adj.sort(key=lambda x: x["score"], reverse=True)
    top = adj[0]
    label = LABEL_MAP.get(top["label"], top["label"])

    return label, float(top["score"]), adj

@app.post("/analyze-audio")
async def analyze_audio(req: AudioPathRequest):
    # Path handling: allow raw path or try to resolve if relative
    path_str = req.path.replace("\\", "/")
    p = Path(path_str)
    
    if not p.is_absolute():
        p = DATA / p
    
    if not p.exists():
        # Fallback: check if it's just a filename in DATA or subfolder
        found = list(DATA.rglob(p.name))
        if found:
            p = found[0]
        else:
             return {
                "pass": False,
                "should_delete": True,
                "reason": "file_not_found",
                "path": str(p)
            }

    try:
        if not p.is_file():
            return {
                "pass": False,
                "should_delete": False,
                "reason": "path_is_directory",
                "path": str(p)
            }

        wav, sr = torchaudio.load(str(p))
        wav = to_mono_16k(wav, sr)
        wav = trim_silence(wav)

        ok, sim, vr, ev, sc = speaker_pass_decision(wav)

        if not ok:
            try:
                if p.exists():
                    os.remove(p)
                    print(f"Deleted non-pass file: {p}")
            except Exception as delete_err:
                print(f"Failed to delete non-pass file {p}: {delete_err}")

            return {
                "pass": False,
                "should_delete": True,
                "reason": "multi_speaker",
                "path": str(p),
                "file_deleted": True,
                "speaker": {
                    "similarity": sim,
                    "vad": vr,
                    "energy_var": ev,
                    "score": sc
                }
            }

        emo, conf, dist = calibrated_emotion(str(p))

        return {
            "pass": True,
            "should_delete": False,
            "emotion": emo,
            "confidence": conf,
            "path": str(p),
            "speaker": {
                "similarity": sim,
                "vad": vr,
                "energy_var": ev,
                "score": sc
            }
        }
    except Exception as e:
        print(f"Analysis Failed: {e}")
        return {
            "pass": False,
            "should_delete": False,
            "reason": f"error: {str(e)}",
            "path": str(p)
        }

@app.post("/delete-file")
async def delete_file(req: AudioPathRequest):
    p = Path(req.path)
    if not p.is_absolute():
        p = DATA / p

    if p.exists():
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                os.remove(p)
            return {"deleted": True}
        except Exception as e:
            return {"deleted": False, "reason": str(e)}
    return {"deleted": False, "reason": "not_found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
