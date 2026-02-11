from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
import torchaudio
import numpy as np
import webrtcvad
from speechbrain.inference.speaker import EncoderClassifier
from transformers import pipeline as hf_pipeline

# ================================
# GLOBAL CONFIG
# ================================

torch.set_grad_enabled(False)

WAVLM_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "TrainModel",
    "saved_wavlm_emotion_model"
)

# ================================
# FASTAPI
# ================================

app = FastAPI()

# ================================
# MODELS (LAZY LOAD)
# ================================

spk_model = None
emotion_model = None


def get_spk_model():
    global spk_model
    if spk_model is None:
        print("Loading speaker model...")
        spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        print("Speaker model ready")
    return spk_model


def get_emotion_model():
    global emotion_model
    if emotion_model is None:
        print("Loading emotion model...")
        emotion_model = hf_pipeline(
            "audio-classification",
            model=WAVLM_MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            top_k=5,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("Emotion model ready")
    return emotion_model


# ================================
# INPUT MODEL
# ================================

class AudioPath(BaseModel):
    path: str


# ================================
# AUDIO UTIL
# ================================

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
# SPEAKER — MULTI SIGNAL ENGINE
# ================================

vad = webrtcvad.Vad(2)


def speaker_similarity(wav):
    model = get_spk_model()
    L = wav.shape[1]

    if L < 32000:
        return 1.0

    a = model.encode_batch(wav[:, :int(L*0.7)])
    b = model.encode_batch(wav[:, int(L*0.3):])

    sim = torch.nn.functional.cosine_similarity(
        a.squeeze(),
        b.squeeze(),
        dim=0
    )
    return float(sim)


def vad_ratio(wav):
    x = (wav.squeeze().numpy()*32768).astype(np.int16)
    sr = 16000
    frame = int(sr*0.03)

    s = t = 0
    for i in range(0, len(x)-frame, frame):
        if vad.is_speech(x[i:i+frame].tobytes(), sr):
            s += 1
        t += 1

    return s/t if t else 0


def energy_var(wav):
    x = wav.squeeze().numpy()
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


# ================================
# EMOTION ENGINE
# ================================

LABEL_MAP = {
    "ANG": "anger",
    "ANX": "anxiety",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness"
}

CALIB = {
    "ANG":1.0,
    "ANX":1.8,
    "HAP":1.8,
    "NEU":0.7,
    "SAD":1.0
}


def calibrated_emotion(path):

    clf = get_emotion_model()
    raw = clf(path)

    adj = []
    for r in raw:
        w = CALIB.get(r["label"],1)
        adj.append({"label":r["label"], "score":r["score"]*w})

    tot = sum(r["score"] for r in adj)
    for r in adj:
        r["score"] /= tot

    adj.sort(key=lambda x:x["score"], reverse=True)

    top = adj[0]
    label = LABEL_MAP.get(top["label"], top["label"])

    return label, float(top["score"]), adj


# ================================
# API
# ================================

@app.post("/analyze-audio")
async def analyze_audio(data: AudioPath):

    path = data.path.replace("\\","/")

    if not os.path.exists(path):
        return {
            "pass": False,
            "should_delete": True,
            "reason": "file_not_found",
            "path": path
        }

    wav, sr = torchaudio.load(path)
    wav = to_mono_16k(wav, sr)
    wav = trim_silence(wav)

    ok, sim, vr, ev, sc = speaker_pass_decision(wav)

    if not ok:
        return {
            "pass": False,
            "should_delete": True,
            "reason": "multi_speaker",
            "path": path,
            "speaker": {
                "similarity": sim,
                "vad": vr,
                "energy_var": ev,
                "score": sc
            }
        }

    emo, conf, dist = calibrated_emotion(path)

    return {
        "pass": True,
        "should_delete": False,
        "emotion": emo,
        "confidence": conf,
        "path": path,
        "speaker": {
            "similarity": sim,
            "vad": vr,
            "energy_var": ev,
            "score": sc
        }
    }
@app.post("/delete-file")
async def delete_file(data: AudioPath):
    if os.path.exists(data.path):
        os.remove(data.path)
        return {"deleted":True}
    return {"deleted":False}
