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
# CONFIG
# ================================

torch.set_grad_enabled(False)

TARGET_SR = 16000
CHUNK_SEC = 20      # mỗi đoạn 20s
HOP_SEC = 15        # overlap để không mất tín hiệu

BASE_DIR = os.path.dirname(__file__)

WAVLM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "TrainModel",
    "saved_wavlm_emotion_model"
)

app = FastAPI()

spk_model = None
emotion_model = None


# ================================
# MODEL LOAD
# ================================

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
            top_k=5
        )
        print("Emotion model ready")
    return emotion_model


# ================================
# INPUT
# ================================

class AudioPath(BaseModel):
    path: str


# ================================
# AUDIO UTILS
# ================================

def to_mono_16k(wav, sr):

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    return wav


def split_audio_chunks(wav):

    chunk = TARGET_SR * CHUNK_SEC
    hop = TARGET_SR * HOP_SEC

    chunks = []

    for i in range(0, wav.shape[1], hop):

        part = wav[:, i:i+chunk]

        if part.shape[1] < TARGET_SR*5:
            continue

        chunks.append(part)

    return chunks


# ================================
# SPEAKER CHECK
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


def detect_emotion(wav):

    clf = get_emotion_model()

    raw = clf({
        "array": wav.squeeze().numpy(),
        "sampling_rate": TARGET_SR
    })

    raw.sort(key=lambda x: x["score"], reverse=True)

    top = raw[0]

    label = LABEL_MAP.get(top["label"], top["label"])

    return label, float(top["score"])


# ================================
# MAIN ANALYSIS
# ================================

@app.post("/analyze-audio")
async def analyze_audio(data: AudioPath):

    path = data.path.replace("\\","/")

    if not os.path.exists(path):
        return {"error": "file_not_found"}

    wav, sr = torchaudio.load(path)

    wav = to_mono_16k(wav, sr)

    chunks = split_audio_chunks(wav)

    emotions = []
    speakers = []

    for chunk in chunks:

        sim = speaker_similarity(chunk)

        emo, conf = detect_emotion(chunk)

        emotions.append((emo, conf))

        speakers.append(sim)

        del chunk

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # emotion voting
    emo_count = {}

    for e,_ in emotions:
        emo_count[e] = emo_count.get(e,0) + 1

    final_emotion = max(emo_count, key=emo_count.get)

    avg_speaker = float(np.mean(speakers))

    return {
        "emotion": final_emotion,
        "speaker_similarity": avg_speaker,
        "segments": len(chunks)
    }


# ================================
# DELETE FILE
# ================================

@app.post("/delete-file")
async def delete_file(data: AudioPath):

    if os.path.exists(data.path):
        os.remove(data.path)
        return {"deleted": True}

    return {"deleted": False}