"""
Vietnamese Speech Emotion Recognition — Demo
=============================================
Web demo sử dụng model SER v5.1 (74.8% accuracy).
Upload file .wav để dự đoán cảm xúc.

Usage:
    python demo.py
    python demo.py --model best_model.keras --port 5000
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import librosa
import joblib
import tensorflow as tf

# Fix GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from flask import Flask, render_template_string, request, jsonify

# ============================================================================
# CONFIG (phải khớp với train.py)
# ============================================================================
SAMPLE_RATE = 16000
DURATION = 4.0
HOP_LENGTH = 512

EMOTIONS = {0: "ANG", 1: "ANX", 2: "HAP", 3: "NEU", 4: "SAD"}
EMOTION_VI = {
    "ANG": "Tức giận 😠",
    "ANX": "Lo lắng 😰",
    "HAP": "Vui vẻ 😄",
    "NEU": "Bình thường 😐",
    "SAD": "Buồn 😢"
}
EMOTION_COLORS = {
    "ANG": "#e74c3c",
    "ANX": "#e67e22",
    "HAP": "#2ecc71",
    "NEU": "#3498db",
    "SAD": "#9b59b6"
}


# ============================================================================
# FEATURE EXTRACTION (khớp 100% với train.py — 103 dims)
# ============================================================================
def pad_audio(y):
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) > target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


def extract_features(y):
    """Trích xuất 103 chiều đặc trưng — giống hệt train.py."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=40, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel)

    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0)

    if np.std(f0) > 0:
        f0 = (f0 - np.mean(f0)) / (np.std(f0) + 1e-6)

    f0 = f0.reshape(1, -1)

    min_len = min(
        zcr.shape[1], rms.shape[1], mfcc.shape[1],
        mel_db.shape[1], f0.shape[1]
    )

    zcr = zcr[:, :min_len]
    rms = rms[:, :min_len]
    mfcc = mfcc[:, :min_len]
    delta = delta[:, :min_len]
    delta2 = delta2[:, :min_len]
    mel_db = mel_db[:, :min_len]
    f0 = f0[:, :min_len]

    features = np.concatenate([
        zcr, rms,
        mfcc, delta, delta2,
        mel_db,
        f0
    ], axis=0)

    return features.T  # (time_steps, 103)


# ============================================================================
# MODEL LOADING & PREDICTION
# ============================================================================
def load_model(model_path):
    """Load model .keras trực tiếp."""
    print(f"  Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"  ✅ Model loaded — params: {model.count_params():,}")
    return model


def load_scaler(scaler_path):
    """Load scaler .pkl."""
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"  ✅ Scaler loaded: {scaler_path}")
        return scaler
    else:
        print(f"  ⚠️ Scaler not found: {scaler_path} — sử dụng normalize thủ công")
        return None


def predict_emotion(model, scaler, audio_path):
    """Dự đoán cảm xúc từ file audio."""
    # Load & preprocess
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = pad_audio(y)

    # Extract features
    features = extract_features(y)  # (T, 103)

    # Normalize
    T, F = features.shape
    if scaler is not None:
        features = scaler.transform(features.reshape(-1, F)).reshape(T, F)
    else:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

    # Predict
    audio_input = np.expand_dims(features, axis=0)  # (1, T, 103)
    probs = model.predict(audio_input, verbose=0)[0]

    pred_idx = np.argmax(probs)
    emotion = EMOTIONS[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs = {EMOTIONS[i]: float(probs[i]) for i in range(len(probs))}

    return emotion, confidence, all_probs


# ============================================================================
# WEB UI
# ============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SER Demo — Nhận Dạng Cảm Xúc Giọng Nói</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            color: #e0e0e0;
        }

        .container {
            max-width: 700px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        h1 {
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .subtitle {
            text-align: center;
            color: #8888aa;
            font-size: 0.9rem;
            margin-bottom: 32px;
        }

        .card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }

        .card-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #667eea;
            margin-bottom: 16px;
        }

        .upload-area {
            border: 2px dashed rgba(102, 126, 234, 0.4);
            border-radius: 12px;
            padding: 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
        .upload-area.dragover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }
        .upload-icon { font-size: 2.5rem; margin-bottom: 8px; }
        .upload-text { font-size: 0.95rem; color: #aaa; }
        .upload-text strong { color: #667eea; }
        input[type="file"] { display: none; }

        .file-info {
            margin-top: 12px;
            padding: 10px 16px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            font-size: 0.85rem;
            color: #667eea;
            display: none;
        }

        .btn-predict {
            display: block;
            width: 100%;
            padding: 14px;
            margin-top: 16px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(102,126,234,0.4); }
        .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

        #result-card { display: none; }
        .emotion-main {
            text-align: center;
            padding: 24px 0;
        }
        .emotion-emoji { font-size: 4rem; margin-bottom: 8px; }
        .emotion-label {
            font-size: 1.5rem;
            font-weight: 700;
        }
        .emotion-conf {
            font-size: 1rem;
            color: #aaa;
            margin-top: 4px;
        }

        .prob-bars { margin-top: 20px; }
        .prob-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            gap: 12px;
        }
        .prob-label {
            width: 40px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #ccc;
        }
        .prob-bar-bg {
            flex: 1;
            height: 24px;
            background: rgba(255,255,255,0.06);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.8s ease;
            display: flex;
            align-items: center;
            padding-left: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
        }

        .loading { display: none; text-align: center; padding: 20px; }
        .spinner {
            width: 40px; height: 40px;
            border: 3px solid rgba(102,126,234,0.2);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .model-info {
            text-align: center;
            color: #666;
            font-size: 0.75rem;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Speech Emotion Recognition</h1>
        <p class="subtitle">Nhận dạng cảm xúc giọng nói tiếng Việt</p>

        <div class="card">
            <div class="card-title">📁 Tải file âm thanh</div>
            <div class="upload-area" id="dropZone" onclick="document.getElementById('audioFile').click()">
                <div class="upload-icon">📂</div>
                <div class="upload-text">Kéo thả file .wav vào đây hoặc <strong>nhấn để chọn</strong></div>
            </div>
            <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.ogg,.m4a">
            <div class="file-info" id="fileInfo"></div>

            <button class="btn-predict" id="btnPredict" disabled onclick="predictEmotion()">
                🔍 Nhận dạng cảm xúc
            </button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Đang phân tích...</div>
        </div>

        <div class="card" id="result-card">
            <div class="card-title">📊 Kết quả</div>
            <div class="emotion-main">
                <div class="emotion-emoji" id="resultEmoji"></div>
                <div class="emotion-label" id="resultLabel"></div>
                <div class="emotion-conf" id="resultConf"></div>
            </div>
            <div class="prob-bars" id="probBars"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('audioFile');
        const fileInfo = document.getElementById('fileInfo');

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                showFileInfo(selectedFile.name, (selectedFile.size / 1024).toFixed(1) + ' KB');
            }
        });

        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                selectedFile = e.dataTransfer.files[0];
                fileInput.files = e.dataTransfer.files;
                showFileInfo(selectedFile.name, (selectedFile.size / 1024).toFixed(1) + ' KB');
            }
        });

        function showFileInfo(name, size) {
            fileInfo.textContent = `✅ ${name} (${size})`;
            fileInfo.style.display = 'block';
            document.getElementById('btnPredict').disabled = false;
        }

        async function predictEmotion() {
            if (!selectedFile) return;

            const loading = document.getElementById('loading');
            const resultCard = document.getElementById('result-card');
            const btnPredict = document.getElementById('btnPredict');

            loading.style.display = 'block';
            resultCard.style.display = 'none';
            btnPredict.disabled = true;

            const formData = new FormData();
            formData.append('audio', selectedFile);

            try {
                const resp = await fetch('/predict', { method: 'POST', body: formData });
                const data = await resp.json();

                if (data.error) {
                    alert('Lỗi: ' + data.error);
                } else {
                    showResult(data);
                }
            } catch (err) {
                alert('Lỗi kết nối: ' + err.message);
            } finally {
                loading.style.display = 'none';
                btnPredict.disabled = false;
            }
        }

        const EMOJI_MAP = { ANG: '😠', ANX: '😰', HAP: '😄', NEU: '😐', SAD: '😢' };
        const NAME_MAP = { ANG: 'Tức giận', ANX: 'Lo lắng', HAP: 'Vui vẻ', NEU: 'Bình thường', SAD: 'Buồn' };
        const COLOR_MAP = { ANG: '#e74c3c', ANX: '#e67e22', HAP: '#2ecc71', NEU: '#3498db', SAD: '#9b59b6' };

        function showResult(data) {
            const resultCard = document.getElementById('result-card');
            document.getElementById('resultEmoji').textContent = EMOJI_MAP[data.emotion] || '❓';
            document.getElementById('resultLabel').textContent = NAME_MAP[data.emotion] || data.emotion;
            document.getElementById('resultLabel').style.color = COLOR_MAP[data.emotion] || '#fff';
            document.getElementById('resultConf').textContent = `Độ tin cậy: ${(data.confidence * 100).toFixed(1)}%`;

            const barsDiv = document.getElementById('probBars');
            barsDiv.innerHTML = '';

            const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
            for (const [label, prob] of sorted) {
                const pct = (prob * 100).toFixed(1);
                barsDiv.innerHTML += `
                    <div class="prob-row">
                        <span class="prob-label">${label}</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:${pct}%;background:${COLOR_MAP[label]}">
                                ${pct}%
                            </div>
                        </div>
                    </div>`;
            }

            resultCard.style.display = 'block';
            resultCard.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""

# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
model = None
scaler = None
model_name = ""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, model_name=model_name)


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        emotion, confidence, all_probs = predict_emotion(model, scaler, tmp_path)
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': all_probs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SER Demo — v5.1')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .keras model file')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to scaler.pkl file')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run server on')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find model
    if args.model:
        model_path = args.model
    else:
        # Tìm model theo thứ tự ưu tiên
        candidates = ['best_model.keras', 'final_model.keras']
        model_path = None
        for name in candidates:
            path = os.path.join(script_dir, name)
            if os.path.exists(path):
                model_path = path
                break

    if not model_path or not os.path.exists(model_path):
        print("❌ Không tìm thấy model!")
        print("   Đặt best_model.keras trong cùng thư mục hoặc dùng: python demo.py --model path/to/model.keras")
        sys.exit(1)

    # Find scaler
    if args.scaler:
        scaler_path = args.scaler
    else:
        scaler_path = os.path.join(script_dir, 'scaler.pkl')

    model_name = os.path.basename(model_path)
    print(f"\n{'='*50}")
    print(f"  🎙️ Vietnamese SER Demo")
    print(f"{'='*50}")
    print(f"  Model: {model_name}")

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    print(f"  Server: http://localhost:{args.port}")
    print(f"{'='*50}\n")

    app.run(host='0.0.0.0', port=args.port, debug=False)
