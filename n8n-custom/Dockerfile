FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg git

RUN pip install torch torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install yt-dlp flask demucs torchcodec pydub

WORKDIR /app
COPY worker.py .
COPY youtube_cookies.txt .

CMD ["python", "worker.py"]
