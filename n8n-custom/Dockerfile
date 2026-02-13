FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    unzip \
    && curl -fsSL https://deno.land/install.sh | sh \
    && ln -s /root/.deno/bin/deno /usr/local/bin/deno \
    && rm -rf /var/lib/apt/lists/*

# torch cpu
RUN pip install torch torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# python deps
RUN pip install --no-cache-dir \
    yt-dlp \
    flask \
    demucs \
    torchcodec \
    pydub

WORKDIR /app
COPY worker.py .
COPY youtube_cookies.txt .

CMD ["python", "worker.py"]

