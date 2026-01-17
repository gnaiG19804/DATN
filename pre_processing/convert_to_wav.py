import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-y",
    "-i", "TNNV.webm",
    "-ac", "2",
    "-ar", "16000",
    "TNNV.wav"
], check=True)

print("Convert xong TNNV.wav")
