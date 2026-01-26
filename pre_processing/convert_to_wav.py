import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-y",
    "-i", "BDD.webm",
    "-ac", "2",
    "-ar", "16000",
    "BDD.wav"
], check=True)

print("Convert xong BDD.wav")