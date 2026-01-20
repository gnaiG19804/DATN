import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-y",
    "-i", "STp2.webm",
    "-ac", "2",
    "-ar", "16000",
    "ST.wav"
], check=True)

print("Convert xong ST.wav")