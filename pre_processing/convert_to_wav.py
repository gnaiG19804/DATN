import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-y",
    "-i", "QBBp3.webm",
    "-ac", "2",
    "-ar", "16000",
    "QBBp3.wav"
], check=True)

print("Convert xong QBBp3.wav")