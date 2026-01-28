import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-y",
    "-i", "CTC8NS.webm",
    "-ac", "2",
    "-ar", "16000",
    "CTC8NS.wav"
], check=True)

print("Convert xong CTC8NS.wav")