import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-y",
    "-i", "CMDAOp2.webm",
    "-ac", "2",
    "-ar", "16000",
    "CMDAOp2.wav"
], check=True)

print("Convert xong MC.wav")