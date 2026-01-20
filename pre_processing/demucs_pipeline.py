import subprocess
import shutil
import sys
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
DEMUCS_MODEL = "htdemucs"
TEMP_OUT = BASE_DIR / "_demucs_tmp"

# Demucs memory-safe params
SEGMENT = "5"     # giảm peak RAM
OVERLAP = "0.1"
SHIFTS = "0"      # QUAN TRỌNG: giảm RAM rất mạnh
# =========================================


def run_demucs(audio_path: Path, device: str) -> bool:
    """
    Chạy demucs với device chỉ định.
    Trả về True nếu thành công, False nếu fail.
    """
    cmd = [
        "demucs",
        "-n", DEMUCS_MODEL,
        "--two-stems=vocals",
        "--device", device,
        "--segment", SEGMENT,
        "--overlap", OVERLAP,
        "--shifts", SHIFTS,
        "-o", str(TEMP_OUT),
        str(audio_path)
    ]

    print(f"⚙️ Demucs ({device})...")
    result = subprocess.run(cmd)

    return result.returncode == 0


def process_audio(audio_path: Path):
    print(f"\n🎵 Đang xử lý: {audio_path.name}")

    # Ưu tiên CUDA → fail thì fallback CPU
    if not run_demucs(audio_path, "cuda"):
        print("⚠️ CUDA fail → thử CPU")
        if not run_demucs(audio_path, "cpu"):
            print(f"❌ Demucs fail hoàn toàn: {audio_path.name}")
            return

    # Không tin cấu trúc folder → tìm vocals.wav bằng glob
    vocals = list(TEMP_OUT.glob("**/vocals.wav"))
    if not vocals:
        print(f"❌ Không tìm thấy vocals cho {audio_path.name}")
        return

    vocal_src = vocals[0]

    # Tạo folder output theo tên file
    target_dir = BASE_DIR / audio_path.stem
    target_dir.mkdir(exist_ok=True)

    target_vocal = target_dir / "vocals.wav"
    shutil.move(str(vocal_src), str(target_vocal))

    # Dọn rác folder demucs của file này
    shutil.rmtree(vocal_src.parent, ignore_errors=True)

    print(f"✅ Đã tạo: {target_vocal}")


def main():
    # Cho phép truyền file wav qua command line
    if len(sys.argv) > 1:
        wav_files = [Path(sys.argv[1])]
    else:
        wav_files = list(BASE_DIR.glob("*.wav"))

    if not wav_files:
        print("❌ Không có file .wav nào để xử lý")
        return

    for wav in wav_files:
        try:
            process_audio(wav)
        except Exception as e:
            print(f"⚠️ Lỗi {wav.name}: {e}")

    # Dọn tmp cuối cùng
    if TEMP_OUT.exists():
        shutil.rmtree(TEMP_OUT, ignore_errors=True)

    print("\n🚀 XONG – Mỗi file đã có folder vocals riêng")


if __name__ == "__main__":
    main()
