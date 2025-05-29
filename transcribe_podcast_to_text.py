import whisper
import os

os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin" 

# ==== CONFIG ====
AUDIO_DIR = "podcasts"
OUTPUT_DIR = "transcripts"
MODEL_NAME = "base"  

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = whisper.load_model(MODEL_NAME)
option = whisper.DecodingOptions(fp16=False)

for filename in os.listdir(AUDIO_DIR):
    if not filename.endswith(".mp3"):
        continue

    audio_path = os.path.join(AUDIO_DIR, filename)
    text_filename = filename.replace(".mp3", ".txt")
    text_path = os.path.join(OUTPUT_DIR, text_filename)

    # Nếu file đã transcribe rồi thì bỏ qua
    if os.path.exists(text_path):
        print(f"Bỏ qua vì đã có: {text_filename}")
        continue

    print(f"Đang xử lý: {filename}")
    try:
        result = model.transcribe(audio_path)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Đã lưu transcript: {text_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý {filename}: {e}")
