import whisper
import os

os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin" 

# ==== CONFIG ====
AUDIO_DIR = "podcasts"
OUTPUT_DIR = "transcripts"
MODEL_NAME = "base"  

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = whisper.load_model(MODEL_NAME) # down model
option = whisper.DecodingOptions(fp16=False)

for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".mp3"):
        audio_path = os.path.join(AUDIO_DIR, filename)
        print("audio_path:", audio_path)
        print(f"Đang xử lý: {filename}")

        result = model.transcribe(audio_path, fp16=False)

        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Đã lưu: {txt_path}")
