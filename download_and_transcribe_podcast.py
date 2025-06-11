import os
import re
import feedparser
import requests
import whisper

# Config
RSS_FEED_URL = "https://feeds.simplecast.com/54nAGcIl" 
NUM_EPISODES = 5
AUDIO_DIR = "podcasts"
TEXT_DIR = "transcripts"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Hàm làm sạch tên file
def clean_filename(s):
    s = s.strip().replace(" ", "_")
    s = re.sub(r'[\\/*?:"<>|]', "", s)
    return s

# Tải podcast
print("Đang tải podcast từ RSS feed...")
feed = feedparser.parse(RSS_FEED_URL)
entries = feed.entries[:NUM_EPISODES]

for i, entry in enumerate(entries, 1):
    title = clean_filename(entry.title)
    audio_url = entry.enclosures[0].href
    mp3_path = os.path.join(AUDIO_DIR, f"{title}.mp3")

    if os.path.exists(mp3_path):
        print(f"Tập {i}: Đã tồn tại, bỏ qua.")
        continue

    print(f"Tải tập {i}: {title}")
    try:
        response = requests.get(audio_url)
        with open(mp3_path, "wb") as f:
            f.write(response.content)
        print(f"Đã lưu: {mp3_path}")
    except Exception as e:
        print(f"Lỗi khi tải {title}: {e}")

# Transcribe 
print("\nBắt đầu chuyển thành text bằng Whisper...")
model = whisper.load_model("base") 

for filename in os.listdir(AUDIO_DIR):
    if not filename.endswith(".mp3"):
        continue

    audio_path = os.path.join(AUDIO_DIR, filename)
    text_filename = filename.replace(".mp3", ".txt")
    text_path = os.path.join(TEXT_DIR, text_filename)

    if os.path.exists(text_path):
        print(f"Đã có transcript: {text_filename}")
        continue

    print(f"Đang xử lý: {filename}")
    try:
        result = model.transcribe(audio_path)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Đã lưu: {text_path}")
    except Exception as e:
        print(f"Lỗi khi transcribe {filename}: {e}")
