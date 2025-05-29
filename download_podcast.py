import os
import requests
import feedparser

# ==== CONFIG ====
RSS_FEED_URL = "https://feeds.npr.org/510289/podcast.xml"
SAVE_DIR = "podcasts"
os.makedirs(SAVE_DIR, exist_ok=True)

# Đọc RSS
feed = feedparser.parse(RSS_FEED_URL)
print(f"📥 Tổng số tập tìm thấy: {len(feed.entries)}")

for entry in feed.entries:
    title = entry.title.replace(" ", "_").replace("?", "").replace("’", "").replace(":", "")
    filename = f"{SAVE_DIR}/{title}.mp3"

    # Bỏ qua nếu file đã tồn tại
    if os.path.exists(filename):
        print(f"⏭️ Bỏ qua vì đã tải: {title}")
        continue

    print(f"Đang tải tập: {title}")
    try:
        audio_url = entry.enclosures[0].href
        response = requests.get(audio_url)
        response.raise_for_status()  # Check lỗi HTTP

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"Đã lưu: {filename}")
    except Exception as e:
        print(f"Lỗi khi tải {title}: {e}")
