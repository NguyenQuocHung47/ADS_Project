import feedparser
import requests
from pydub import AudioSegment
import os
import re

# ==== CONFIG ====
RSS_FEED_URL = "https://feeds.simplecast.com/54nAGcIl"  
SAVE_DIR = "podcasts"
NUM_EPISODES = 3 

def clean_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(" ", "_")

os.makedirs(SAVE_DIR, exist_ok=True)
feed = feedparser.parse(RSS_FEED_URL)

for i, entry in enumerate(feed.entries[:NUM_EPISODES]):
    title = title = clean_filename(entry.title)
    audio_url = entry.enclosures[0].href  

    print(f"Tải tập {i+1}: {title}")
    response = requests.get(audio_url)

    if response.status_code == 200:
        mp3_path = os.path.join(SAVE_DIR, f"{title}.mp3")
        with open(mp3_path, "wb") as f:
            f.write(response.content)
        print(f"Đã lưu: {mp3_path}")
    else:
        print(f"Lỗi: {audio_url}")
