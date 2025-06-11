import cohere
import time
from typing import List

# Nhập API Key Cohere trial tại đây
COHERE_API_KEY = "jlYcnl5zKGCbEc6n7ZT132tFlYjcksGyRzAvUeb0"
# Khởi tạo client
co = cohere.Client(COHERE_API_KEY)

# Cấu hình
MAX_CHUNK_CHARS = 3000  # tối đa ký tự mỗi chunk để giảm số request
RATE_LIMIT_SECONDS = 12  # chờ giữa các request để tránh 429 (5 calls/min)

def chunk_text(text: str, max_chunk_chars: int = MAX_CHUNK_CHARS, min_chunk_chars: int = 250) -> List[str]:
    """
    Chia văn bản thành các đoạn nhỏ theo ký tự,
    và gộp các đoạn quá nhỏ vào đoạn trước.
    """
    parts = [text[i : i + max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < min_chunk_chars:
            merged[-1] += part
        else:
            merged.append(part)
    return merged

def summarize_chunk_with_cohere(text_chunk: str, retry: int = 2) -> str:
    """
    Gửi một đoạn văn bản đến Cohere API để tóm tắt,
    retry khi gặp 429, kèm rate limiting.
    """
    for attempt in range(retry + 1):
        try:
            response = co.summarize(
                text=text_chunk,
                model="command",  # Updated for Cohere API v5+
                length="medium",
                format="bullets",
                temperature=0.3,
                additional_command="Tóm tắt bằng tiếng Việt"
            )
            # In v5+, the response is a dictionary-like object
            # Access the summary directly
            return response.summary if hasattr(response, 'summary') else response['summary']
        except Exception as e:
            err = str(e)
            if "status_code: 429" in err and attempt < retry:
                print(f"Nhận 429, chờ {RATE_LIMIT_SECONDS}s rồi retry ({attempt + 1}/{retry})")
                time.sleep(RATE_LIMIT_SECONDS)
                continue
            return f"Lỗi Cohere API: {err}"
    return "Lỗi Cohere API: quá nhiều lỗi"

def summarize_with_cohere(text: str) -> str:
    """
    Tóm tắt văn bản dài bằng Cohere API,
    tự động chia nhỏ, rate limited.
    """
    text = text.strip()
    if not text:
        return "Văn bản trống, không thể tóm tắt"

    # nếu ngắn, tóm tắt trực tiếp
    if len(text) <= MAX_CHUNK_CHARS:
        return summarize_chunk_with_cohere(text)

    # tách đoạn
    chunks = chunk_text(text)
    summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Tóm tắt đoạn {idx}/{len(chunks)}...")
        summaries.append(summarize_chunk_with_cohere(chunk))
        if idx < len(chunks):
            time.sleep(RATE_LIMIT_SECONDS)
    return "\n\n".join(summaries)