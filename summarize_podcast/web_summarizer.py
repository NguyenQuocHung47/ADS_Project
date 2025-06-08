import cohere
from typing import List

# Nhập API Key Cohere trial tại đây
COHERE_API_KEY = "jlYcnl5zKGCbEc6n7ZT132tFlYjcksGyRzAvUeb0"
# Khởi tạo client
co = cohere.Client(COHERE_API_KEY)


def chunk_text(text: str, max_chunk_chars: int = 4000, min_chunk_chars: int = 250) -> List[str]:
    """
    Chia văn bản dài thành các đoạn nhỏ theo ký tự,
    và gộp các đoạn quá nhỏ vào đoạn trước.
    """
    parts = [text[i:i + max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < min_chunk_chars:
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def summarize_chunk_with_cohere(text_chunk: str) -> str:
    """
    Gửi một đoạn văn bản đến Cohere API để tóm tắt.
    """
    try:
        response = co.summarize(
            text=text_chunk,
            model="summarize-xlarge",
            length="medium",
            format="bullets",
            temperature=0.3,
            additional_command="Tóm tắt bằng tiếng Việt"
        )
        return response.summary
    except Exception as e:
        return f"Lỗi Cohere API: {e}"


def summarize_with_cohere(text: str) -> str:
    """
    Tóm tắt toàn bộ văn bản bằng Cohere API,
    tự động chia nhỏ nếu quá dài.
    """
    if not text.strip():
        return "Văn bản trống, không thể tóm tắt"

    # Nếu ngắn hơn ngưỡng, tóm tắt trực tiếp
    if len(text) <= 4000:
        return summarize_chunk_with_cohere(text)

    # Chia nhỏ văn bản
    chunks = chunk_text(text)
    summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        summary = summarize_chunk_with_cohere(chunk)
        summaries.append(summary)
    # Ghép kết quả
    return "\n\n".join(summaries)
