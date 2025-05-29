import json
import os
import time
import argparse
import math
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==== CONFIG MẶC ĐỊNH ====
TRANSCRIPT_DIR = "transcripts"
OUTPUT_JSON = "output/summaries.json"
HF_MODELS = {
    "distilbart": "sshleifer/distilbart-cnn-12-6",
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "pegasus": "google/pegasus-xsum",
    "bart-samsum": "philschmid/bart-large-cnn-samsum",
    "meeting-summary": "knkarthick/MEETING_SUMMARY"
}

DEFAULT_MAX_INPUT_LENGTH = 1024
DEFAULT_SUMMARY_LENGTH = 150

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

def read_transcripts() -> Dict[str, str]:
    transcripts = {}
    for filename in os.listdir(TRANSCRIPT_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TRANSCRIPT_DIR, filename), "r", encoding="utf-8") as f:
                transcripts[filename] = f.read()
    return transcripts

def chunk_text(text: str, tokenizer, max_len: int) -> List[str]:
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_chunks(chunks: List[str], tokenizer, model, max_input_length: int, summary_length: int) -> str:
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=max_input_length, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=summary_length,
            min_length=summary_length // 2,
            num_beams=4
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary.strip())
    return "\n".join(summaries)

def summarize_hf(text: str, model_id: str, max_input_length: int, summary_length: int) -> str:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        chunks = chunk_text(text, tokenizer, max_input_length)
        return summarize_chunks(chunks, tokenizer, model, max_input_length, summary_length)
    except Exception as e:
        return f"Lỗi khi xử lý với mô hình {model_id}: {str(e)}"

def main(selected_models: List[str], max_input_length: int, summary_length: int):
    transcripts = read_transcripts()
    all_results = []

    for filename, text in transcripts.items():
        entry = {"filename": filename, "summaries": {}}

        for model_key in selected_models:
            if model_key not in HF_MODELS:
                entry["summaries"][model_key] = {"summary": "⚠ Mô hình không tồn tại", "time": 0}
                continue

            model_id = HF_MODELS[model_key]
            print(f"🔄 Đang tóm tắt {filename} với mô hình {model_key}...")

            start = time.time()
            summary = summarize_hf(text, model_id, max_input_length, summary_length)
            duration = time.time() - start

            entry["summaries"][model_key] = {
                "summary": summary,
                "time": round(duration, 2),
                "model_id": model_id
            }

        all_results.append(entry)
        print(f"✅ Đã xử lý xong {filename}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Đã lưu toàn bộ kết quả vào {OUTPUT_JSON}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="distilbart,t5-small",
                        help=f"Danh sách mô hình, phân cách bằng dấu phẩy. Các mô hình hỗ trợ: {', '.join(HF_MODELS.keys())}")
    parser.add_argument("--max-input-length", type=int, default=DEFAULT_MAX_INPUT_LENGTH,
                        help="Số token tối đa đầu vào mỗi lần (mặc định 1024)")
    parser.add_argument("--summary-length", type=int, default=DEFAULT_SUMMARY_LENGTH,
                        help="Số token mong muốn cho mỗi phần tóm tắt (mặc định 150)")

    args = parser.parse_args()
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    main(selected_models, args.max_input_length, args.summary_length)