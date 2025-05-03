import json
import os
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import openai
from typing import Dict, List

# ==== CONFIG ====
TRANSCRIPT_DIR = "transcripts"
OUTPUT_JSON = "output/summaries.json"
HF_MODELS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-base",
}
SUMMARY_LENGTH = 150  # Số token mong muốn
MAX_INPUT_LENGTH = 1024  # Giới hạn đầu vào cho model local

# Khởi tạo môi trường
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

def read_transcripts() -> Dict[str, str]:
    """Đọc tất cả file transcript trong thư mục"""
    transcripts = {}
    for filename in os.listdir(TRANSCRIPT_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TRANSCRIPT_DIR, filename), "r", encoding="utf-8") as f:
                transcripts[filename] = f.read()
    return transcripts

def summarize_hf(text: str, model_name: str) -> str:
    """Tóm tắt dùng HuggingFace models"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Chunk text nếu quá dài
        inputs = tokenizer(
            text[:MAX_INPUT_LENGTH],
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True
        )
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=SUMMARY_LENGTH,
            min_length=SUMMARY_LENGTH//2,
            num_beams=4
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Lỗi model {model_name}: {str(e)}"

def main():
    transcripts = read_transcripts()
    all_results = []
    
    for filename, text in transcripts.items():
        entry = {"filename": filename, "summaries": {}}
        
        # HuggingFace models
        for model_name in HF_MODELS:
            start = time.time()
            entry['summaries'][model_name] = {
                "summary": summarize_hf(text, HF_MODELS[model_name]),
                "time": time.time() - start
            }
        
        all_results.append(entry)
        print(f"Đã xử lý {filename}")

    # Lưu kết quả
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Đã lưu kết quả vào {OUTPUT_JSON}")

if __name__ == "__main__":
    main()