import json
import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from openai import OpenAI
from typing import Dict, List

# ==== CONFIG ====
TRANSCRIPT_DIR = "transcripts"
OUTPUT_JSON = "summaries.json"
# OPENAI_API_KEY = "sk-your-api-key"
OPENAI_MODEL = "gpt-3.5-turbo"

# Danh sách model HuggingFace
HF_MODELS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-base",
    "pegasus": "google/pegasus-xsum",
    "flan-t5": "google/flan-t5-base",
    "distilbart": "sshleifer/distilbart-cnn-12-6",
}

SUMMARY_LENGTH = 150
MAX_INPUT_LENGTH = 1024

# Khởi tạo môi trường
output_dir = os.path.dirname(OUTPUT_JSON)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# client = OpenAI(api_key=OPENAI_API_KEY)

def read_transcripts() -> Dict[str, str]:
    """Đọc tất cả file transcript trong thư mục"""
    transcripts = {}
    for filename in os.listdir(TRANSCRIPT_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TRANSCRIPT_DIR, filename), "r", encoding="utf-8") as f:
                transcripts[filename] = f.read()
    return transcripts

# def summarize_gpt(text: str) -> str:
#     """Tóm tắt dùng OpenAI API (phiên bản mới)"""
#     if not client:
#         return "OpenAI bị tắt do lỗi quota"
    
#     try:
#         response = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             messages=[{
#                 "role": "user",
#                 "content": f"Tóm tắt thành {SUMMARY_LENGTH//2} câu tiếng Việt:\n{text[:3000]}"
#             }]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Lỗi OpenAI: {str(e)}"

def summarize_hf(text: str, model_name: str) -> str:
    """Tóm tắt dùng HuggingFace models"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
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
        return f"Lỗi {model_name}: {str(e)}"

def main():
    transcripts = read_transcripts()
    all_results = []
    
    for filename, text in transcripts.items():
        entry = {"filename": filename, "summaries": {}}
        
        # if USE_OPENAI:
        #     start = time.time()
        #     entry['summaries']['gpt'] = {
        #         "summary": summarize_gpt(text),
        #         "time": time.time() - start
        #     }
        
        # HuggingFace models
        for model_alias, model_name in HF_MODELS.items():
            start = time.time()
            entry['summaries'][model_alias] = {
                "summary": summarize_hf(text, model_name),
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