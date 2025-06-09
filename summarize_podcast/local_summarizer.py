import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional

# Danh sách model hỗ trợ và cấu hình tương ứng
SUPPORTED_MODELS: Dict[str, Dict[str, any]] = {
    "bart-large": {
        "path": "facebook/bart-large-cnn",
        "max_input": 1024,
        "default_summary_len": 150
    },
    "pegasus-dm": {
        "path": "google/pegasus-cnn_dailymail",
        "max_input": 1024,
        "default_summary_len": 150
    },
    "t5-base": {
        "path": "t5-base",
        "max_input": 512,
        "default_summary_len": 120
    }
}

class LocalSummarizer:
    def __init__(self, model_key: str):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_key}' không được hỗ trợ. Các model khả dụng: {', '.join(SUPPORTED_MODELS.keys())}")
        self.model_config = SUPPORTED_MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["path"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config["path"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def chunk_text(self, text: str, max_len: int) -> List[str]:
        tokens = self.tokenizer.encode(text, truncation=False)
        chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def summarize_chunks(self, chunks: List[str], max_input_length: int, summary_length: int) -> str:
        summaries = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True
            ).to(self.device)

            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=summary_length,
                min_length=max(20, summary_length // 2),
                num_beams=4,
                early_stopping=True
            )
            summaries.append(self.tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        return "\n\n".join(summaries)

    def summarize(self, text: str, max_input_length: Optional[int] = None, summary_length: Optional[int] = None) -> str:
        if max_input_length is None:
            max_input_length = self.model_config["max_input"]
        if summary_length is None:
            summary_length = self.model_config["default_summary_len"]

        # Tóm tắt từng phần
        chunks = self.chunk_text(text, max_input_length)
        first_pass = self.summarize_chunks(chunks, max_input_length, summary_length)

        # Nếu có nhiều đoạn → tóm tắt thêm lần nữa (optional)
        if len(chunks) > 1:
            final_chunks = self.chunk_text(first_pass, max_input_length)
            return self.summarize_chunks(final_chunks, max_input_length, summary_length)

        return first_pass


def summarize_hf(text: str, model_key: str, max_input_length: Optional[int] = None, summary_length: Optional[int] = None) -> str:
    summarizer = LocalSummarizer(model_key)
    return summarizer.summarize(text, max_input_length, summary_length)