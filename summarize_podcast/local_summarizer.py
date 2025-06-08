from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
import torch

# Danh sách model hỗ trợ và cấu hình tương ứng
SUPPORTED_MODELS: Dict[str, Dict[str, any]] = {
    "distilbart": {
        "path": "sshleifer/distilbart-cnn-12-6",
        "max_input": 1024,
        "default_summary_len": 150
    },
    "t5-small": {
        "path": "t5-small",
        "max_input": 512,
        "default_summary_len": 100
    },
    "pegasus": {
        "path": "google/pegasus-xsum",
        "max_input": 512,
        "default_summary_len": 64
    }
}

class LocalSummarizer:
    """
    Lớp để tóm tắt văn bản sử dụng các model local.
    """
    def __init__(self, model_key: str):
        """
        Khởi tạo LocalSummarizer với model được chỉ định.
        
        Args:
            model_key: Tên model (phải có trong SUPPORTED_MODELS).
        
        Raises:
            ValueError: Nếu model không được hỗ trợ.
        """
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_key}' không được hỗ trợ. Các model khả dụng: {', '.join(SUPPORTED_MODELS.keys())}")
        
        self.model_config = SUPPORTED_MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["path"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config["path"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def chunk_text(self, text: str, max_len: int) -> List[str]:
        """
        Chia văn bản thành các đoạn nhỏ phù hợp với giới hạn đầu vào của model.
        
        Args:
            text: Văn bản cần chia.
            max_len: Độ dài tối đa của mỗi đoạn.
        
        Returns:
            Danh sách các đoạn văn bản.
        """
        tokens = self.tokenizer.encode(text, truncation=False, return_tensors="pt")[0]
        chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    def summarize_chunks(self, chunks: List[str], max_input_length: int, summary_length: int) -> str:
        """
        Tóm tắt từng đoạn và kết hợp các kết quả.
        
        Args:
            chunks: Danh sách các đoạn văn bản.
            max_input_length: Độ dài tối đa của đầu vào.
            summary_length: Độ dài mong muốn của bản tóm tắt.
        
        Returns:
            Văn bản tóm tắt kết hợp từ các đoạn.
        """
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
        """
        Tóm tắt văn bản sử dụng model local.
        
        Args:
            text: Văn bản cần tóm tắt.
            max_input_length: Độ dài tối đa của đầu vào (nếu None, sử dụng giá trị mặc định của model).
            summary_length: Độ dài mong muốn của bản tóm tắt (nếu None, sử dụng giá trị mặc định của model).
        
        Returns:
            Văn bản tóm tắt.
        
        Raises:
            RuntimeError: Nếu có lỗi trong quá trình tóm tắt.
        """
        try:
            if max_input_length is None:
                max_input_length = self.model_config["max_input"]
            if summary_length is None:
                summary_length = self.model_config["default_summary_len"]
            
            chunks = self.chunk_text(text, max_input_length)
            return self.summarize_chunks(chunks, max_input_length, summary_length)
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tóm tắt với model '{self.model_config['path']}': {str(e)}")

# Hàm tiện ích để tóm tắt với model cụ thể
def summarize_hf(text: str, model_key: str, max_input_length: Optional[int] = None, summary_length: Optional[int] = None) -> str:
    """
    Tóm tắt văn bản sử dụng model local được chỉ định.
    
    Args:
        text: Văn bản cần tóm tắt.
        model_key: Tên model (phải có trong SUPPORTED_MODELS).
        max_input_length: Độ dài tối đa của đầu vào (nếu None, sử dụng giá trị mặc định của model).
        summary_length: Độ dài mong muốn của bản tóm tắt (nếu None, sử dụng giá trị mặc định của model).
    
    Returns:
        Văn bản tóm tắt.
    
    Raises:
        ValueError: Nếu model không được hỗ trợ.
        RuntimeError: Nếu có lỗi trong quá trình tóm tắt.
    """
    summarizer = LocalSummarizer(model_key)
    return summarizer.summarize(text, max_input_length, summary_length)