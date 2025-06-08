import json
import os
import time
import argparse
from typing import Dict, List
from local_summarizer import SUPPORTED_MODELS as LOCAL_MODELS, summarize_hf
from web_summarizer import summarize_with_cohere

# Cấu hình đường dẫn
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPT_DIR = os.path.join(PROJECT_DIR, "transcripts")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "summaries.json")

# Thời gian giới hạn (giây) cho mỗi model
TIMEOUT: float = None

# Tạo thư mục nếu chưa tồn tại
def setup_directories() -> None:
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trả về danh sách model hỗ trợ
def get_available_models() -> Dict[str, List[str]]:
    return {
        "local": list(LOCAL_MODELS.keys()),
        "web": ["cohere"],
        "all": list(LOCAL_MODELS.keys()) + ["cohere"]
    }

# Kiểm tra model hợp lệ
def validate_models(selected_models: List[str], model_type: str = "all") -> List[str]:
    available = get_available_models()
    valid = [m for m in selected_models if m in available[model_type]]
    if not valid:
        raise ValueError(
            f"Không có model hợp lệ. Local: {', '.join(available['local'])}; Web: {', '.join(available['web'])}"
        )
    return valid

# Đọc transcript từ thư mục
def read_transcripts() -> Dict[str, str]:
    files = {}
    if not os.path.exists(TRANSCRIPT_DIR):
        return files
    for fname in os.listdir(TRANSCRIPT_DIR):
        if fname.endswith('.txt'):
            path = os.path.join(TRANSCRIPT_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    files[fname] = content
            except Exception:
                continue
    return files

# Gọi tóm tắt với model, có giới hạn thời gian
def summarize_with_model(model_key: str, text: str, max_len: int, summary_len: int) -> dict:
    start = time.time()
    result = {"model_id": model_key, "time": 0, "summary": "", "status": "success"}
    try:
        if model_key in LOCAL_MODELS:
            summary = summarize_hf(text, model_key, max_len, summary_len)
        else:
            # model_key == 'cohere'
            summary = summarize_with_cohere(text)
        elapsed = time.time() - start
        if TIMEOUT and elapsed > TIMEOUT:
            raise TimeoutError(f"Quá thời gian: {elapsed:.2f}s > {TIMEOUT}s")
        result['summary'] = summary
    except Exception as e:
        result.update({"summary": f"Lỗi: {e}", "status": "failed"})
    result['time'] = round(time.time() - start, 2)
    return result

# Hàm chính
def main() -> None:
    global TIMEOUT
    parser = argparse.ArgumentParser(description="Tóm tắt văn bản bằng local hoặc Cohere API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--local', action='store_true', help='Chỉ dùng model local')
    group.add_argument('--web', action='store_true', help='Chỉ dùng Cohere API')
    group.add_argument('--all', action='store_true', help='Dùng tất cả model')
    group.add_argument('--models', type=str, help='Danh sách model, phân tách bởi dấu phẩy')
    parser.add_argument('--max-len', type=int, default=1024, help='Độ dài tối đa đầu vào cho local')
    parser.add_argument('--summary-len', type=int, default=150, help='Độ dài mong muốn của tóm tắt')
    parser.add_argument('--timeout', type=float, default=None, help='Giới hạn thời gian (s) cho mỗi model')
    parser.add_argument('--output', type=str, default=OUTPUT_JSON, help='Đường dẫn file kết quả JSON')
    args = parser.parse_args()

    TIMEOUT = args.timeout
    setup_directories()
    try:
        if args.local:
            selected = validate_models(list(LOCAL_MODELS.keys()), 'local')
        elif args.web:
            selected = validate_models(['cohere'], 'web')
        elif args.all:
            selected = validate_models(get_available_models()['all'])
        else:
            sel = [m.strip() for m in args.models.split(',')]
            selected = validate_models(sel)
    except ValueError as e:
        print(e)
        return

    transcripts = read_transcripts()
    if not transcripts:
        print(f"Không tìm thấy file transcript trong {TRANSCRIPT_DIR}")
        return

    results = []
    for fname, text in transcripts.items():
        print(f"Xử lý file: {fname}")
        model_results = {}
        if len(text) > 50000:
            print(f"File {fname} dài {len(text)} ký tự, sẽ tự chia đoạn.")
        for model in selected:
            print(f"- Model: {model}")
            res = summarize_with_model(model, text, args.max_len, args.summary_len)
            model_results[model] = res
        results.append({'filename': fname, 'summaries': model_results})

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Kết quả đã lưu tại {args.output}")
    except Exception as e:
        print(f"Lỗi lưu file: {e}")

if __name__ == '__main__':
    main()