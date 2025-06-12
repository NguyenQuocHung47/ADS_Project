import torch
import numpy as np
import nltk
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer


nltk.download("punkt", quiet=True)
model_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Tính ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
def calculate_rouge_scores(reference: str, candidate: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }

# Tính BLEU score
def calculate_bleu_score(reference: str, candidate: str):
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    try:
        return nltk.translate.bleu_score.sentence_bleu(reference_tokens, candidate_tokens)
    except:
        return 0.0

# Tính MSE giữa embedding của hai đoạn văn (dùng tokenizer đơn giản hóa)
def calculate_mse_score(reference: str, candidate: str):
    try:
        ref_vec = model_embedder.encode(reference, convert_to_tensor=True)
        cand_vec = model_embedder.encode(candidate, convert_to_tensor=True)
        sim = cosine_similarity(ref_vec.unsqueeze(0), cand_vec.unsqueeze(0)).item()
        distance = 1 - sim
        return distance  # càng nhỏ càng giống nhau
    except Exception:
        return 1.0

# Tính BCE loss giả định
def calculate_bce_score(reference: str, candidate: str):
    try:
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        if len(ref_words) == 0:
            return 1.0
        labels = [1 if word in cand_words else 0 for word in ref_words]
        preds = [0.9 if word in cand_words else 0.1 for word in ref_words]
        if all(label == labels[0] for label in labels):
            return 1.0  # tránh lỗi log_loss nếu toàn 1 hoặc toàn 0
        return log_loss(labels, preds, labels=[0, 1])
    except:
        return 1.0