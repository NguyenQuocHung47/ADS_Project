import streamlit as st
import os
import re
import time
import json
import threading
import atexit
import signal
import sys
from datetime import datetime
from typing import List, Dict, Optional
import whisper
from pydub import AudioSegment
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.metrics import (
    calculate_rouge_scores,
    calculate_bleu_score,
    calculate_mse_score,
    calculate_bce_score,
)

st.set_page_config(
        page_title="Podcast Transcription & Summarization",
        page_icon="🎙️",
        layout="wide"
    )

# Utility function for model name display
def get_display_name(model_key):
    """Convert internal model key to user-friendly display name"""
    display_names = {
        "bart-large": "BART",
        "pegasus-dm": "PEGASUS",
        "t5-base": "T5",
        "cohere": "Cohere"
    }
    return display_names.get(model_key, model_key.upper())

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import cohere

# Import functions from summarize_podcast folder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'summarize_podcast'))
from local_summarizer import LocalSummarizer, SUPPORTED_MODELS

# Import configurations
SAVE_DIR = "podcasts"
TEMP_DIR = "temp_audio"  # Separate directory for temporary files
TRANSCRIPT_DIR = "transcripts"
OUTPUT_DIR = "output"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "summaries.json")
MODEL_NAME = "base"  # Whisper model
# Model configurations
MAX_INPUT_LENGTH = 1024
SUMMARY_LENGTH = 150

# Cohere API configuration
COHERE_API_KEY = "jlYcnl5zKGCbEc6n7ZT132tFlYjcksGyRzAvUeb0"
# Try to get API key from secrets, but don't fail if not available
try:
    if "cohere_api_key" in st.secrets:
        COHERE_API_KEY = st.secrets["cohere_api_key"]
except Exception:
    # Secrets not available, will use the hardcoded trial key
    pass

# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variable to track loaded models for cleanup
loaded_models = {}

def cleanup_resources():
    """Clean up resources when the application is closing"""
    try:
        # Clear loaded models from memory
        global loaded_models
        for model_name, model in loaded_models.items():
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        loaded_models.clear()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up temporary files
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                try:
                    os.remove(os.path.join(TEMP_DIR, file))
                except:
                    pass
        
        print("Resources cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup functions (signal handlers don't work reliably in Streamlit)
atexit.register(cleanup_resources)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = []
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}
if 'api_performance' not in st.session_state:
    st.session_state.api_performance = []

def clean_filename(name):
    """Clean filename"""
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(" ", "_")

def calculate_text_similarity(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def calculate_summary_quality_metrics(original_text, summary):
    """Calculate quality metrics for summary"""
    try:
        # Text similarity score
        similarity_score = calculate_text_similarity(original_text, summary)
        
        # Compression ratio
        compression_ratio = len(summary) / len(original_text) if len(original_text) > 0 else 0
        
        # Word overlap ratio
        original_words = set(original_text.lower().split())
        summary_words = set(summary.lower().split())
        word_overlap = len(original_words.intersection(summary_words)) / len(original_words) if len(original_words) > 0 else 0
        
        return {
            'similarity_score': similarity_score,
            'compression_ratio': compression_ratio,
            'word_overlap': word_overlap,
            'summary_length': len(summary),
            'original_length': len(original_text)
        }
    except Exception as e:
        return {
            'similarity_score': 0.0,
            'compression_ratio': 0.0,
            'word_overlap': 0.0,
            'summary_length': len(summary) if summary else 0,
            'original_length': len(original_text) if original_text else 0
        }

def calculate_mse_score(reference_scores, predicted_scores):
    """Calculate MSE between reference and predicted scores"""
    try:
        if len(reference_scores) != len(predicted_scores):
            return float('inf')
        return mean_squared_error(reference_scores, predicted_scores)
    except:
        return float('inf')

def calculate_bce_score(y_true, y_pred):
    """Calculate Binary Cross Entropy score"""
    try:
        # Ensure values are between 0 and 1
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_true = np.clip(y_true, 0, 1)
        
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(bce)
    except:
        return float('inf')

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores between reference and candidate texts"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    except Exception as e:
        return {
            'rouge1_f': 0.0, 'rouge1_p': 0.0, 'rouge1_r': 0.0,
            'rouge2_f': 0.0, 'rouge2_p': 0.0, 'rouge2_r': 0.0,
            'rougeL_f': 0.0, 'rougeL_p': 0.0, 'rougeL_r': 0.0,
        }

def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score between reference and candidate texts"""
    try:
        # Tokenize the texts
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        # Calculate BLEU score with smoothing
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
        
        return float(bleu_score)
    except Exception as e:
        return 0.0

def calculate_model_comparison_metrics(results):
    """Calculate ROUGE, BLEU, MSE and BCE scores between different models"""
    if len(results) < 1:
        return {}
    
    comparison_metrics = {
        'rouge_scores': [],
        'bleu_scores': [],
        'model_pairs': [],
        'mse_similarity': 0.0,
        'bce_similarity': 0.0,
        'num_comparisons': 0
    }
    
    # Group results by filename to compare models on same content
    files_data = {}
    for result in results:
        filename = result['filename']
        if filename not in files_data:
            files_data[filename] = {}
        
        # Store summaries and their quality metrics
        if 'summaries' in result:
            for model_name, summary in result['summaries'].items():
                if not summary.startswith("Error:"):  # Skip error summaries
                    files_data[filename][model_name] = {
                        'summary': summary,
                        'length': len(summary),
                        'similarity': calculate_text_similarity(result.get('transcript', ''), summary)
                    }
    
    # Calculate comparison metrics between models
    all_rouge_scores = []
    all_bleu_scores = []
    similarity_scores_1 = []
    similarity_scores_2 = []
    
    for filename, models_data in files_data.items():
        model_names = list(models_data.keys())
        if len(model_names) >= 2:
            # Compare each pair of models
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    
                    summary1 = models_data[model1]['summary']
                    summary2 = models_data[model2]['summary']
                    
                    # Calculate ROUGE scores (using model1 as reference, model2 as candidate)
                    rouge_scores = calculate_rouge_scores(summary1, summary2)
                    # Use display names for model pairs
                    rouge_scores['model_pair'] = f"{get_display_name(model1)} vs {get_display_name(model2)}"
                    rouge_scores['filename'] = filename
                    all_rouge_scores.append(rouge_scores)
                    
                    # Calculate BLEU score
                    bleu_score = calculate_bleu_score(summary1, summary2)
                    all_bleu_scores.append({
                        'bleu_score': bleu_score,
                        'model_pair': f"{get_display_name(model1)} vs {get_display_name(model2)}",
                        'filename': filename
                    })
                    
                    # Get similarity scores for MSE/BCE calculation
                    sim1 = models_data[model1]['similarity']
                    sim2 = models_data[model2]['similarity']
                    
                    similarity_scores_1.append(sim1)
                    similarity_scores_2.append(sim2)
    
    if similarity_scores_1 and similarity_scores_2:
        # Calculate MSE between model similarity scores
        mse_score = calculate_mse_score(similarity_scores_1, similarity_scores_2)
        
        # Calculate BCE score (treating similarity scores as probabilities)
        bce_score = calculate_bce_score(
            np.array(similarity_scores_1), 
            np.array(similarity_scores_2)
        )
        
        comparison_metrics.update({
            'rouge_scores': all_rouge_scores,
            'bleu_scores': all_bleu_scores,
            'mse_similarity': mse_score,
            'bce_similarity': bce_score,
            'num_comparisons': len(similarity_scores_1)
        })
    
    return comparison_metrics

def summarize_with_local_model_enhanced(text, model_key):
    """Enhanced summarization with T5 anti-repetition fixes"""
    start_time = time.time()
    try:
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_key}' not supported")
        
        model_config = SUPPORTED_MODELS[model_key]
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_config["path"])
        
        # Chunk text if needed
        max_input = model_config["max_input"]
        summary_length = model_config["default_summary_len"]
        
        def chunk_text(text, max_len):
            tokens = tokenizer.encode(text, truncation=False)
            chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
            return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        
        chunks = chunk_text(text, max_input)
        summaries = []
        
        for chunk in chunks:
            # Add T5 prefix for better performance
            processed_chunk = chunk
            if "t5" in model_config["path"].lower():
                processed_chunk = f"summarize: {chunk}"
            
            inputs = tokenizer(
                processed_chunk,
                return_tensors="pt",
                max_length=max_input,
                truncation=True
            )
            
            # Enhanced generation parameters to reduce repetition
            if "t5" in model_config["path"].lower():
                # T5-specific anti-repetition parameters
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=summary_length,
                    min_length=max(20, summary_length // 3),
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    do_sample=False
                )
            else:
                # Standard parameters for other models
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=summary_length,
                    min_length=max(20, summary_length // 2),
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2
                )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        result = "\n\n".join(summaries)
        
        # If multiple chunks, do a final summarization pass
        if len(chunks) > 1:
            final_chunks = chunk_text(result, max_input)
            if len(final_chunks) > 1:
                final_summaries = []
                for chunk in final_chunks:
                    processed_chunk = chunk
                    if "t5" in model_config["path"].lower():
                        processed_chunk = f"summarize: {chunk}"
                    
                    inputs = tokenizer(processed_chunk, return_tensors="pt", max_length=max_input, truncation=True)
                    
                    if "t5" in model_config["path"].lower():
                        summary_ids = model.generate(
                            inputs["input_ids"],
                            max_length=summary_length,
                            min_length=max(20, summary_length // 3),
                            num_beams=4,
                            early_stopping=True,
                            repetition_penalty=1.3,
                            no_repeat_ngram_size=3,
                            length_penalty=1.0,
                            do_sample=False
                        )
                    else:
                        summary_ids = model.generate(
                            inputs["input_ids"],
                            max_length=summary_length,
                            min_length=max(20, summary_length // 2),
                            num_beams=4,
                            early_stopping=True,
                            repetition_penalty=1.1,
                            no_repeat_ngram_size=2
                        )
                    
                    final_summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
                result = "\n\n".join(final_summaries)
        
        # Calculate accuracy metrics
        quality_metrics = calculate_summary_quality_metrics(text, result)
        
        # Record performance
        end_time = time.time()
        st.session_state.api_performance.append({
            'api': f'Local-{model_key}',
            'operation': 'Summarization',
            'duration': end_time - start_time,
            'timestamp': datetime.now(),
            'status': 'Success',
            'input_length': len(text),
            'output_length': len(result),
            'similarity_score': quality_metrics['similarity_score'],
            'compression_ratio': quality_metrics['compression_ratio'],
            'word_overlap': quality_metrics['word_overlap'],
            'model_key': model_key
        })
        
        return result
        
    except Exception as e:
        end_time = time.time()
        st.session_state.api_performance.append({
            'api': f'Local-{model_key}',
            'operation': 'Summarization',
            'duration': end_time - start_time,
            'timestamp': datetime.now(),
            'status': 'Error',
            'error': str(e),
            'similarity_score': 0.0,
            'compression_ratio': 0.0,
            'word_overlap': 0.0
        })
        return f"Error with model {model_key}: {str(e)}"

def summarize_with_local_model(text, model_key):
    """Summarize text using local models from summarize_podcast folder"""
    # Use enhanced version for better T5 performance
    return summarize_with_local_model_enhanced(text, model_key)

def summarize_with_cohere(text, model_key="cohere"):
    """Summarize text using Cohere API"""
    start_time = time.time()
    try:
        if not COHERE_API_KEY:
            st.warning("⚠️ Cohere API key is not configured. Please add it to your secrets.toml file.")
            raise ValueError("Cohere API key is not configured. Please add a valid API key to use this feature.")
        
        # Initialize Cohere client
        co = cohere.Client(COHERE_API_KEY)
        
        # Split text into chunks if it's too long (Cohere has a limit)
        max_chunk_length = 25000  # Cohere's limit is around 25k characters
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        all_summaries = []
        for chunk in chunks:
            # Call Cohere API for summarization
            # Updated for Cohere API v5+
            response = co.summarize(
                text=chunk,
                length='medium',
                format='paragraph',
                extractiveness='medium',
                temperature=0.3,
                model="command",  # Specify model for v5+ compatibility
            )
            
            # In v5+, the response is a dictionary-like object
            # Access the summary directly
            summary = response.summary if hasattr(response, 'summary') else response['summary']
            all_summaries.append(summary)
        
        # Combine all summaries
        result = "\n\n".join(all_summaries)
        
        # If we have multiple chunks, summarize again to get a cohesive summary
        if len(chunks) > 1 and len(result) > max_chunk_length:
            response = co.summarize(
                text=result[:max_chunk_length],  # Take only what fits
                length='medium',
                format='paragraph',
                extractiveness='medium',
                temperature=0.3,
                model="command",  # Specify model for v5+ compatibility
            )
            result = response.summary if hasattr(response, 'summary') else response['summary']
        
        # Calculate accuracy metrics
        quality_metrics = calculate_summary_quality_metrics(text, result)
        
        # Record performance
        end_time = time.time()
        st.session_state.api_performance.append({
            'api': 'Cohere',
            'operation': 'Summarization',
            'duration': end_time - start_time,
            'timestamp': datetime.now(),
            'status': 'Success',
            'input_length': len(text),
            'output_length': len(result),
            'similarity_score': quality_metrics['similarity_score'],
            'compression_ratio': quality_metrics['compression_ratio'],
            'word_overlap': quality_metrics['word_overlap'],
            'model_key': model_key
        })
        
        return result
        
    except Exception as e:
        end_time = time.time()
        st.session_state.api_performance.append({
            'api': 'Cohere',
            'operation': 'Summarization',
            'duration': end_time - start_time,
            'timestamp': datetime.now(),
            'status': 'Error',
            'error': str(e),
            'similarity_score': 0.0,
            'compression_ratio': 0.0,
            'word_overlap': 0.0
        })
        return f"Error with Cohere API: {str(e)}"

def transcribe_audio(file_path, whisper_model):
    """Transcribe audio using Whisper"""
    start_time = time.time()
    try:
        result = whisper_model.transcribe(file_path, fp16=False)
        transcript = result["text"]
        
        # Record performance
        end_time = time.time()
        st.session_state.api_performance.append({
            'api': 'whisper',
            'operation': 'Transcription',
            'duration': end_time - start_time,
            'timestamp': datetime.now(),
            'status': 'Success',
            'input_length': 0,  # We don't know the audio length in characters
            'output_length': len(transcript),
            'similarity_score': 0.0,
            'compression_ratio': 0.0,
            'word_overlap': 0.0
        })
        
        return transcript
    except Exception as e:
        # Record error
        end_time = time.time()
        st.session_state.api_performance.append({
            'api': 'whisper',
            'operation': 'Transcription',
            'duration': end_time - start_time,
            'timestamp': datetime.now(),
            'status': 'Error',
            'error': str(e),
            'similarity_score': 0.0,
            'compression_ratio': 0.0,
            'word_overlap': 0.0
        })
        raise e

def process_audio_files(uploaded_files, selected_models):
    """Process uploaded audio files"""
    if not uploaded_files:
        st.error("No files uploaded!")
        return
    
    if not selected_models:
        st.error("No models selected!")
        return
    
    # Load Whisper model
    with st.spinner("Loading Whisper model..."):
        if 'whisper_model' not in st.session_state:
            whisper_model = whisper.load_model(MODEL_NAME, device="cpu")
            st.session_state.whisper_model = whisper_model
            loaded_models['whisper'] = whisper_model
        else:
            whisper_model = st.session_state.whisper_model
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    total_files = len(uploaded_files)
    all_results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")
        
        # Save uploaded file (permanently in SAVE_DIR, temporarily in TEMP_DIR for processing)
        permanent_path = os.path.join(SAVE_DIR, uploaded_file.name)
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        
        # Save permanent copy
        with open(permanent_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Save temporary copy for processing
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Transcribe
            with st.spinner(f"Transcribing {uploaded_file.name}..."):
                transcript = transcribe_audio(temp_path, whisper_model)
            
            # Save transcript
            title = clean_filename(os.path.splitext(uploaded_file.name)[0])
            txt_path = os.path.join(TRANSCRIPT_DIR, f"{title}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            # Summarize with selected local models
            summaries = {}
            with st.spinner(f"Summarizing {uploaded_file.name}..."):
                for model_key in selected_models:
                    try:
                        if model_key == "cohere":
                            summary = summarize_with_cohere(transcript, model_key)
                        else:
                            summary = summarize_with_local_model(transcript, model_key)
                        summaries[model_key] = summary
                    except Exception as e:
                        summaries[model_key] = f"Error: {str(e)}"
            
            # Store results
            entry = {
                "filename": uploaded_file.name,
                "audio_path": permanent_path,  # Store permanent path
                "transcript_path": txt_path,
                "transcript": transcript,
                "summaries": summaries,
                "timestamp": datetime.now().isoformat()
            }
            all_results.append(entry)
            
            # Display results in real-time
            with results_container:
                st.subheader(f"Results for: {uploaded_file.name}")
                
                # Create columns for each model
                model_keys = list(summaries.keys())
                
                # Use the global get_display_name function
                
                if len(model_keys) == 1:
                    st.write(f"**{get_display_name(model_keys[0])} Summary:**")
                    st.write(summaries[model_keys[0]])
                elif len(model_keys) == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{get_display_name(model_keys[0])} Summary:**")
                        st.write(summaries[model_keys[0]])
                    with col2:
                        st.write(f"**{get_display_name(model_keys[1])} Summary:**")
                        st.write(summaries[model_keys[1]])
                else:
                    # For 3 or more models, use tabs
                    tabs = st.tabs([get_display_name(key) for key in model_keys])
                    for i, (key, summary) in enumerate(summaries.items()):
                        with tabs[i]:
                            st.write(f"**{get_display_name(key)} Summary:**")
                            st.write(summary)
                
                with st.expander("View Full Transcript"):
                    st.text_area("Transcript", transcript, height=200, key=f"transcript_{i}")
                
                st.divider()
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up only temporary file (keep permanent file)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {e}")
    
    # Save all results to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    st.session_state.processing_results = all_results
    status_text.text("Processing complete!")
    st.success("All files processed successfully!")

def main(): 
    st.title("🎙️ Podcast Transcription & Summarization")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 Process Audio Files", "📊 Model Performance"])
    
    with tab1:
        st.header("Welcome to Podcast Transcription & Summarization!")
        st.write("""
        This application helps you transcribe and summarize audio files using advanced AI models.
        
        **Features:**
        - 🎵 Audio transcription using OpenAI Whisper
        - 📝 Text summarization using multiple local models (BART, Pegasus, T5)
        - 📊 Performance monitoring and accuracy metrics for all models
        - 💾 Automatic saving of transcripts and summaries
        
        **Supported formats:** MP3, WAV, M4A, FLAC, and more
        """)
        
        st.divider()
        
        # Model selection section
        st.subheader("Select Summarization Models")
        
        # Display model information
        with st.expander("ℹ️ Model Information"):
            st.write("""
            **Available Summarization Models:**
            - **BART**: Facebook's BART model fine-tuned on CNN/DailyMail dataset
            - **PEGASUS**: Google's Pegasus model for abstractive summarization
            - **T5**: Google's T5 model for text-to-text generation
            - **Cohere**: Cohere's API for advanced summarization
            
            **Available Transcription Models (Whisper):**
            - **Tiny**: Fastest model with lowest accuracy (good for quick tests)
            - **Base**: Good balance between speed and accuracy
            - **Small**: Better accuracy than Base, but slower
            - **Medium**: High accuracy with longer processing time
            - **Large**: Highest accuracy, but very slow and memory-intensive
            
            You can select multiple summarization models to compare their performance and accuracy.
            """)
        
        # Get available models and their display names
        available_models = list(SUPPORTED_MODELS.keys())
        
        # Create the multiselect with display names
        selected_display_names = st.multiselect(
            "Choose which models to use for summarization:",
            [get_display_name(model) for model in available_models],
            default=[get_display_name(model) for model in available_models],
            help="You can select multiple models to compare their performance"
        )
        
        # Convert selected display names back to model keys
        reverse_mapping = {get_display_name(k): k for k in available_models}
        selected_models = [reverse_mapping[display_name] for display_name in selected_display_names]
        
        if not selected_models:
            st.warning("Please select at least one model for summarization.")
        
        # Transcription model selection
        st.subheader("Select Transcription Model")
        whisper_models = {
            "tiny": "Tiny (Fast, less accurate)",
            "base": "Base (Balanced speed/accuracy)",
            "small": "Small (More accurate, slower)",
            "medium": "Medium (High accuracy, slow)",
            "large": "Large (Highest accuracy, very slow)"
        }
        
        selected_whisper_model = st.selectbox(
            "Choose Whisper model size for transcription:",
            options=list(whisper_models.keys()),
            format_func=lambda x: whisper_models[x],
            index=1,  # Default to "base"
            help="Larger models are more accurate but slower and use more memory"
        )
        
        # Update the MODEL_NAME global variable and clear cached model if changed
        global MODEL_NAME
        if MODEL_NAME != selected_whisper_model:
            MODEL_NAME = selected_whisper_model
            # Clear the cached model so it will reload with the new size
            if 'whisper_model' in st.session_state:
                del st.session_state.whisper_model
                if 'whisper' in loaded_models:
                    del loaded_models['whisper']
                st.info(f"Whisper model changed to {whisper_models[selected_whisper_model]}. The new model will be loaded when processing files.")
        else:
            MODEL_NAME = selected_whisper_model
        
        # File upload section
        st.subheader("Upload Audio Files")
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'],
            accept_multiple_files=True,
            help="You can upload multiple audio files at once"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                file_size = len(file.getbuffer()) / (1024 * 1024)  # Convert to MB
                st.write(f"- {file.name} ({file_size:.2f} MB)")
        
        # Show saved audio files
        if os.path.exists(SAVE_DIR) and os.listdir(SAVE_DIR):
            with st.expander("📂 Previously Saved Audio Files"):
                saved_files = os.listdir(SAVE_DIR)
                st.write(f"**{len(saved_files)} saved file(s):**")
                for file in saved_files:
                    file_path = os.path.join(SAVE_DIR, file)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                        st.write(f"- {file} ({file_size:.2f} MB)")
        
        # Process button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Process Files", type="primary", disabled=not uploaded_files or not selected_models):
                process_audio_files(uploaded_files, selected_models)
        with col2:
            if st.button("🧹 Clear Cache", help="Clear loaded models and temporary files to free memory"):
                # Clear models from session state
                if 'whisper_model' in st.session_state:
                    del st.session_state.whisper_model
                
                # Clear loaded models
                cleanup_resources()
                
                st.success("Cache cleared successfully!")
                st.rerun()
        
        # Display previous results if available
        if st.session_state.processing_results:
            st.divider()
            st.subheader("Previous Results")
            
            for i, result in enumerate(st.session_state.processing_results):
                with st.expander(f"📁 {result['filename']}"):
                    # Show file info
                    if 'audio_path' in result and os.path.exists(result['audio_path']):
                        file_size = os.path.getsize(result['audio_path']) / (1024 * 1024)
                        st.info(f"📂 Audio file saved: {result['audio_path']} ({file_size:.2f} MB)")
                    
                    # Display summaries from all models
                    summaries = result.get('summaries', {})
                    model_keys = list(summaries.keys())
                    
                    # Use the global get_display_name function
                    
                    if len(model_keys) == 1:
                        st.write(f"**{get_display_name(model_keys[0])} Summary:**")
                        st.write(summaries[model_keys[0]])
                    elif len(model_keys) == 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{get_display_name(model_keys[0])} Summary:**")
                            st.write(summaries[model_keys[0]])
                        with col2:
                            st.write(f"**{get_display_name(model_keys[1])} Summary:**")
                            st.write(summaries[model_keys[1]])
                    else:
                        # For 3 or more models, use tabs
                        tabs = st.tabs([get_display_name(key) for key in model_keys])
                        for j, (key, summary) in enumerate(summaries.items()):
                            with tabs[j]:
                                st.write(f"**{get_display_name(key)} Summary:**")
                                st.write(summary)
                    
                    if st.button(f"View Transcript", key=f"view_transcript_{i}"):
                        st.text_area("Full Transcript", result['transcript'], height=300)
    
    with tab2:
        st.header("📊 Model Performance Dashboard")
        st.write("Monitor the performance and accuracy metrics of all models used during processing.")
        
        if not st.session_state.api_performance:
            st.info("No model performance data available yet. Process some audio files to see performance metrics.")
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(st.session_state.api_performance)
        
        # Ensure all required columns exist
        required_columns = ['api', 'operation', 'status', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return
        
        # Add duration column if missing
        if 'duration' not in df.columns:
            df['duration'] = 0.0
        
        # Normalize model names for display
        def normalize_model_name(api_name):
            # Handle Local- prefix models
            if api_name.startswith('Local-'):
                base_name = api_name.replace('Local-', '')
                # Use the display name function for the base name
                return f"Local {get_display_name(base_name)}"
            # Handle non-local models using the display name function
            elif api_name in ['bart-large', 'pegasus-dm', 't5-base', 'cohere']:
                return get_display_name(api_name)
            # Handle Whisper models
            elif api_name == 'whisper':
                return f'Whisper {MODEL_NAME.capitalize()}'
            # Return other model names as is
            return api_name
            
        df['normalized_name'] = df['api'].apply(normalize_model_name)
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_calls = len(df)
            st.metric("Total Model Calls", total_calls)
        
        with col2:
            avg_duration = df['duration'].mean() if 'duration' in df.columns else 0.0
            st.metric("Avg Response Time", f"{avg_duration:.2f}s")
        
        with col3:
            success_rate = (df['status'] == 'Success').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col4:
            # Filter for successful summarization operations only
            summary_df = df[(df['operation'] == 'Summarization') & (df['status'] == 'Success')]
            if not summary_df.empty and 'similarity_score' in summary_df.columns:
                avg_similarity = summary_df['similarity_score'].mean()
                st.metric("Avg Similarity Score", f"{avg_similarity:.3f}")
            else:
                st.metric("Avg Similarity Score", "N/A")
        
        with col5:
            total_duration = df['duration'].sum()
            st.metric("Total Processing Time", f"{total_duration:.2f}s")
        
        st.divider()
        
        # Performance charts
        st.subheader("Response Time by Model")
        if not df.empty:
            # Create a copy with better column names for display
            plot_df = df.copy()
            plot_df = plot_df.rename(columns={
                'normalized_name': 'Model',
                'duration': 'Duration (seconds)'
            })
            fig = px.box(plot_df, x='Model', y='Duration (seconds)', title="Response Time Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy metrics charts (only for successful summarization operations)
        summary_df = df[(df['operation'] == 'Summarization') & (df['status'] == 'Success')]
        # Add normalized name to summary_df
        summary_df['normalized_name'] = summary_df['api'].apply(normalize_model_name)
        
        if not summary_df.empty and 'similarity_score' in summary_df.columns:
            st.divider()
            st.subheader("📈 Model Accuracy Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Similarity Scores by Model")
                # Create a copy with better column names for display
                plot_df = summary_df.copy()
                plot_df = plot_df.rename(columns={
                    'normalized_name': 'Model',
                    'similarity_score': 'Similarity Score'
                })
                fig = px.box(plot_df, x='Model', y='Similarity Score', 
                           title="Text Similarity Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Compression Ratio by Model")
                # Create a copy with better column names for display
                plot_df = summary_df.copy()
                plot_df = plot_df.rename(columns={
                    'normalized_name': 'Model',
                    'compression_ratio': 'Compression Ratio'
                })
                fig = px.box(plot_df, x='Model', y='Compression Ratio', 
                           title="Compression Ratio Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.subheader("Word Overlap by Model")
                # Create a copy with better column names for display
                plot_df = summary_df.copy()
                plot_df = plot_df.rename(columns={
                    'normalized_name': 'Model',
                    'word_overlap': 'Word Overlap'
                })
                fig = px.box(plot_df, x='Model', y='Word Overlap', 
                           title="Word Overlap Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy vs Speed scatter plot
            st.subheader("⚡ Accuracy vs Speed Analysis")
            # Create a copy with better column names for display
            plot_df = summary_df.copy()
            plot_df = plot_df.rename(columns={
                'duration': 'Processing Time',
                'similarity_score': 'Similarity Score',
                'compression_ratio': 'Compression Ratio',
                'word_overlap': 'Word Overlap',
                'normalized_name': 'Model'
            })
            
            fig = px.scatter(plot_df, x='Processing Time', y='Similarity Score', color='Model',
                           size='Compression Ratio', hover_data=['Word Overlap'],
                           title="Model Performance: Accuracy vs Speed")
            fig.update_layout(xaxis_title="Processing Time (seconds)", 
                            yaxis_title="Similarity Score",
                            legend_title="Model")
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison table
            st.subheader("📊 Model Performance Comparison")
            if len(summary_df) > 0:
                model_stats = summary_df.groupby('normalized_name').agg({
                    'duration': 'mean',
                    'similarity_score': 'mean',
                    'compression_ratio': 'mean',
                    'word_overlap': 'mean'
                }).round(4)
                
                model_stats = model_stats.reset_index()
                
                # Rename columns for better display
                model_stats.columns = ['Model', 'Avg Duration (s)', 'Similarity Score',
                                     'Compression Ratio', 'Word Overlap']
                
                st.dataframe(model_stats, use_container_width=True)
            
            # Model comparison metrics (ROUGE, BLEU, MSE and BCE)
            if st.session_state.processing_results:
                st.subheader("🔄 Model Comparison Metrics")
                comparison_metrics = calculate_model_comparison_metrics(st.session_state.processing_results)
                st.info("""
                    **ROUGE Scores**: Measure overlap between model outputs (higher is better)
                    - ROUGE-1: Unigram overlap
                    - ROUGE-2: Bigram overlap  
                    - ROUGE-L: Longest common subsequence
                    
                    **BLEU Score**: Measures n-gram precision between outputs (higher is better)
                    
                    **MSE/BCE**: Measure consistency between model similarity scores (lower is better)
                    """)
                
                if comparison_metrics and comparison_metrics['num_comparisons'] > 0:
                    # ---- Individual Model Evaluation ----
                    individual_scores = {}
                    for result in st.session_state.processing_results:
                        transcript = result.get("transcript", "")
                        summaries = result.get("summaries", {})
                        for model, summary in summaries.items():
                            if not summary or summary.startswith("Error"):
                                continue
                            if model not in individual_scores:
                                individual_scores[model] = {
                                    'rouge1': [], 'rouge2': [], 'rougeL': [],
                                    'bleu': [], 'mse': [], 'bce': [], 'count': 0
                                }

                            # Score between summary and original
                            rouge = calculate_rouge_scores(transcript, summary)
                            bleu = calculate_bleu_score(transcript, summary)
                            mse = calculate_mse_score(transcript, summary)
                            bce = calculate_bce_score(transcript, summary)

                            individual_scores[model]['rouge1'].append(rouge['rouge1_f'])
                            individual_scores[model]['rouge2'].append(rouge['rouge2_f'])
                            individual_scores[model]['rougeL'].append(rouge['rougeL_f'])
                            individual_scores[model]['bleu'].append(bleu)
                            individual_scores[model]['mse'].append(mse)
                            individual_scores[model]['bce'].append(bce)
                            individual_scores[model]['count'] += 1

                    # Create dataframe from results
                    if individual_scores:
                        import plotly.graph_objects as go

                        df_individual = pd.DataFrame([
                            {
                                "Model": get_display_name(model),
                                "ROUGE-1": np.mean(scores['rouge1']),
                                "ROUGE-2": np.mean(scores['rouge2']),
                                "ROUGE-L": np.mean(scores['rougeL']),
                                "BLEU": np.mean(scores['bleu']),
                                "MSE": np.mean(scores['mse']),
                                "BCE": np.mean(scores['bce']),
                            }
                            for model, scores in individual_scores.items()
                        ])

                        df_individual = df_individual.round(4)

                        # 📊 1. ROUGE Chart
                        st.markdown("**ROUGE Scores by Model**")
                        fig_rouge = go.Figure()
                        for rouge_type in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
                            fig_rouge.add_trace(go.Bar(
                                x=df_individual["Model"],
                                y=df_individual[rouge_type],
                                name=rouge_type
                            ))
                        fig_rouge.update_layout(barmode='group', yaxis_title="ROUGE F1 Score", xaxis_title="Model")
                        st.plotly_chart(fig_rouge, use_container_width=True)

                        # 📊 2. BLEU Chart
                        st.markdown("**BLEU Score by Model**")
                        fig_bleu = go.Figure(data=[
                            go.Bar(x=df_individual["Model"], y=df_individual["BLEU"], name="BLEU")
                        ])
                        fig_bleu.update_layout(yaxis_title="BLEU", xaxis_title="Model")
                        st.plotly_chart(fig_bleu, use_container_width=True)

                        # 📊 3. MSE Chart
                        st.markdown("**MSE Score by Model**")
                        fig_mse = go.Figure(data=[
                            go.Bar(x=df_individual["Model"], y=df_individual["MSE"], name="MSE")
                        ])
                        fig_mse.update_layout(yaxis_title="MSE (lower is better)", xaxis_title="Model")
                        st.plotly_chart(fig_mse, use_container_width=True)

                        # 📊 4. BCE Chart
                        st.markdown("**BCE Score by Model**")
                        fig_bce = go.Figure(data=[
                            go.Bar(x=df_individual["Model"], y=df_individual["BCE"], name="BCE")
                        ])
                        fig_bce.update_layout(yaxis_title="BCE (lower is better)", xaxis_title="Model")
                        st.plotly_chart(fig_bce, use_container_width=True)

                    else:
                        st.info("There is insufficient data to evaluate each model individually.")
                    
                    # ROUGE scores detailed view
                    if comparison_metrics['rouge_scores']:
                        st.subheader("📊 ROUGE Scores Breakdown")
                        rouge_df = pd.DataFrame(comparison_metrics['rouge_scores'])
                        
                        # Create visualizations for ROUGE scores
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create a copy with renamed columns for better display
                            plot_df = rouge_df.copy()
                            plot_df = plot_df.rename(columns={
                                'rouge1_f': 'ROUGE-1',
                                'rouge2_f': 'ROUGE-2',
                                'rougeL_f': 'ROUGE-L'
                            })
                            fig = px.box(plot_df, y=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], 
                                       title="ROUGE F1 Scores Distribution")
                            fig.update_layout(yaxis_title="F1 Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Average ROUGE scores by model pair
                            avg_rouge_by_pair = rouge_df.groupby('model_pair')[['rouge1_f', 'rouge2_f', 'rougeL_f']].mean().reset_index()
                            # Rename columns for better display
                            avg_rouge_by_pair = avg_rouge_by_pair.rename(columns={
                                'rouge1_f': 'ROUGE-1',
                                'rouge2_f': 'ROUGE-2',
                                'rougeL_f': 'ROUGE-L',
                                'model_pair': 'Model Pair'
                            })
                            fig = px.bar(avg_rouge_by_pair, x='Model Pair', y=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                                       title="Average ROUGE Scores by Model Pair")
                            fig.update_layout(xaxis_title="Model Pair", yaxis_title="F1 Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed ROUGE table
                        st.subheader("Detailed ROUGE Scores")
                        display_rouge_df = rouge_df[['filename', 'model_pair', 'rouge1_f', 'rouge2_f', 'rougeL_f']].copy()
                        # Rename columns for better display
                        display_rouge_df.columns = ['Filename', 'Model Pair', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
                        display_rouge_df = display_rouge_df.round(4)
                        st.dataframe(display_rouge_df, use_container_width=True)
                    
                    # BLEU scores detailed view
                    if comparison_metrics['bleu_scores']:
                        st.subheader("📈 BLEU Scores Analysis")
                        bleu_df = pd.DataFrame(comparison_metrics['bleu_scores'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create a copy with renamed columns for better display
                            plot_df = bleu_df.copy()
                            plot_df = plot_df.rename(columns={
                                'bleu_score': 'BLEU Score'
                            })
                            fig = px.box(plot_df, y='BLEU Score', title="BLEU Scores Distribution")
                            fig.update_layout(yaxis_title="BLEU Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            avg_bleu_by_pair = bleu_df.groupby('model_pair')['bleu_score'].mean().reset_index()
                            # Rename columns for better display
                            avg_bleu_by_pair = avg_bleu_by_pair.rename(columns={
                                'model_pair': 'Model Pair',
                                'bleu_score': 'BLEU Score'
                            })
                            fig = px.bar(avg_bleu_by_pair, x='Model Pair', y='BLEU Score',
                                       title="Average BLEU Scores by Model Pair")
                            fig.update_layout(xaxis_title="Model Pair", yaxis_title="BLEU Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed BLEU table
                        st.subheader("Detailed BLEU Scores")
                        display_bleu_df = bleu_df[['filename', 'model_pair', 'bleu_score']].copy()
                        # Rename columns for better display
                        display_bleu_df.columns = ['Filename', 'Model Pair', 'BLEU Score']
                        display_bleu_df = display_bleu_df.round(4)
                        st.dataframe(display_bleu_df, use_container_width=True)

                else:
                    st.info("Need at least 2 different models processing the same content to calculate comparison metrics.")
        
        # Timeline chart
        st.subheader("Model Calls Timeline")
        if not df.empty:
            try:
                # Make a copy to avoid modifying the original dataframe
                chart_df = df.copy()
                
                # Convert timestamp to datetime safely
                if 'timestamp' in chart_df.columns:
                    chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
                    
                    # Create the chart
                    fig = px.scatter(chart_df, x='timestamp', y='duration', color='normalized_name', 
                                   size='duration', hover_data=['operation', 'status'],
                                   title="Model Calls Over Time")
                    fig.update_layout(xaxis_title="Time", yaxis_title="Duration (seconds)", 
                                    legend_title="Model")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Timeline chart requires timestamp data.")
            except Exception as e:
                st.warning(f"Could not create timeline chart: {str(e)}")
        
        # Detailed performance table
        st.subheader("Detailed Performance Log")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            model_filter = st.selectbox("Filter by Model", ["All"] + list(df['normalized_name'].unique()))
        with col2:
            operation_filter = st.selectbox("Filter by Operation", ["All"] + list(df['operation'].unique()))
        with col3:
            status_filter = st.selectbox("Filter by Status", ["All"] + list(df['status'].unique()))
        
        # Apply filters
        try:
            filtered_df = df.copy()
            if model_filter != "All" and 'normalized_name' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['normalized_name'] == model_filter]
            if operation_filter != "All" and 'operation' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['operation'] == operation_filter]
            if status_filter != "All" and 'status' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
        except Exception as e:
            st.warning(f"Error applying filters: {str(e)}")
            filtered_df = df.copy()
        
        # Display table
        if not filtered_df.empty:
            # Select columns based on what's available
            base_columns = ['timestamp', 'normalized_name', 'operation', 'duration', 'status']
            # Only include columns that exist in the DataFrame
            available_base_columns = [col for col in base_columns if col in filtered_df.columns]
            
            accuracy_columns = []
            if 'similarity_score' in filtered_df.columns:
                potential_accuracy_columns = ['similarity_score', 'compression_ratio', 'word_overlap']
                # Only include columns that exist in the DataFrame
                accuracy_columns = [col for col in potential_accuracy_columns if col in filtered_df.columns]
            
            display_columns = available_base_columns + accuracy_columns
            
            # Ensure we have at least one column to display
            if not display_columns:
                st.warning("No valid columns to display")
                return
                
            display_df = filtered_df[display_columns].copy()
            
            # Rename columns for better display
            display_df = display_df.rename(columns={
                'normalized_name': 'Model',
                'duration': 'Duration (s)',
                'operation': 'Operation',
                'status': 'Status',
                'timestamp': 'Time',
                'similarity_score': 'Similarity Score',
                'compression_ratio': 'Compression Ratio',
                'word_overlap': 'Word Overlap'
            })
            
            # Round numerical columns
            if 'duration' in display_df.columns:
                display_df['duration'] = display_df['duration'].round(3)
            if accuracy_columns:
                for col in accuracy_columns:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].round(4)
            
            # Format timestamp if it exists and is a datetime type
            if 'timestamp' in display_df.columns:
                try:
                    # Ensure timestamp is datetime type before using dt accessor
                    if not pd.api.types.is_datetime64_any_dtype(display_df['timestamp']):
                        display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
                except Exception as e:
                    # If formatting fails, leave as is
                    st.warning(f"Could not format timestamp: {str(e)}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No data matches the selected filters.")
        
        # Clear performance data button
        if st.button("🗑️ Clear Performance Data"):
            st.session_state.api_performance = []
            st.rerun()

if __name__ == "__main__":
    main()