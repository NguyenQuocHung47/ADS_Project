import os
import re
import time
import json
import streamlit as st
import tempfile

# Import heavy libraries only when needed
st.sidebar.info("Loading dependencies... This may take a moment on first run.")

# Lazy loading for heavy ML libraries
@st.cache_resource
def load_whisper():
    import whisper
    return whisper

@st.cache_resource
def load_transformers():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    return AutoTokenizer, AutoModelForSeq2SeqLM

# Import configurations from existing files
SAVE_DIR = "podcasts"
TRANSCRIPT_DIR = "transcripts"
OUTPUT_DIR = "output"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "summaries.json")
MODEL_NAME = "base"  # Whisper model
HF_MODELS = {
    "bart": "facebook/bart-large-cnn",
}
SUMMARY_LENGTH = 150
MAX_INPUT_LENGTH = 1024

# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions
def clean_filename(name):
    """Clean filename from download_podcast.py"""
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(" ", "_")

def summarize_hf(text, model_name):
    """Summarize text using HuggingFace models from summarize_podcast.py"""
    try:
        # Load transformers only when needed
        AutoTokenizer, AutoModelForSeq2SeqLM = load_transformers()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Chunk text if too long
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
        return f"Error with model {model_name}: {str(e)}"

# Initialize session state variables if they don't exist
if 'files' not in st.session_state:
    st.session_state.files = []
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# App title and description
st.title("Podcast Transcription and Summarization")
st.markdown("Upload MP3 files to transcribe and summarize podcast content.")

# File uploader section
uploaded_files = st.file_uploader(
    "Upload MP3 files", 
    type=["mp3"], 
    accept_multiple_files=True,
    help="Only MP3 files are supported"
)

# Process uploaded files
if uploaded_files:
    new_files = []
    for uploaded_file in uploaded_files:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Add to session state if not already there
        file_info = {
            "name": uploaded_file.name,
            "path": temp_path,
            "size": uploaded_file.size / (1024 * 1024)  # Size in MB
        }
        
        # Check if this file is already in our list (by name)
        if not any(f["name"] == file_info["name"] for f in st.session_state.files):
            new_files.append(file_info)
            st.session_state.files.append(file_info)
    
    if new_files:
        st.success(f"Added {len(new_files)} new file(s)")

# Display selected files
if st.session_state.files:
    st.subheader("Selected Files")
    
    # Create columns for file display
    cols = st.columns([3, 1, 1])
    cols[0].write("**Filename**")
    cols[1].write("**Size (MB)**")
    cols[2].write("**Action**")
    
    # List files with remove buttons
    files_to_remove = []
    for i, file_info in enumerate(st.session_state.files):
        cols = st.columns([3, 1, 1])
        cols[0].write(file_info["name"])
        cols[1].write(f"{file_info['size']:.2f}")
        if cols[2].button("Remove", key=f"remove_{i}"):
            files_to_remove.append(i)
    
    # Remove files (in reverse order to avoid index shifting)
    for i in sorted(files_to_remove, reverse=True):
        # Remove the temporary file
        try:
            os.unlink(st.session_state.files[i]["path"])
        except:
            pass
        # Remove from session state
        st.session_state.files.pop(i)
    
    # Clear all button
    if st.button("Clear All Files"):
        # Remove all temporary files
        for file_info in st.session_state.files:
            try:
                os.unlink(file_info["path"])
            except:
                pass
        st.session_state.files = []
        st.rerun()

# Process button
if st.session_state.files and not st.session_state.processing:
    if st.button("Process Files"):
        st.session_state.processing = True
        st.session_state.results = []  # Clear previous results
        st.rerun()

# Processing logic
if st.session_state.processing:
    st.subheader("Processing Files")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load Whisper model if not already loaded
    if st.session_state.whisper_model is None:
        status_text.text("Loading Whisper model...")
        progress_bar.progress(10)
        # Explicitly use FP32 to avoid warnings on CPU
        whisper = load_whisper()
        st.session_state.whisper_model = whisper.load_model(MODEL_NAME, device="cpu")
        progress_bar.progress(20)
    
    # Process each file
    total_files = len(st.session_state.files)
    all_results = []
    
    # Calculate progress steps
    progress_per_file = 80 / total_files  # 80% of progress bar for processing files
    base_progress = 20  # Starting at 20% after model loading
    
    for i, file_info in enumerate(st.session_state.files):
        file_path = file_info["path"]
        file_name = file_info["name"]
        title = clean_filename(os.path.splitext(file_name)[0])
        
        # Calculate current progress
        current_progress = base_progress + (i * progress_per_file)
        progress_bar.progress(int(current_progress))
        status_text.text(f"Processing file {i+1}/{total_files}: {file_name}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"Error: File not found: {file_path}")
            continue
        
        # Display file info
        st.write(f"File: {file_name}")
        st.write(f"Size: {file_info['size']:.2f} MB")
        
        # Transcribe
        try:
            status_text.text(f"Transcribing {file_name}...")
            # Explicitly use FP32 to avoid warnings
            result = st.session_state.whisper_model.transcribe(file_path, fp16=False)
            transcript = result["text"]
            
            # Save transcript
            txt_path = os.path.join(TRANSCRIPT_DIR, f"{title}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            progress_bar.progress(int(current_progress + (progress_per_file * 0.6)))  # 60% of file progress
        except Exception as e:
            st.error(f"Error transcribing {file_name}: {str(e)}")
            continue
        
        # Summarize
        try:
            status_text.text(f"Summarizing {file_name}...")
            model_name = HF_MODELS["bart"]
            summary = summarize_hf(transcript, model_name)
            progress_bar.progress(int(current_progress + (progress_per_file * 0.9)))  # 90% of file progress
        except Exception as e:
            st.error(f"Error summarizing {file_name}: {str(e)}")
            continue
        
        # Add to results
        entry = {
            "filename": file_name,
            "transcript_path": txt_path,
            "transcript": transcript[:500] + "..." if len(transcript) > 500 else transcript,  # Preview
            "summaries": {
                "bart": {
                    "summary": summary,
                    "time": time.time()
                }
            }
        }
        all_results.append(entry)
        st.session_state.results.append(entry)
        
        # Complete this file's progress
        progress_bar.progress(int(base_progress + ((i + 1) * progress_per_file)))
    
    # Save all results to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    status_text.text("Processing complete!")
    progress_bar.progress(100)
    
    # Reset processing flag
    st.session_state.processing = False

# Display results
if st.session_state.results:
    st.subheader("Results")
    
    for i, result in enumerate(st.session_state.results):
        with st.expander(f"File: {result['filename']}"):
            st.write("**Transcript Preview:**")
            st.text(result["transcript"])
            
            st.write("**Summary:**")
            st.write(result["summaries"]["bart"]["summary"])
            
            # Add a download button for the transcript
            with open(result["transcript_path"], "r", encoding="utf-8") as f:
                transcript_data = f.read()
            st.download_button(
                label="Download Full Transcript",
                data=transcript_data,
                file_name=f"{os.path.splitext(result['filename'])[0]}_transcript.txt",
                mime="text/plain"
            )

# Add information about saved files
st.sidebar.header("Information")
st.sidebar.info(
    f"""
    - Transcripts are saved in: `{os.path.abspath(TRANSCRIPT_DIR)}`
    - Summaries are saved in: `{os.path.abspath(OUTPUT_JSON)}`
    """
)

# Add model information
st.sidebar.header("Model Information")
st.sidebar.write(f"**Whisper Model:** {MODEL_NAME}")
st.sidebar.write(f"**Summarization Model:** {HF_MODELS['bart']}")