import os
import re
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import whisper
import json
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tkinterdnd2  # For drag and drop functionality

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

# Import functions from existing files
def clean_filename(name):
    """Clean filename from download_podcast.py"""
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(" ", "_")

def summarize_hf(text, model_name):
    """Summarize text using HuggingFace models from summarize_podcast.py"""
    try:
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

class PodcastTranscriptApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Podcast Transcription and Summarization")
        self.root.geometry("900x700")  # Increased window size
        self.root.resizable(True, True)
        
        self.files = []
        self.whisper_model = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Drop zone section
        drop_frame = ttk.LabelFrame(main_frame, text="Drag and Drop Audio Files Here", padding="10")
        drop_frame.pack(fill=tk.X, pady=10)
        
        # Create a drop zone
        self.drop_zone = tk.Canvas(drop_frame, bg="#f0f0f0", height=100)
        self.drop_zone.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add text to the drop zone
        self.drop_zone.create_text(
            400, 50, 
            text="Drag and drop audio files here\nor click to select files", 
            fill="gray", font=("Arial", 12)
        )
        
        # Configure the drop zone for drag and drop
        self.drop_zone.drop_target_register(tkinterdnd2.DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.drop_files)
        
        # Bind click event to the drop zone
        self.drop_zone.bind("<Button-1>", self.select_files)
        
        # Files list section
        files_frame = ttk.LabelFrame(main_frame, text="Selected Files", padding="10")
        files_frame.pack(fill=tk.X, pady=10)  # Changed from fill=tk.BOTH, expand=True to fill=tk.X
        
        # Create a frame to hold the listbox and buttons side by side
        files_content_frame = ttk.Frame(files_frame)
        files_content_frame.pack(fill=tk.X)
        
        # Left side: Files listbox with scrollbar
        files_list_frame = ttk.Frame(files_content_frame)
        files_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for files list
        scrollbar = ttk.Scrollbar(files_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Files listbox - with fixed height to make it smaller
        self.files_listbox = tk.Listbox(files_list_frame, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE, height=5)  # Set fixed height
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.files_listbox.yview)
        
        # Right side: Buttons for files management
        files_btn_frame = ttk.Frame(files_content_frame)
        files_btn_frame.pack(side=tk.RIGHT, padx=5)
        
        remove_btn = ttk.Button(files_btn_frame, text="Remove Selected", command=self.remove_selected)
        remove_btn.pack(pady=2)
        
        clear_btn = ttk.Button(files_btn_frame, text="Clear All", command=self.clear_files)
        clear_btn.pack(pady=2)
        
        # Process section
        process_frame = ttk.Frame(main_frame, padding="10")
        process_frame.pack(fill=tk.X, pady=10)
        
        self.process_btn = ttk.Button(process_frame, text="Process Files", command=self.confirm_process)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(process_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(process_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=5)
        
        # Results section - made larger
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbar for results text
        results_scrollbar = ttk.Scrollbar(results_frame)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Results text widget - with increased height
        self.results_text = tk.Text(results_frame, yscrollcommand=results_scrollbar.set, wrap=tk.WORD, height=15)  # Increased height
        self.results_text.pack(fill=tk.BOTH, expand=True)
        results_scrollbar.config(command=self.results_text.yview)
    
    def drop_files(self, event):
        """Handle files dropped onto the drop zone"""
        files = self.parse_drop_data(event.data)
        self.add_files(files)
    
    def parse_drop_data(self, data):
        """Parse the data from a drop event"""
        self.update_results(f"Raw drop data: {data}\n")
        
        # Handle different formats of drop data
        files = []
        
        if data.startswith('{'):
            # Windows format with curly braces
            files = data.strip('{}').split('} {')
        elif data.startswith('[') and data.endswith(']'):
            # List format
            import ast
            try:
                files = ast.literal_eval(data)
            except:
                files = [data]
        else:
            # Try to handle other formats
            if ' ' in data:
                # Space-separated list
                files = data.split()
            else:
                # Single file
                files = [data]
        
        # Clean up file paths
        cleaned_files = []
        for file in files:
            # Remove quotes if present
            if isinstance(file, str):
                if file.startswith('"') and file.endswith('"'):
                    file = file[1:-1]
                elif file.startswith("'") and file.endswith("'"):
                    file = file[1:-1]
            cleaned_files.append(file)
        
        self.update_results(f"Parsed files: {cleaned_files}\n")
        return cleaned_files
    
    def select_files(self, event=None):
        """Open file dialog to select audio files"""
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=filetypes
        )
        
        if files:
            self.add_files(files)
    
    def add_files(self, files):
        """Add files to the list"""
        for file in files:
            # Only check if file exists and is not already in the list
            if not os.path.isfile(file):
                messagebox.showwarning("File Not Found", f"The file {os.path.basename(file)} does not exist.")
                continue
                
            # Check if it's an MP3 file (same as in transcribe_podcast_to_text.py)
            if not file.lower().endswith('.mp3'):
                messagebox.showwarning("Not an MP3 File", f"{os.path.basename(file)} is not an MP3 file. Only MP3 files are supported.")
                continue
                
            if file not in self.files:
                self.files.append(file)
                self.files_listbox.insert(tk.END, os.path.basename(file))
    
    # Removed is_audio_file method as it's no longer needed
    
    def remove_selected(self):
        """Remove selected files from the list"""
        selected_indices = self.files_listbox.curselection()
        if not selected_indices:
            return
        
        # Remove in reverse order to avoid index shifting
        for i in sorted(selected_indices, reverse=True):
            del self.files[i]
            self.files_listbox.delete(i)
    
    def clear_files(self):
        """Clear all files"""
        self.files = []
        self.files_listbox.delete(0, tk.END)
    
    def confirm_process(self):
        """Confirm before processing files"""
        if not self.files:
            messagebox.showwarning("No Files", "Please add audio files first.")
            return
        
        result = messagebox.askyesno(
            "Confirm Processing", 
            f"Process {len(self.files)} file(s)?\n\nThis will transcribe and summarize the selected audio files."
        )
        
        if result:
            self.process_files()
    
    def process_files(self):
        """Process the files in a separate thread"""
        self.process_btn.config(state=tk.DISABLED)
        self.status_var.set("Starting...")
        self.progress_var.set(0)
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        
        # Start processing thread
        thread = threading.Thread(target=self.process_files_thread)
        thread.daemon = True
        thread.start()
    
    def process_files_thread(self):
        """Thread function to process files"""
        try:
            # Load Whisper model if not already loaded
            if self.whisper_model is None:
                self.update_status("Loading Whisper model...")
                self.update_progress(10)  # Show some progress while loading model
                # Explicitly use FP32 to avoid warnings on CPU
                self.whisper_model = whisper.load_model(MODEL_NAME, device="cpu")
                self.update_progress(20)  # Model loaded
            
            total_files = len(self.files)
            all_results = []
            
            # Calculate progress steps
            progress_per_file = 80 / total_files  # 80% of progress bar for processing files
            base_progress = 20  # Starting at 20% after model loading
            
            for i, file_path in enumerate(self.files):
                file_name = os.path.basename(file_path)
                title = clean_filename(os.path.splitext(file_name)[0])
                
                # Calculate current progress
                current_progress = base_progress + (i * progress_per_file)
                self.update_progress(current_progress)
                self.update_status(f"Processing file {i+1}/{total_files}: {file_name}")
                
                # Debug file path
                self.update_results(f"Processing file: {file_path}\n")
                
                # Check if file exists
                if not os.path.exists(file_path):
                    self.update_results(f"Error: File not found: {file_path}\n\n")
                    continue
                
                # Skip length checking - just log file info
                self.update_status(f"Processing {file_name}...")
                try:
                    # Just get file info for logging purposes
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    self.update_results(f"File: {file_name}\nSize: {file_size:.2f} MB\n")
                    self.update_progress(current_progress + (progress_per_file * 0.2))  # 20% of file progress
                except Exception as e:
                    self.update_results(f"Warning: Could not get file info for {file_name}: {str(e)}\n")
                    # Continue anyway
                
                # Transcribe
                try:
                    self.update_status(f"Transcribing {file_name}...")
                    # Explicitly use FP32 to avoid warnings
                    result = self.whisper_model.transcribe(file_path, fp16=False)
                    transcript = result["text"]
                    
                    # Save transcript
                    txt_path = os.path.join(TRANSCRIPT_DIR, f"{title}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    self.update_progress(current_progress + (progress_per_file * 0.6))  # 60% of file progress
                except Exception as e:
                    self.update_results(f"Error transcribing {file_name}: {str(e)}\n\n")
                    continue
                
                # Summarize
                try:
                    self.update_status(f"Summarizing {file_name}...")
                    model_name = HF_MODELS["bart"]
                    summary = summarize_hf(transcript, model_name)
                    self.update_progress(current_progress + (progress_per_file * 0.9))  # 90% of file progress
                except Exception as e:
                    self.update_results(f"Error summarizing {file_name}: {str(e)}\n\n")
                    continue
                
                # Add to results
                entry = {
                    "filename": file_name,
                    "transcript_path": txt_path,
                    "summaries": {
                        "bart": {
                            "summary": summary,
                            "time": time.time()
                        }
                    }
                }
                all_results.append(entry)
                
                # Display results
                self.update_results(f"File: {file_name}\n")
                self.update_results(f"Summary: {summary}\n\n")
                
                # Complete this file's progress
                self.update_progress(base_progress + ((i + 1) * progress_per_file))
            
            # Save all results to JSON
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            self.update_status("Processing complete!")
            self.update_progress(100)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_results(f"An error occurred: {str(e)}\n")
            import traceback
            self.update_results(f"Traceback: {traceback.format_exc()}\n")
        
        finally:
            # Re-enable the process button
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
    
    def get_audio_duration(self, file_path):
        """Get the duration of an audio file in seconds - simplified approach"""
        try:
            # Check if file exists
            if not os.path.isfile(file_path):
                raise Exception(f"File not found: {file_path}")
                
            # Log file details for debugging
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            self.update_results(f"File info: {file_path}\nSize: {file_size:.2f} MB\n")
            
            # Instead of trying to get actual duration, we'll estimate based on file size
            # Assuming average MP3 bitrate of 128 kbps (16 KB/s)
            # This is a rough estimate: 1 MB ≈ 60 seconds at 128 kbps
            estimated_duration = file_size * 60  # Rough estimate in seconds
            
            self.update_results(f"Estimated duration based on file size: {estimated_duration:.2f} seconds\n")
            return estimated_duration
        except Exception as e:
            self.update_results(f"Error checking file: {str(e)}\n")
            # Return a default duration to allow processing to continue
            return 180  # Default to 3 minutes
    
    def update_status(self, message):
        """Update status label from a thread"""
        self.root.after(0, lambda: self.status_var.set(message))
    
    def update_progress(self, value):
        """Update progress bar from a thread"""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def update_results(self, text):
        """Update results text from a thread"""
        self.root.after(0, lambda: self.results_text.insert(tk.END, text))

if __name__ == "__main__":
    # Check if ffmpeg is in PATH
    try:
        # Try to add ffmpeg to PATH if it exists in common locations
        ffmpeg_paths = [
            r"C:\Users\Admin\ffmpeg-7.1.1-essentials_build\bin",
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin"
        ]
        
        for path in ffmpeg_paths:
            if os.path.exists(path):
                os.environ["PATH"] += os.pathsep + path
                break
    except:
        pass
    
    # Use tkinterdnd2 for drag and drop functionality
    root = tkinterdnd2.TkinterDnD.Tk()
    app = PodcastTranscriptApp(root)
    root.mainloop()