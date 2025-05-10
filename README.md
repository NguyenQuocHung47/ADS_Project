# Podcast Transcription and Summarization

This application allows you to transcribe and summarize podcast audio files. It provides a user-friendly interface for adding podcast links (URLs or local file paths), validating them, transcribing the content, and generating summaries.

## Features

- Add multiple podcast links (URLs or local file paths) for batch processing
- Download podcasts from URLs automatically
- Validate audio files (checks if they are valid audio files)
- Check if audio files are long enough to be considered podcasts (minimum 2 minutes)
- Transcribe audio files using OpenAI's Whisper model
- Summarize transcripts using HuggingFace's BART model
- Display and save results

## Requirements

- Python 3.7+
- FFmpeg (required for audio processing)
- Required Python packages (see requirements.txt)

## Installation

1. **Install FFmpeg**:
   FFmpeg is required for audio processing. Install it based on your operating system:
   
   - **Windows**:
     - Download from https://ffmpeg.org/download.html (choose a Windows build)
     - Extract the ZIP file to a location like `C:\ffmpeg`
     - Add the `bin` folder to your PATH environment variable:
       - Right-click on "This PC" or "My Computer" → Properties → Advanced system settings → Environment Variables
       - Edit the "Path" variable and add the path to the bin folder (e.g., `C:\ffmpeg\bin`)
     - Restart your command prompt or IDE
   
   - **macOS**:
     ```
     brew install ffmpeg
     ```
   
   - **Linux**:
     ```
     sudo apt update
     sudo apt install ffmpeg
     ```

2. **Create a Python Virtual Environment** (recommended):
   ```
   python -m venv venv
   ```
   
   Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Install Required Python Packages**:
   ```
   pip install -r requirements.txt
   ```
   
   Note: The first time you run this, it will download several large models which may take some time depending on your internet connection.

4. **GPU Acceleration** (optional):
   If you have a compatible NVIDIA GPU, you can enable GPU acceleration for faster processing:
   - Uncomment the GPU-specific torch line in requirements.txt
   - Reinstall with: `pip install -r requirements.txt`

## Usage

1. Run the application:
   ```
   python podcast_ui.py
   ```

2. Add audio files using one of these methods:
   - **Drag and drop** audio files directly into the drop zone
   - **Click** on the drop zone to open a file selection dialog
   - You can add multiple files at once

3. Click "Process Files" to start transcribing and summarizing the selected audio files.

4. Review the results in the application window. Transcripts and summaries are also saved to disk.

## File Structure

- `podcast_ui.py`: Main application file with the user interface
- `download_podcast.py`: Script for downloading podcasts from RSS feeds
- `transcribe_podcast_to_text.py`: Script for transcribing audio files
- `summarize_podcast.py`: Script for summarizing transcripts
- `podcasts/`: Directory where downloaded audio files are saved
- `transcripts/`: Directory where transcripts are saved
- `output/`: Directory where summaries are saved as JSON

## Models

- Transcription: OpenAI's Whisper (base model)
- Summarization: Facebook's BART-large-CNN

## Note

- Processing large audio files may take significant time and computational resources
- The application validates that audio files are at least 2 minutes long to be considered podcasts
- You must confirm before processing to avoid accidental processing of large files