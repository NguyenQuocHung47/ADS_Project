# Streamlit Podcast Transcription & Summarization App

This is the new Streamlit-based UI for the Podcast Transcription and Summarization application.

## Features

### Tab 1: Process Audio Files
- **Welcome Interface**: Greets users and explains the application features
- **File Upload**: Drag and drop or select multiple audio files (MP3, WAV, M4A, FLAC, etc.)
- **Real-time Processing**: Shows progress and results as files are processed
- **Dual Summarization**: Uses both HuggingFace BART and Cohere API for summarization
- **Results Display**: Shows summaries side-by-side with expandable transcript view

### Tab 2: API Performance Dashboard
- **Performance Metrics**: Total API calls, average response time, success rate, total processing time
- **Visual Analytics**: 
  - Response time distribution by API (box plot)
  - API usage distribution (pie chart)
  - Timeline of API calls (scatter plot)
- **Detailed Logging**: Filterable table of all API calls with timestamps
- **Data Management**: Clear performance data option

## APIs Monitored

1. **OpenAI Whisper**: Audio transcription
2. **HuggingFace BART**: Text summarization
3. **Cohere API**: Alternative text summarization

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
python -m streamlit run streamlit_app.py
```

## Usage

1. **Upload Files**: Go to the "Process Audio Files" tab and upload your audio files
2. **Process**: Click the "Process Files" button to start transcription and summarization
3. **View Results**: Results appear in real-time as files are processed
4. **Monitor Performance**: Switch to the "API Performance" tab to see detailed analytics

## File Structure

- `streamlit_app.py`: Main Streamlit application
- `podcast_ui.py`: Original tkinter-based UI (legacy)
- `transcripts/`: Generated transcripts are saved here
- `output/summaries.json`: JSON file containing all processing results
- `podcasts/`: Temporary storage for uploaded files during processing

## Configuration

- **Whisper Model**: Currently set to "base" model (can be changed in the code)
- **Cohere API Key**: Set in the code (should be moved to environment variables for production)
- **Summary Length**: 150 tokens for BART model
- **Max Input Length**: 1024 tokens for text processing

## Performance Features

The application tracks detailed performance metrics for all API calls:
- Response times
- Success/failure rates
- Input/output data sizes
- Timestamps for all operations
- Error logging

This allows users to monitor the efficiency and reliability of different AI services used in the application.