import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
import re
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# API clients
from mistralai.client import MistralClient as Mistral
from groq import Groq

# YouTube transcript API
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    TooManyRequests
)

import numpy as np
import time
from dataclasses import dataclass
from mistralai import Mistral
import yt_dlp
import tempfile
import logging

# Get the absolute path to the assets directory
ASSETS_DIR = Path(__file__).parent / "assets"
if not ASSETS_DIR.exists():
    ASSETS_DIR.mkdir(exist_ok=True)

# Logo path handling
LOGO_PATH = ASSETS_DIR / "ytwhisperer-logo.png"

@dataclass
class TranscriptChunk:
    text: str
    start_time: float
    end_time: float

# Load environment variables
load_dotenv()

# Get Mistral API key from environment or secrets
def get_mistral_api_key():
    # Try to get from Streamlit secrets first (for cloud deployment)
    if hasattr(st.secrets, "MISTRAL_API_KEY"):
        return st.secrets.MISTRAL_API_KEY
    # Fallback to environment variable (for local development)
    return os.environ.get("MISTRAL_API_KEY", "")

def init_mistral_client():
    # Get API key from session state (UI input) or from secrets/env
    api_key = st.session_state.get("mistral_api_key_input") or get_mistral_api_key()
    if not api_key:
        return None
    
    # Basic API key validation - less restrictive pattern
    if not isinstance(api_key, str) or len(api_key.strip()) < 32:
        st.error("Invalid API key format. API key should be at least 32 characters long.")
        return None
    
    try:
        client = Mistral(api_key=api_key.strip())
        # Store the working API key in session state
        st.session_state.mistral_api_key = api_key.strip()
        return client
    except Exception as e:
        st.error(f"Error initializing Mistral client: {str(e)}")
        return None

# Get Groq API key from various sources
def get_groq_api_key():
    """Get Groq API key from various sources."""
    # First try environment variable
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key

    # Then try Streamlit secrets
    if hasattr(st.secrets, "GROQ_API_KEY"):
        return st.secrets.GROQ_API_KEY
    
    # Finally, try session state (UI input)
    return st.session_state.get("groq_api_key_input", "")

def init_groq_client():
    """Initialize Groq client with API key."""
    api_key = get_groq_api_key()
    if not api_key:
        return None
    
    if not isinstance(api_key, str) or len(api_key.strip()) < 32:
        st.error("Invalid Groq API key format. API key should be at least 32 characters long.")
        return None
    
    try:
        client = Groq(api_key=api_key.strip())
        # Store the working API key in session state
        st.session_state.groq_api_key = api_key.strip()
        return client
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None

# Initialize API keys from environment if not already set
if "mistral_api_key" not in st.session_state:
    st.session_state.mistral_api_key = get_mistral_api_key()
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = get_groq_api_key()

# Security configurations
if "api_call_limit" not in st.session_state:
    st.session_state.api_call_limit = 100  # Adjust based on your needs
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "last_api_call_reset" not in st.session_state:
    st.session_state.last_api_call_reset = datetime.now()

# Rate limiting function
def check_rate_limit():
    # Reset counter every hour
    if datetime.now() - st.session_state.last_api_call_reset > timedelta(hours=1):
        st.session_state.api_call_count = 0
        st.session_state.last_api_call_reset = datetime.now()
    
    if st.session_state.api_call_count >= st.session_state.api_call_limit:
        remaining_time = st.session_state.last_api_call_reset + timedelta(hours=1) - datetime.now()
        st.error(f"Rate limit exceeded. Please try again in {remaining_time.seconds // 60} minutes.")
        return False
    
    st.session_state.api_call_count += 1
    return True

# Initialize session state and results placeholder
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []
if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "results_placeholder" not in st.session_state:
    st.session_state.results_placeholder = st.empty()
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "mistral_api_key" not in st.session_state:
    st.session_state.mistral_api_key = get_mistral_api_key()
if "mistral_api_key_input" not in st.session_state:
    st.session_state.mistral_api_key_input = ""
if "mistral_client" not in st.session_state:
    st.session_state.mistral_client = init_mistral_client()
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None
if "using_whisper" not in st.session_state:
    st.session_state.using_whisper = False

# Initialize or update client if needed
def get_or_create_client():
    if st.session_state.mistral_client is None:
        st.session_state.mistral_client = init_mistral_client()
    return st.session_state.mistral_client

# Secure the API key display in session state
def secure_api_key(key):
    if not key:
        return ""
    return f"{key[:4]}...{key[-4:]}"

# Translations dictionary
translations = {
    "en": {
        "title": "YTWhisperer",
        "controls": "Settings",
        "enter_url": "Enter YouTube URL:",
        "generate_summary": "Generate Summary",
        "show_transcript": "Show Transcript",
        "clear_history": "Clear Chat History",
        "video_summary": "Video Summary",
        "full_transcript": "Full Transcript",
        "chat_title": "Video Q&A Chat",
        "ask_placeholder": "Ask anything about the video...",
        "transcript_loaded": "✅ Transcript successfully loaded!",
        "generating_embeddings": "🔄 Generating embeddings... This may take a few minutes.",
        "processing_batch": "Processing batch {} of {}...",
        "embeddings_done": "✅ Embeddings successfully generated! You can now ask questions about the video.",
        "invalid_url": "Invalid YouTube URL",
        "no_transcript_using_whisper": "No transcript found. Using Whisper for transcription..."
    },
    "hr": {
        "title": "YTWhisperer",
        "controls": "Postavke",
        "enter_url": "Unesite YouTube URL:",
        "generate_summary": "Generiraj Sažetak",
        "show_transcript": "Prikaži Transkript",
        "clear_history": "Očisti Povijest Chata",
        "video_summary": "Sažetak Videa",
        "full_transcript": "Potpuni Transkript",
        "chat_title": "Video Q&A Chat",
        "ask_placeholder": "Pitajte bilo što o videu...",
        "transcript_loaded": "✅ Transkript uspješno učitan!",
        "generating_embeddings": "🔄 Generiranje embeddings-a... Ovo može potrajati nekoliko minuta.",
        "processing_batch": "Obrađujem grupu {} od {}...",
        "embeddings_done": "✅ Embeddings uspješno generirani! Sada možete postavljati pitanja o videu.",
        "invalid_url": "Nevažeći YouTube URL",
        "no_transcript_using_whisper": "Nema transkripta. Koristim Whisper za transkripciju..."
    }
}

def get_text(key):
    return translations[st.session_state.language][key]

# Helper functions
def extract_video_id(url):
    # List of possible YouTube URL patterns
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # Shortened youtu.be URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URLs
        r'^([0-9A-Za-z_-]{11})$'  # Direct video ID
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error(get_text("invalid_url"))
            return None
            
        # Don't fetch thumbnail URL to avoid tracking
        return {
            "id": video_id
        }
    except Exception as e:
        st.error(f"Error getting video info: {e}")
        return None

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def process_transcript(transcript_list: List[Dict]) -> str:
    """Process transcript and store chunks with timestamps"""
    st.session_state.raw_chunks = []
    current_chunk = []
    current_text = []
    chunk_start_time = transcript_list[0]['start']
    
    for entry in transcript_list:
        current_text.append(entry['text'])
        current_chunk.append(entry)
        
        # Create a new chunk every ~2000 characters
        if len(' '.join(current_text)) >= 2000:
            chunk_text = ' '.join(current_text)
            chunk_end_time = entry['start'] + entry['duration']
            st.session_state.raw_chunks.append(
                TranscriptChunk(
                    text=chunk_text,
                    start_time=chunk_start_time,
                    end_time=chunk_end_time
                )
            )
            current_text = []
            current_chunk = []
            chunk_start_time = chunk_end_time
    
    # Add the last chunk if there's any remaining text
    if current_text:
        chunk_text = ' '.join(current_text)
        last_entry = transcript_list[-1]
        st.session_state.raw_chunks.append(
            TranscriptChunk(
                text=chunk_text,
                start_time=chunk_start_time,
                end_time=last_entry['start'] + last_entry['duration']
            )
        )
    
    return ' '.join(entry['text'] for entry in transcript_list)

def get_relevant_chunks(similarities: List[float], top_k: int = 3) -> List[TranscriptChunk]:
    """Get the most relevant transcript chunks based on similarity scores"""
    chunk_scores = list(enumerate(similarities))
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    relevant_chunks = []
    for idx, score in chunk_scores[:top_k]:
        relevant_chunks.append(st.session_state.raw_chunks[idx])
    
    # Sort chunks by timestamp
    relevant_chunks.sort(key=lambda x: x.start_time)
    return relevant_chunks

def split_text(text, chunk_size=2000):  
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def batch_list(lst, batch_size=32):
    """Split a list into batches"""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def get_embeddings_with_retry(client, texts, max_retries=3, initial_delay=2):
    """Get embeddings with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                delay = initial_delay * (attempt + 1)  # Exponential backoff
                time.sleep(delay)
            else:
                raise e
    return None

class StreamlitLogger:
    """Custom logger for yt-dlp that uses Streamlit for output."""
    def debug(self, msg):
        if msg.startswith('[download]'):
            st.write(msg)
    
    def warning(self, msg):
        st.warning(msg)
    
    def error(self, msg):
        st.error(msg)

def download_and_convert_to_mp3(video_url: str, output_path: str) -> str:
    """Download YouTube video and convert it to MP3."""
    try:
        # Ensure output path exists
        os.makedirs(output_path, exist_ok=True)
        
        # Create full path for the output file
        mp3_path = os.path.join(output_path, "audio.mp3")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': os.path.join(output_path, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'logger': StreamlitLogger(),
            'progress_hooks': [],
            'quiet': False
        }
        
        try:
            # First, try to extract video info to verify URL is valid
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                st.info("Extracting video information...")
                info = ydl.extract_info(video_url, download=False)
                st.info(f"Found video: {info.get('title', 'Unknown title')}")
                
                # Now download the video
                st.info("Starting download and conversion...")
                ydl.download([video_url])
        except Exception as e:
            st.error(f"yt-dlp error: {str(e)}")
            return None
        
        # Verify file exists
        if not os.path.exists(mp3_path):
            # Check if the file exists with a different extension
            potential_files = [f for f in os.listdir(output_path) if f.startswith('audio.')]
            if potential_files:
                st.error(f"Found files but not mp3: {potential_files}")
            else:
                st.error(f"No audio files found in {output_path}")
            return None
            
        st.success("Audio file created successfully")
        return mp3_path
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        # List directory contents for debugging
        try:
            st.error(f"Directory contents: {os.listdir(output_path)}")
        except OSError as e:
            st.error(f"Could not list directory contents: {str(e)}")
        return None

def call_groq_whisper(mp3_file):
    """
    Placeholder function for calling Groq's Whisper API.
    Replace this implementation with the actual API call to get the transcript.
    """
    # For now, return a dummy transcript. In production, replace this placeholder logic.
    return "This is a dummy transcript generated from " + mp3_file

def whisper_transcription(video_file, max_retries=3):
    """
    Uses Groq API to perform Whisper transcription on the given video.
    Converts video to MP3 using ffmpeg and attempts transcription up to max_retries.
    Returns the transcript as a string if successful, or None otherwise.
    """
    import subprocess
    import time
    transcript = None
    for attempt in range(max_retries):
        try:
            # Convert video to MP3
            mp3_file = video_file.rsplit('.', 1)[0] + ".mp3"
            # Using ffmpeg to convert video to mp3
            subprocess.run(["ffmpeg", "-i", video_file, mp3_file], check=True)
            
            # Placeholder for calling Groq's Whisper API
            # Replace the following line with the actual API call to get the transcript
            transcript = call_groq_whisper(mp3_file)
            
            if transcript and transcript.strip():
                logging.info(f"Whisper transcription succeeded on attempt {attempt+1}")
                return transcript
            else:
                logging.error(f"Whisper transcription attempt {attempt+1} returned empty transcript.")
        except Exception as e:
            logging.error(f"Whisper transcription attempt {attempt+1} failed with error: {e}")
        time.sleep(2)  # wait before retrying
    return transcript

def transcribe_with_whisper(audio_file_path: str, language: str = None) -> List[Dict]:
    """Transcribe audio using Groq's Whisper API."""
    client = init_groq_client()
    if not client:
        st.error("Failed to initialize Groq client. Please check your API key.")
        return None
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            with open(audio_file_path, "rb") as audio_file:
                st.info(f"Attempt {attempt + 1}/{max_retries}: Sending audio file to Groq API...")
                response = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json"
                )
                
                # Handle different response types: dict or object with segments attribute
                if isinstance(response, dict):
                    segments = response.get("segments")
                elif hasattr(response, "segments"):
                    segments = response.segments
                else:
                    st.error(f"Unexpected response type: {type(response)}")
                    if attempt < max_retries - 1:
                        st.warning(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    return None

                if not segments:
                    error_msg = str(response) if response else "Empty response"
                    st.error(f"Unexpected response format: {error_msg}")
                    if attempt < max_retries - 1:
                        st.warning(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    return None

                # Convert Whisper response to match YouTube transcript format
                transcript = []
                for segment in segments:
                    if isinstance(segment, dict):
                        text = segment.get("text")
                        start = segment.get("start")
                        end = segment.get("end")
                    else:
                        text = getattr(segment, "text", None)
                        start = getattr(segment, "start", None)
                        end = getattr(segment, "end", None)
                    if text is None or start is None or end is None:
                        continue
                    transcript.append({
                        "text": text,
                        "start": start,
                        "duration": end - start
                    })

                st.success("Transcription completed successfully!")
                
                # Delete the audio file after transcription
                try:
                    os.remove(audio_file_path)
                    st.info("Audio file deleted successfully.")
                except Exception as del_e:
                    st.warning(f"Failed to delete audio file: {str(del_e)}")
                
                return transcript
                
        except Exception as e:
            error_msg = str(e)
            if "<!DOCTYPE html>" in error_msg:
                st.error("Groq API is temporarily unavailable (Error 520). This is a server-side issue.")
            else:
                st.error(f"Error during transcription: {error_msg}")
            
            if attempt < max_retries - 1:
                st.warning(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                st.error("Maximum retry attempts reached. Please try again later.")
                return None
    
    return None

def get_transcript(video_id):
    """Get transcript for a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        current_language = st.session_state.language
        
        # First try to get manual transcripts in current language
        try:
            transcript = transcript_list.find_manually_created_transcript([current_language])
            return transcript.fetch()
        except NoTranscriptFound:
            # Then try auto-generated transcripts in current language
            try:
                transcript = transcript_list.find_generated_transcript([current_language])
                return transcript.fetch()
            except NoTranscriptFound:
                # If not found in current language, try English as fallback
                if current_language != "en":
                    try:
                        # Try manual English transcript
                        transcript = transcript_list.find_manually_created_transcript(["en"])
                        return transcript.translate(current_language).fetch()
                    except NoTranscriptFound:
                        try:
                            # Try auto-generated English transcript
                            transcript = transcript_list.find_generated_transcript(["en"])
                            return transcript.translate(current_language).fetch()
                        except NoTranscriptFound:
                            pass
                
                # If still no transcript found, try Whisper
                st.warning(get_text("no_transcript_using_whisper"))
                return transcribe_with_whisper(video_id)
            
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video. Trying Whisper...")
        return transcribe_with_whisper(video_id)
    except VideoUnavailable:
        st.error("Video is unavailable. Please check the URL.")
        return None
    except TooManyRequests:
        st.error("Too many requests. Please try again later.")
        return None
    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None

# Main content area

# URL input in main area
video_url = st.text_input(get_text("enter_url"))

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        # Create columns to constrain video width
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Most basic embed possible
            st.markdown(
                f'<iframe src="https://www.youtube-nocookie.com/embed/{video_id}" width="100%" height="315" frameborder="0"></iframe>',
                unsafe_allow_html=True
            )
        
        try:
            # Try to get YouTube transcript first
            transcript_list = get_transcript(video_id)
            st.session_state.using_whisper = False
        except Exception:
            st.warning("YouTube transcript not available. Attempting to use Whisper transcription...")
            
            # Create temporary directory for audio processing
            with tempfile.TemporaryDirectory() as temp_dir:
                st.info("Downloading and converting video to audio...")
                # Download and convert video to MP3
                mp3_path = download_and_convert_to_mp3(video_url, temp_dir)
                
                if mp3_path and os.path.exists(mp3_path):
                    st.info("Starting Whisper transcription...")
                    # Transcribe using Whisper
                    transcript_list = transcribe_with_whisper(mp3_path, st.session_state.language)
                    if transcript_list:
                        st.session_state.using_whisper = True
                    else:
                        st.error("Failed to transcribe video using Whisper.")
                        st.stop()
                else:
                    st.error("Failed to process video for transcription.")
                    st.stop()
        
        if transcript_list:
            st.session_state.transcript_list = transcript_list  # Store the raw list
            st.session_state.transcript = ' '.join(entry['text'] for entry in transcript_list)  # Store joined text
            st.session_state.video_info = get_video_info(video_url)
            process_transcript(transcript_list)
            
            # Show transcription source
            if st.session_state.using_whisper:
                st.info("This transcription was generated using Groq's Whisper model.")
            
            st.success(get_text("transcript_loaded"))
            
            # Process transcript for embeddings
            chunks = [chunk.text for chunk in st.session_state.raw_chunks]
            st.session_state.chunks = chunks
            
            # Only generate embeddings if this is a new video
            if video_id != st.session_state.current_video_id:
                st.info(get_text("generating_embeddings"))
                
                # Initialize client here when needed
                client = get_or_create_client()
                if not client:
                    st.error("Failed to initialize Mistral client. Please check your API key.")
                    st.stop()
                
                # Generate embeddings for chunks in batches
                st.session_state.chunk_embeddings = []
                chunk_batches = batch_list(chunks, batch_size=4)  
                
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for batch_idx, batch in enumerate(chunk_batches):
                    try:
                        progress_text.text(get_text("processing_batch").format(batch_idx + 1, len(chunk_batches)))
                        embeddings = get_embeddings_with_retry(client, batch)
                        
                        if not embeddings:
                            st.error(f"Failed to generate embeddings for batch {batch_idx + 1}")
                            continue
                            
                        st.session_state.chunk_embeddings.extend(embeddings)
                        
                        # Update progress
                        progress = (batch_idx + 1) / len(chunk_batches)
                        progress_bar.progress(progress)
                        
                        # Add longer delay between batches
                        if batch_idx < len(chunk_batches) - 1:
                            time.sleep(3)  # 3 seconds delay between batches
                            
                    except Exception as e:
                        st.error(f"Error generating embeddings for batch {batch_idx + 1}: {str(e)}")
                        if "rate limit" in str(e).lower():
                            st.warning("Rate limit hit. Waiting 10 seconds before continuing...")
                            time.sleep(10)  # Longer wait on rate limit
                        continue
                
                progress_bar.empty()
                progress_text.empty()
                
                st.success(get_text("embeddings_done"))
                # Store the current video ID
                st.session_state.current_video_id = video_id
    else:
        st.error(get_text("invalid_url"))

# Create a placeholder for results
results_placeholder = st.empty()

# Sidebar
with st.sidebar:
    # Add logo at the top of the sidebar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading logo: {str(e)}")

    st.divider()
    
    # Show warning if no API key is set
    if not st.session_state.mistral_api_key:
        st.warning("Please enter your Mistral API key in Settings ⚙️")
        st.stop()
        
    # Action Buttons outside of settings
    if st.button(get_text("generate_summary"), 
                use_container_width=True, 
                disabled=not st.session_state.transcript):
        if not check_rate_limit():
            st.stop()
        
        with st.spinner("Generating summary..."):
            try:
                client = get_or_create_client()
                if not client:
                    st.error("Failed to initialize Mistral client. Please check your API key.")
                    st.stop()
                
                # Define language-specific templates
                templates = {
                    'hr': '''Kreiraj sažetak na hrvatskom jeziku sa sljedećim sekcijama:
                    
                    1. Kratki Pregled (2-3 rečenice)
                    2. Ključne Točke (natuknice)
                    3. Glavni Zaključci (2-3 ključna zaključka)
                    
                    Formatiraj odgovor u Markdown formatu za bolju čitljivost.''',
                    'en': '''Create a summary in English with these sections:
                    
                    1. Brief Overview (2-3 sentences)
                    2. Key Points (bullet points)
                    3. Main Takeaways (2-3 key conclusions)
                    
                    Format the response in Markdown for better readability.'''
                }
                
                messages = [
                    {"role": "system", "content": f"You are an AI assistant that creates clear, well-structured summaries of YouTube videos.\n{templates[st.session_state.language]}"},
                    {"role": "user", "content": f"Generate a summary in {st.session_state.language} language for this transcript: {st.session_state.transcript}"}
                ]
                
                response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                with results_placeholder.container():
                    st.header(f" 📝 {get_text('video_summary')}")
                    st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                if "rate limit" in str(e).lower():
                    st.warning("Rate limit hit. Please wait a moment and try again.")
    
    if st.button(get_text("show_transcript"), 
                use_container_width=True, 
                disabled=not st.session_state.transcript):
        with results_placeholder.container():
            st.header(f" 📄 {get_text('full_transcript')}")
            formatted_transcript = ""
            for chunk in st.session_state.raw_chunks:
                timestamp = f"**`[{format_timestamp(chunk.start_time)} - {format_timestamp(chunk.end_time)}]`**"
                formatted_transcript += f"{timestamp}\n\n{chunk.text}\n\n---\n\n"
            st.markdown(formatted_transcript)
    
    if st.button(get_text("clear_history"), 
                use_container_width=True, 
                disabled=not st.session_state.transcript):
        st.session_state.chat_history = []
        results_placeholder.empty()
        st.rerun()
    
    st.divider()
    
    # Settings Section (formerly Controls)
    with st.expander(" ⚙️ Settings", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Language", "API Keys", "Help"])
        
        # Tab 1: Language Settings
        with tab1:
            # Custom CSS for language selector
            st.markdown("""
                <style>
                .lang-text {
                    font-weight: 600;
                    font-size: 1.1em;
                    text-align: center;
                    padding: 5px;
                }
                .lang-active {
                    color: #ff4b4b;
                }
                /* Center toggle in its column */
                div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Create three equal columns with more space for the center
            col1, col2, col3 = st.columns([3, 4, 3])
            
            # Display EN with active state
            with col1:
                st.markdown(
                    f'<div class="lang-text {("lang-active" if st.session_state.language == "en" else "")}">EN</div>',
                    unsafe_allow_html=True
                )
            
            # Center the toggle switch
            with col2:
                previous_language = st.session_state.language
                is_croatian = st.toggle(
                    "Language",
                    value=st.session_state.language == "hr",
                    help="Toggle between English and Croatian",
                    label_visibility="collapsed",
                    key="language_toggle"
                )
            
            # Display HR with active state
            with col3:
                st.markdown(
                    f'<div class="lang-text {("lang-active" if st.session_state.language == "hr" else "")}">HR</div>',
                    unsafe_allow_html=True
                )
            
            # Update language based on toggle state, but preserve the session state
            new_language = "hr" if is_croatian else "en"
            if new_language != previous_language:
                st.session_state.language = new_language
                # Don't clear transcript or other important states
                # Just rerun to update the UI text
                st.rerun()
        
        # Tab 2: API Keys
        with tab2:
            # Mistral API Key
            st.subheader("Mistral API Key")
            if not st.session_state.mistral_api_key:
                st.text_input(
                    "Enter Mistral API Key",
                    type="password",
                    key="mistral_api_key_input",
                    help="Required for chat and embeddings functionality"
                )
                if st.session_state.mistral_api_key_input:
                    if init_mistral_client():
                        st.success("Mistral API Key saved!")
            else:
                # Check if the key is from environment
                env_key = os.environ.get("MISTRAL_API_KEY", "")
                is_env_key = st.session_state.mistral_api_key == env_key and env_key != ""
                source = "(from .env)" if is_env_key else "(from UI)"
                st.success(f"Mistral API Key configured {source}")
                if st.button("Clear Mistral API Key"):
                    st.session_state.mistral_api_key = None
                    st.session_state.mistral_api_key_input = ""
                    st.session_state.mistral_client = None
                    st.experimental_rerun()
            
            st.divider()
            
            # Groq API Key
            st.subheader("Groq API Key")
            if not st.session_state.groq_api_key:
                st.text_input(
                    "Enter Groq API Key",
                    type="password",
                    key="groq_api_key_input",
                    help="Required for Whisper transcription when YouTube subtitles are unavailable"
                )
                if st.session_state.groq_api_key_input:
                    if init_groq_client():
                        st.success("Groq API Key saved!")
            else:
                # Check if the key is from environment
                env_key = os.environ.get("GROQ_API_KEY", "")
                is_env_key = st.session_state.groq_api_key == env_key and env_key != ""
                source = "(from .env)" if is_env_key else "(from UI)"
                st.success(f"Groq API Key configured {source}")
                if st.button("Clear Groq API Key"):
                    st.session_state.groq_api_key = None
                    st.session_state.groq_api_key_input = ""
                    st.experimental_rerun()
            
            # Display rate limit information
            st.divider()
            st.subheader("Rate Limits")
            calls_remaining = st.session_state.api_call_limit - st.session_state.api_call_count
            st.write(f"API calls remaining: {calls_remaining}")
            if st.session_state.api_call_count > 0:
                reset_time = st.session_state.last_api_call_reset + timedelta(hours=1)
                st.write(f"Resets at: {reset_time.strftime('%H:%M:%S')}")
        
        # Tab 3: Help
        with tab3:
            st.markdown("""
            ### API Key Configuration
            You can set your API keys in two ways:
            1. In the `.env` file using:
               - `MISTRAL_API_KEY=your_key`
               - `GROQ_API_KEY=your_key`
            2. Directly in this UI (will override .env for this session)
            """)
        
        st.divider()

# Chat interface in main area
if st.session_state.transcript:
    st.divider()
    st.subheader(get_text("chat_title"))

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(get_text("ask_placeholder")):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if not check_rate_limit():
                st.stop()
                
            with st.spinner("Thinking..."):
                try:
                    # Get relevant context using embeddings
                    client = get_or_create_client()
                    response = client.embeddings.create(
                        model="mistral-embed",
                        inputs=[prompt]
                    )
                    q_embedding = response.data[0].embedding

                    similarities = []
                    for chunk_embedding in st.session_state.chunk_embeddings:
                        similarity = np.dot(q_embedding, chunk_embedding)
                        similarities.append(similarity)
                    
                    relevant_chunks = get_relevant_chunks(similarities)
                    
                    # Format context with timestamps
                    context = "\n\n".join([
                        f"[{format_timestamp(chunk.start_time)} - {format_timestamp(chunk.end_time)}] {chunk.text}"
                        for chunk in relevant_chunks
                    ])
                    
                    # Generate chat completion
                    messages = [
                        {"role": "system", "content": """You are an AI assistant analyzing a YouTube video. Use the following transcript excerpts with timestamps to answer the user's question:

""" + context + """

Important instructions for your response:
1. When referencing specific parts of the video, format timestamps in bold monospace like this: **`[MM:SS - MM:SS]`**
2. At the end of your response, always include:
   - A "Timestamp references:" section with timestamps in the same format: **`[MM:SS - MM:SS]`**
   - A "Confidence:" percentage (95% if very confident, lower if less certain)
3. Format your response in a clear, concise manner
4. If you're not confident about certain details, express that uncertainty"""},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = client.chat.complete(
                        model="mistral-large-latest",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    st.markdown(response.choices[0].message.content)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    if "rate limit" in str(e).lower():
                        st.warning("Rate limit hit. Please wait a moment and try again.")