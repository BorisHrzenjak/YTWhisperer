import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
from mistralai import Mistral
import os
from dotenv import load_dotenv
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta
from pathlib import Path

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

# Security configurations
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "last_api_call_reset" not in st.session_state:
    st.session_state.last_api_call_reset = datetime.now()
if "api_call_limit" not in st.session_state:
    st.session_state.api_call_limit = 100  # Adjust based on your needs

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
        "controls": "Controls",
        "enter_url": "Enter YouTube URL:",
        "generate_summary": "Generate Summary",
        "show_metadata": "Show Metadata",
        "show_transcript": "Show Transcript",
        "clear_history": "Clear Chat History",
        "video_summary": "Video Summary",
        "video_metadata": "Video Metadata",
        "full_transcript": "Full Transcript",
        "chat_title": "Video Q&A Chat",
        "ask_placeholder": "Ask anything about the video...",
        "transcript_loaded": "✅ Transcript successfully loaded!",
        "generating_embeddings": "🔄 Generating embeddings... This may take a few minutes.",
        "processing_batch": "Processing batch {} of {}...",
        "embeddings_done": "✅ Embeddings successfully generated! You can now ask questions about the video.",
        "invalid_url": "Invalid YouTube URL"
    },
    "hr": {
        "title": "YTWhisperer",
        "controls": "Kontrole",
        "enter_url": "Unesite YouTube URL:",
        "generate_summary": "Generiraj Sažetak",
        "show_metadata": "Prikaži Metapodatke",
        "show_transcript": "Prikaži Transkript",
        "clear_history": "Očisti Povijest Chata",
        "video_summary": "Sažetak Videa",
        "video_metadata": "Metapodaci Videa",
        "full_transcript": "Potpuni Transkript",
        "chat_title": "Video Q&A Chat",
        "ask_placeholder": "Pitajte bilo što o videu...",
        "transcript_loaded": "✅ Transkript uspješno učitan!",
        "generating_embeddings": "🔄 Generiranje embeddings-a... Ovo može potrajati nekoliko minuta.",
        "processing_batch": "Obrađujem grupu {} od {}...",
        "embeddings_done": "✅ Embeddings uspješno generirani! Sada možete postavljati pitanja o videu.",
        "invalid_url": "Nevažeći YouTube URL"
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

# Main content area
st.title(get_text("title"))

# URL input in main area
url = st.text_input(get_text("enter_url"))

# Video section
if url:
    video_id = extract_video_id(url)
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
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if transcript:
                st.session_state.transcript = process_transcript(transcript)
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
        except Exception as e:
            st.error(f"Error loading transcript: {str(e)}")
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
            st.image(str(LOGO_PATH), width=150)
        except Exception:
            st.warning("Logo not found. Please check assets folder.")
    st.divider()
    
    # Language selection in a smaller format
    cols = st.columns([1, 1, 1])  # Three equal columns
    with cols[0]:
        st.write("🌍")
    
    # EN button with letters in rows
    if cols[1].button("E\nN", type="primary" if st.session_state.language == "en" else "secondary"):
        st.session_state.language = "en"
        st.rerun()
    
    # HR button with letters in rows
    if cols[2].button("H\nR", type="primary" if st.session_state.language == "hr" else "secondary"):
        st.session_state.language = "hr"
        st.rerun()
    
    st.title(get_text("controls"))
    st.divider()

    # Settings expander
    with st.expander("⚙️ Settings"):
        st.markdown("""
        ### API Key Configuration
        You can set your Mistral API key in two ways:
        1. In the `.env` file using `MISTRAL_API_KEY=your_key`
        2. Directly in this UI (will override .env for this session)
        """)
        
        # API Key input with secure display
        current_key_display = secure_api_key(st.session_state.mistral_api_key)
        source = "from .env" if st.session_state.mistral_api_key == os.environ.get("MISTRAL_API_KEY", "") else "from UI"
        st.write(f"Current API Key: {current_key_display} ({source})")
        
        api_key_input = st.text_input(
            "Mistral API Key",
            value=st.session_state.mistral_api_key_input,
            type="password",
            help="Enter your Mistral API key here if not using secrets.toml"
        )
        if api_key_input != st.session_state.mistral_api_key_input:
            st.session_state.mistral_api_key_input = api_key_input
            # Initialize client with new key
            if init_mistral_client():
                st.success("API key updated successfully!")
                st.rerun()

        # Display rate limit information
        st.write("---")
        st.write("Rate Limit Status:")
        calls_remaining = st.session_state.api_call_limit - st.session_state.api_call_count
        st.write(f"API calls remaining: {calls_remaining}")
        if st.session_state.api_call_count > 0:
            reset_time = st.session_state.last_api_call_reset + timedelta(hours=1)
            st.write(f"Resets at: {reset_time.strftime('%H:%M:%S')}")

    # Show warning if no API key is set
    if not st.session_state.mistral_api_key:
        st.warning("Please enter your Mistral API key in Settings ⚙️")
        st.stop()

    # Always show buttons, but disable them if no transcript
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
                    
                messages = [
                    {"role": "system", "content": """You are an AI assistant that creates clear, well-structured summaries of YouTube videos.
                    Create a summary with these sections:
                    
                    1. Brief Overview (2-3 sentences)
                    2. Key Points (bullet points)
                    3. Main Takeaways (2-3 key conclusions)
                    
                    Format the response in Markdown for better readability.
                    """},
                    {"role": "user", "content": st.session_state.transcript}
                ]
                
                response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                with results_placeholder.container():
                    st.header(f"📝 {get_text('video_summary')}")
                    st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                if "rate limit" in str(e).lower():
                    st.warning("Rate limit hit. Please wait a moment and try again.")
    
    if st.button(get_text("show_metadata"), 
                use_container_width=True, 
                disabled=not st.session_state.transcript):
        if not check_rate_limit():
            st.stop()
            
        with st.spinner("Extracting metadata..."):
            try:
                client = get_or_create_client()
                if not client:
                    st.error("Failed to initialize Mistral client. Please check your API key.")
                    st.stop()
                    
                messages = [
                    {"role": "system", "content": """Extract and organize key metadata from this video transcript.
                    Include:
                    
                    1. Main Topics Discussed
                    2. Key Terms and Concepts
                    3. Important Details and References
                    
                    Format the response in Markdown for better readability.
                    """},
                    {"role": "user", "content": st.session_state.transcript}
                ]
                
                response = client.chat.complete(
                    model="mistral-tiny",  # Using the more reliable model
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                with results_placeholder.container():
                    st.header(f"ℹ️ {get_text('video_metadata')}")
                    st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error generating metadata: {str(e)}")
                if "rate limit" in str(e).lower():
                    st.warning("Rate limit hit. Please wait a moment and try again.")
    
    if st.button(get_text("show_transcript"), 
                use_container_width=True, 
                disabled=not st.session_state.transcript):
        with results_placeholder.container():
            st.header(f"📄 {get_text('full_transcript')}")
            formatted_transcript = ""
            for chunk in st.session_state.raw_chunks:
                # Format timestamp in a visually distinct way
                timestamp = f"**`[{format_timestamp(chunk.start_time)} - {format_timestamp(chunk.end_time)}]`**"
                # Add timestamp and text with proper spacing and formatting
                formatted_transcript += f"{timestamp}\n\n{chunk.text}\n\n---\n\n"
            st.markdown(formatted_transcript)
    
    if st.button(get_text("clear_history"), 
                use_container_width=True, 
                disabled=not st.session_state.transcript):
        st.session_state.chat_history = []
        results_placeholder.empty()
        st.rerun()

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
                        model="mistral-tiny",
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