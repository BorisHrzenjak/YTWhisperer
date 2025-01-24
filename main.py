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

@dataclass
class TranscriptChunk:
    text: str
    start_time: float
    end_time: float

# Load environment variables
load_dotenv()

# Initialize Mistral client
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    st.error("MISTRAL_API_KEY not set. Please set the environment variable.")
    st.stop()

client = Mistral(api_key=api_key)

# Translations dictionary
translations = {
    "en": {
        "title": "YouTube Video Analyzer",
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
        "title": "YouTube Video Analizator",
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

# Create a reference to the placeholder
results_placeholder = st.session_state.results_placeholder

# Helper functions
def extract_video_id(url):
    # Extract video ID from YouTube URL
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def get_video_info(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
            
        return {
            "id": video_id,
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
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
            # Display video player
            st.video(f"https://www.youtube.com/watch?v={video_id}")
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if transcript:
                st.session_state.transcript = process_transcript(transcript)
                st.success(get_text("transcript_loaded"))
                
                # Process transcript for embeddings
                chunks = [chunk.text for chunk in st.session_state.raw_chunks]
                st.session_state.chunks = chunks
                
                st.info(get_text("generating_embeddings"))
                
                # Generate embeddings for chunks in batches
                st.session_state.chunk_embeddings = []
                chunk_batches = batch_list(chunks, batch_size=4)  
                
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for batch_idx, batch in enumerate(chunk_batches):
                    try:
                        progress_text.text(get_text("processing_batch").format(batch_idx + 1, len(chunk_batches)))
                        embeddings = get_embeddings_with_retry(client, batch)
                        if embeddings:
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
                
        except Exception as e:
            st.error(f"Error loading transcript: {str(e)}")
    else:
        st.error(get_text("invalid_url"))

# Create a placeholder for results
results_placeholder = st.empty()

# Sidebar
with st.sidebar:
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

    # Always show buttons, but disable them if no transcript
    if st.button(get_text("generate_summary"), 
                use_container_width=True, 
                disabled=not st.session_state.transcript):
        with st.spinner("Generating summary..."):
            try:
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
        with st.spinner("Extracting metadata..."):
            try:
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
            with st.spinner("Thinking..."):
                try:
                    # Get relevant context using embeddings
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