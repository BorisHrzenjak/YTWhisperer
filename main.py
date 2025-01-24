import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
from mistralai import Mistral
import os
from dotenv import load_dotenv
import numpy as np
from youtube_transcript_api.formatters import TextFormatter
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

# App configuration
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="📺",
    layout="wide"
)

# Initialize session state
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "video_info" not in st.session_state:
    st.session_state.video_info = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []
if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks = []

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
st.title("YouTube Video Analyzer")

# URL input in main area
url = st.text_input("Enter YouTube URL:")

if url:
    video_id = extract_video_id(url)
    if video_id:
        # Create columns to constrain video width
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Display video player
            st.video(f"https://www.youtube.com/watch?v={video_id}")

        if not st.session_state.video_info:
            with st.spinner("Fetching video info..."):
                video_info = get_video_info(url)
                if video_info:
                    st.session_state.video_info = video_info
                    st.session_state.chat_history = []
                    
        if st.session_state.video_info:
            # Fetch and process transcript
            if not st.session_state.transcript:
                with st.spinner("Fetching transcript..."):
                    try:
                        transcript_list = YouTubeTranscriptApi.get_transcript(st.session_state.video_info["id"])
                        st.session_state.transcript = process_transcript(transcript_list)
                        chunks = [chunk.text for chunk in st.session_state.raw_chunks]
                        st.session_state.chunks = chunks
                        
                        # Show success message and transcript preview
                        st.success("✅ Transcript successfully loaded!")
                        with st.expander("View Transcript Preview"):
                            preview_chunk = st.session_state.raw_chunks[0]
                            st.markdown(f"""
                            **Preview of first chunk ({format_timestamp(preview_chunk.start_time)} - {format_timestamp(preview_chunk.end_time)}):**
                            
                            {preview_chunk.text[:500]}...
                            """)
                        
                        st.info("🔄 Generating embeddings... This may take a few minutes.")
                        
                        # Generate embeddings for chunks in batches
                        st.session_state.chunk_embeddings = []
                        chunk_batches = batch_list(chunks, batch_size=4)  
                        
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        for batch_idx, batch in enumerate(chunk_batches):
                            try:
                                progress_text.text(f"Processing batch {batch_idx + 1} of {len(chunk_batches)}...")
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
                        
                        st.success("✅ Embeddings successfully generated! You can now ask questions about the video.")
                        
                    except Exception as e:
                        st.error(f"Error fetching transcript: {str(e)}")

# Create a placeholder for results below the video
results_placeholder = st.empty()

# Sidebar for controls
with st.sidebar:
    st.title("Controls")
    st.divider()
    
    if st.session_state.transcript:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
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
                    model="mistral-medium",
                    messages=messages,
                    temperature=0.7,
                )
                with results_placeholder.container():
                    st.header("📝 Video Summary")
                    st.write(response.choices[0].message.content)
        
        if st.button("Show Metadata", use_container_width=True):
            with st.spinner("Extracting metadata..."):
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
                    model="mistral-medium",
                    messages=messages,
                    temperature=0.7,
                )
                with results_placeholder.container():
                    st.header("ℹ️ Video Metadata")
                    st.write(response.choices[0].message.content)

        if st.button("Show Transcript", use_container_width=True):
            with results_placeholder.container():
                st.header("📄 Full Transcript")
                # Create a formatted version of the transcript with timestamps
                formatted_transcript = ""
                for chunk in st.session_state.raw_chunks:
                    formatted_transcript += f"[{format_timestamp(chunk.start_time)} - {format_timestamp(chunk.end_time)}]\n{chunk.text}\n\n"
                st.markdown(formatted_transcript)
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            results_placeholder.empty()
            st.rerun()

# Chat interface in main area
if st.session_state.transcript:
    st.divider()
    st.subheader("Video Q&A Chat")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about the video..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
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
                        {"role": "system", "content": f"You are an AI assistant analyzing a YouTube video. Use the following transcript excerpts with timestamps to answer the user's question. If referencing specific parts of the video, mention the timestamp:\n\n{context}"},
                        {"role": "user", "content": prompt}
                    ]
                    
                    chat_response = client.chat.complete(
                        model="mistral-medium",
                        messages=messages,
                        temperature=0.7,
                    )
                    
                    response_text = chat_response.choices[0].message.content
                    st.markdown(response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")