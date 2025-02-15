# YTWhisperer 🎥 💬

YTWhisperer is an AI-powered YouTube video analysis tool that helps you extract insights, summaries, and engage in meaningful conversations about video content using both Mistral AI and Groq language models.

## Features ✨

- **Dual AI Model Support**: Choose between Mistral AI and Groq for analysis
- **Transcript Analysis**: Automatically fetches and processes YouTube video transcripts
- **Smart Summaries**: Generates concise, well-structured video summaries
- **Interactive Q&A**: Ask questions about the video content and get AI-powered responses
- **Metadata Extraction**: Extracts key topics, concepts, and references from videos
- **Multilingual Support**: Available in English and Croatian
- **Rate Limiting**: Built-in protection against API overuse
- **Secure**: Multiple API key configuration options (environment variables, UI input, or Streamlit secrets)
- **Whisper Integration**: Alternative transcription method using Groq's Whisper API
- **Video Download**: Built-in YouTube video download capability with yt-dlp
- **Custom Logging**: Streamlit-integrated logging system for better user feedback

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/YTWhisperer.git
cd YTWhisperer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
   - Copy `.env.example` to `.env`
   - Add your API keys (optional, can also be entered through UI):
     - `MISTRAL_API_KEY` for Mistral AI
     - `GROQ_API_KEY` for Groq

## Usage 🎮

1. Start the application:
```bash
streamlit run main.py
```

2. Configure your API keys either:
   - In the `.env` file
   - Through the Settings ⚙️ panel in the UI
   - Via Streamlit secrets (for cloud deployment)

3. Paste a YouTube URL and explore:
   - Generate video summaries
   - View metadata analysis
   - Ask questions about the content
   - View full transcripts
   - Switch between AI models
   - Download video content (when needed)

## Requirements 📋

- Python 3.7+
- streamlit >= 1.31.0
- youtube_transcript_api >= 0.6.1
- mistralai >= 0.0.12
- groq >= 0.4.2
- python-dotenv >= 1.0.0
- numpy >= 1.24.0
- yt-dlp >= 2023.12.30

## Security 🔒

- Multiple API key configuration options
- Session-based key storage
- Secure key masking in the UI
- Rate limiting prevents API abuse
- HTTPS recommended for deployment

## Rate Limits ⚡

- 100 API calls per hour per session
- Automatic counter reset
- Visual tracking in the Settings panel
- Separate tracking for each AI model

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- [Mistral AI](https://mistral.ai/) for their powerful language model
- [Groq](https://groq.com/) for their language model and Whisper API  -- **CURRENTLY NOT WORKING PROPERLY**
- [Streamlit](https://streamlit.io/) for the awesome web framework
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript access
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video download capabilities

## Support 💬

If you have any questions or run into issues, please open an issue in the GitHub repository.
