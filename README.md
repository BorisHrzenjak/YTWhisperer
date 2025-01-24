# YTWhisperer 🎥 💬

YTWhisperer is an AI-powered YouTube video analysis tool that helps you extract insights, summaries, and engage in meaningful conversations about video content using the Mistral AI language model.

## Features ✨

- **Transcript Analysis**: Automatically fetches and processes YouTube video transcripts
- **Smart Summaries**: Generates concise, well-structured video summaries
- **Interactive Q&A**: Ask questions about the video content and get AI-powered responses
- **Metadata Extraction**: Extracts key topics, concepts, and references from videos
- **Multilingual Support**: Available in English and Croatian
- **Rate Limiting**: Built-in protection against API overuse
- **Secure**: API keys are handled securely with multiple configuration options

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
   - Add your Mistral API key (optional, can also be entered through UI)

## Usage 🎮

1. Start the application:
```bash
streamlit run main.py
```

2. Enter your Mistral API key either:
   - In the `.env` file as `MISTRAL_API_KEY=your_key_here`
   - Through the Settings ⚙️ panel in the UI

3. Paste a YouTube URL and explore:
   - Generate video summaries
   - View metadata analysis
   - Ask questions about the content
   - View full transcripts

## Requirements 📋

- Python 3.7+
- Streamlit
- YouTube Transcript API
- Mistral AI API
- python-dotenv
- NumPy

## Security 🔒

- API keys can be configured via `.env` or UI
- Keys are never exposed in the interface
- Rate limiting prevents API abuse
- Session-based key storage
- HTTPS recommended for deployment

## Rate Limits ⚡

- 100 API calls per hour per session
- Automatic counter reset
- Visual tracking in the Settings panel

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
- [Streamlit](https://streamlit.io/) for the awesome web framework
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript access

## Support 💬

If you have any questions or run into issues, please open an issue in the GitHub repository.
