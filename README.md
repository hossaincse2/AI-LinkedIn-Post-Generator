# AI LinkedIn Post Generator

A sophisticated AI-powered LinkedIn post generator built with LangChain, OpenAI GPT-4, and HuggingFace embeddings.

## 📁 Folder Structure

```
linkedin_post_generator/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── static/                        # Static files directory
│   ├── css/
│   │   └── styles.css            # CSS styles
│   ├── js/
│   │   └── script.js             # JavaScript functionality
│   └── images/                   # Optional: images/icons
├── templates/                     # HTML templates directory
│   └── index.html                # Main HTML template
└── README.md                     # Project documentation
```
## Features

- 🤖 AI-powered content generation using OpenAI GPT-4
- 🌍 Multi-language support (English, Spanish, French, Bengali, German, Portuguese)
- 📊 Vector similarity search using HuggingFace embeddings
- 🎯 Few-shot prompt engineering with professional examples
- 🌐 Beautiful web interface with real-time generation
- ☁️ Modal deployment support for scalable cloud hosting
- 🔄 LangChain integration for robust AI workflows

## Tech Stack

- **Backend**: Python, Flask, LangChain
- **AI Models**: OpenAI GPT-4, HuggingFace Sentence Transformers
- **Vector Store**: FAISS for similarity search
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Modal for serverless deployment
- **Embeddings**: l3cube-pune/bengali-sentence-similarity-sbert

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/linkedin-post-generator.git
cd linkedin-post-generator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**:
```bash
python linkedin_generator.py
```

## Environment Variables

Create a `.env` file with:

```
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_token_here
FLASK_ENV=development
PORT=5000
```

## Usage

### Web Interface
1. Open http://localhost:5000 in your browser
2. Enter a topic (e.g., "AI", "Remote Work", "Digital Marketing")
3. Select your preferred language
4. Click "Generate LinkedIn Post"
5. Copy the generated content to your clipboard

### Python API
```python
from linkedin_generator import LinkedInPostGenerator

# Initialize generator
generator = LinkedInPostGenerator(
    openai_api_key="your-key",
    github_token="your-token"
)

# Generate post
post = generator.generate_post("Artificial Intelligence", "English")
print(post.content)
```

### Modal Deployment
```bash
# Deploy to Modal
modal deploy modal_app.py

# Test the deployed function
modal run modal_app.py
```

## API Endpoints

### POST /generate
Generate a LinkedIn post

**Request Body:**
```json
{
  "topic": "Artificial Intelligence",
  "language": "English"
}
```

**Response:**
```json
{
  "content": "Generated LinkedIn post content...",
  "topic": "Artificial Intelligence",
  "language": "English",
  "hashtags": ["#AI", "#Innovation", "#Technology"],
  "generated_at": "2024-01-15T10:30:00"
}
```

## Testing

Run tests with different topics:
```bash
python linkedin_generator.py test
```

## Few-Shot Examples

The system uses professionally crafted examples for:
- AI and Technology topics
- Remote Work and Leadership
- Digital Marketing and Strategy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open a GitHub issue or contact [your-email@example.com]
"""

# Create README.md file
with open("README.md", "w") as f:
    f.write(README_CONTENT)

print("Setup files created successfully!")
print("\\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Set up your .env file with API keys")
print("3. Run the application: python linkedin_generator.py")
print("4. Open http://localhost:5000 in your browser")
