# setup.py
from setuptools import setup, find_packages

setup(
    name="linkedin-post-generator",
    version="1.0.0",
    description="AI-powered LinkedIn post generator using LangChain and OpenAI",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.10.0",
        "flask>=2.3.3",
        "flask-cors>=4.0.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "huggingface-hub>=0.20.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "python-dotenv>=1.0.0",
        "modal>=0.62.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# .env (environment variables file)
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_token_here
FLASK_ENV=development
PORT=5000

# modal_app.py (Modal deployment configuration)
import modal
import os

# Define the Modal app
app = modal.App("linkedin-post-generator")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "langchain==0.1.0",
    "openai==1.10.0",
    "flask==2.3.3",
    "flask-cors==4.0.0",
    "sentence-transformers==2.2.2",
    "faiss-cpu==1.7.4",
    "huggingface-hub==0.20.0",
    "transformers==4.36.0",
    "torch==2.1.0",
    "numpy==1.24.3",
    "pandas==2.0.3",
    "python-dotenv==1.0.0"
])

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("github-token")
    ],
    timeout=300
)
def generate_linkedin_post_modal(topic: str, language: str = "English"):
    """Modal function to generate LinkedIn posts"""
    from linkedin_generator import LinkedInPostGenerator
    
    generator = LinkedInPostGenerator(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        github_token=os.environ["GITHUB_TOKEN"]
    )
    
    post = generator.generate_post(topic, language)
    return {
        "content": post.content,
        "topic": post.topic,
        "language": post.language,
        "hashtags": post.hashtags,
        "generated_at": post.generated_at
    }

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("github-token")
    ]
)
@modal.web_endpoint(method="POST")
def web_generate_post(item: dict):
    """Web endpoint for generating posts via Modal"""
    topic = item.get("topic", "")
    language = item.get("language", "English")
    
    if not topic:
        return {"error": "Topic is required"}, 400
    
    result = generate_linkedin_post_modal.remote(topic, language)
    return result

if __name__ == "__main__":
    with app.run():
        # Test the function
        result = generate_linkedin_post_modal.remote("Artificial Intelligence")
        print("Generated post:", result)
