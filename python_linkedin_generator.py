#!/usr/bin/env python3
"""
AI LinkedIn Post Generator using LangChain
Built with OpenAI GPT-4, HuggingFace Embeddings, and Modal integration
"""

import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

# LangChain imports
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Web framework imports
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Additional imports
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LinkedInPost:
    """Data class for LinkedIn post structure"""
    topic: str
    language: str
    content: str
    hashtags: List[str]
    generated_at: str

class LinkedInPostGenerator:
    """Main class for generating LinkedIn posts using LangChain"""
    
    def __init__(self, openai_api_key: str, github_token: str):
        self.openai_api_key = openai_api_key
        self.github_token = github_token
        self.setup_models()
        self.setup_few_shot_examples()
        
    def setup_models(self):
        """Initialize LangChain models and embeddings"""
        try:
            # Initialize OpenAI LLM
            self.llm = OpenAI(
                openai_api_key=self.openai_api_key,
                model_name="gpt-4o-mini",  # Using GPT-4 nano equivalent
                temperature=0.7,
                max_tokens=500
            )
            
            # Initialize HuggingFace Embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="l3cube-pune/bengali-sentence-similarity-sbert",
                model_kwargs={'device': 'cpu'}
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def setup_few_shot_examples(self):
        """Set up few-shot examples for prompt engineering"""
        self.few_shot_examples = {
            "AI": """🤖 The future of work is here, and it's powered by AI!

Just attended an amazing conference on artificial intelligence, and I'm blown away by how rapidly this field is evolving. From automating routine tasks to enabling breakthrough discoveries, AI is reshaping every industry.

Key takeaways:
✅ AI augments human capabilities, doesn't replace them
✅ Ethical AI development is crucial for sustainable progress
✅ Continuous learning is essential to stay relevant

The companies that embrace AI thoughtfully will lead tomorrow's economy. Are you ready to be part of this transformation?

#AI #ArtificialIntelligence #FutureOfWork #Innovation #Technology #MachineLearning #Ethics #DigitalTransformation""",

            "Remote Work": """🏠 Remote work isn't just a trend—it's a fundamental shift in how we think about productivity and work-life balance.

After 3 years of leading a fully remote team, here's what I've learned:

🎯 Results matter more than hours logged
🤝 Communication becomes intentional and efficient  
🌍 Access to global talent pools
⚖️ Better work-life integration (not balance!)
📈 Increased productivity when done right

The key? Building trust, setting clear expectations, and investing in the right tools and culture.

Companies still debating remote work are missing out on incredible talent and innovation opportunities.

#RemoteWork #WorkFromHome #Productivity #Leadership #FutureOfWork #DistributedTeams #WorkLifeBalance #Management""",

            "Digital Marketing": """📱 Digital marketing in 2024: It's not about being everywhere—it's about being where your audience actually is.

Just wrapped up a campaign that taught me valuable lessons:

💡 Authenticity beats perfection every time
📊 Data tells the story, but creativity sells it
🎯 Micro-targeting > broad reach
🤝 Community building > follower count
📱 Mobile-first isn't optional anymore

The brands winning today are those that listen first, then speak. They're creating genuine connections, not just conversions.

What's your biggest digital marketing challenge right now?

#DigitalMarketing #MarketingStrategy #SocialMedia #ContentMarketing #BrandBuilding #CustomerEngagement #DataDriven #Innovation"""
        }
        
        # Create vector store from examples for similarity search
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Create vector store from few-shot examples"""
        try:
            texts = list(self.few_shot_examples.values())
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            docs = text_splitter.create_documents(texts)
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            self.vector_store = None
    
    def create_prompt_template(self, language: str = "English") -> PromptTemplate:
        """Create prompt template with few-shot examples"""
        
        template = """You are a professional LinkedIn content creator and social media expert. Your task is to generate engaging LinkedIn posts that drive meaningful engagement and professional networking.

Here are examples of excellent LinkedIn posts:

EXAMPLE 1 - AI Topic:
{ai_example}

EXAMPLE 2 - Remote Work Topic:
{remote_work_example}

EXAMPLE 3 - Digital Marketing Topic:
{digital_marketing_example}

INSTRUCTIONS:
Create a LinkedIn post about "{topic}" in {language} language following these guidelines:

✅ START with an engaging hook (emoji, question, or bold statement)
✅ INCLUDE personal insights, experiences, or observations
✅ USE bullet points or numbered lists for key points
✅ ADD 6-8 relevant hashtags at the end
✅ INCLUDE a call-to-action or engaging question
✅ KEEP it professional but conversational
✅ LENGTH: 150-300 words
✅ MAKE it shareable and discussion-worthy

TONE: Professional yet approachable, thought-provoking, and authentic

Topic: {topic}
Language: {language}

LinkedIn Post:"""

        return PromptTemplate(
            input_variables=["topic", "language"],
            template=template.format(
                ai_example=self.few_shot_examples["AI"],
                remote_work_example=self.few_shot_examples["Remote Work"],
                digital_marketing_example=self.few_shot_examples["Digital Marketing"],
                topic="{topic}",
                language="{language}"
            )
        )
    
    def generate_post(self, topic: str, language: str = "English") -> LinkedInPost:
        """Generate LinkedIn post using LangChain"""
        try:
            # Create prompt template
            prompt_template = self.create_prompt_template(language)
            
            # Create LLM Chain
            llm_chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template,
                verbose=True
            )
            
            # Generate content
            content = llm_chain.run(topic=topic, language=language)
            
            # Extract hashtags from content
            hashtags = self.extract_hashtags(content)
            
            # Create LinkedIn post object
            post = LinkedInPost(
                topic=topic,
                language=language,
                content=content.strip(),
                hashtags=hashtags,
                generated_at=datetime.now().isoformat()
            )
            
            logger.info(f"Generated post for topic: {topic}")
            return post
            
        except Exception as e:
            logger.error(f"Error generating post: {e}")
            raise
    
    def extract_hashtags(self, content: str) -> List[str]:
        """Extract hashtags from generated content"""
        import re
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, content)
        return hashtags
    
    def get_similar_examples(self, topic: str, k: int = 2) -> List[str]:
        """Get similar examples using vector similarity search"""
        if not self.vector_store:
            return []
        
        try:
            similar_docs = self.vector_store.similarity_search(topic, k=k)
            return [doc.page_content for doc in similar_docs]
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

# Flask Web Application
app = Flask(__name__)
CORS(app)

# Global generator instance
generator = None

def initialize_generator():
    """Initialize the LinkedIn post generator"""
    global generator
    
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    github_token = os.getenv("GITHUB_TOKEN", "your-github-token")
    
    if not openai_api_key or openai_api_key == "your-openai-api-key":
        logger.warning("OpenAI API key not found. Using demo mode.")
        return None
    
    try:
        generator = LinkedInPostGenerator(openai_api_key, github_token)
        logger.info("Generator initialized successfully")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI LinkedIn Post Generator</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #2c3e50;
            }
            input, select, button {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }
            button {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                cursor: pointer;
                font-weight: bold;
                transition: transform 0.