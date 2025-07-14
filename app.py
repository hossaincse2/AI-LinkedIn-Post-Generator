#!/usr/bin/env python3
"""
GitHub Models API integration for LinkedIn Post Generator
"""

import os
import sys
import signal
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass

# LangChain imports - Use ChatOpenAI instead of OpenAI
from langchain_openai import ChatOpenAI  # Changed from OpenAI to ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI client for direct API calls
from openai import OpenAI as OpenAIClient

# Web framework imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Additional imports
import json
import logging
from datetime import datetime
import time
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

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

    def __init__(self, github_token: str):
        self.github_token = github_token
        self.setup_models()
        self.setup_few_shot_examples()

        # Test GitHub Models API
        logger.info("Testing GitHub Models API connection...")
        if self.test_github_models_api():
            logger.info("GitHub Models API is working correctly!")
        else:
            logger.warning("GitHub Models API test failed, but continuing...")

    def setup_models(self):
        """Initialize LangChain models and embeddings"""
        # Use the correct model name with provider prefix
        self.model_name = "openai/gpt-4o-mini"

        try:
            # Initialize ChatOpenAI LLM with GitHub Models
            # GitHub Models API endpoint
            endpoint = "https://models.github.ai/inference"

            # Use ChatOpenAI instead of OpenAI for chat completions
            self.llm = ChatOpenAI(
                base_url=endpoint,
                api_key=self.github_token,
                model=self.model_name,
                temperature=0.7,
                max_tokens=500
            )

            # Initialize HuggingFace Embeddings
            logger.info("Initializing HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            logger.info(f"Models initialized successfully with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.error(f"Make sure your GITHUB_TOKEN is valid and has access to GitHub Models")
            logger.error(f"You can get a GitHub token at: https://github.com/settings/tokens")
            logger.error(f"GitHub Models endpoint: {endpoint}")
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
            template=template,
            partial_variables={
                "ai_example": self.few_shot_examples["AI"],
                "remote_work_example": self.few_shot_examples["Remote Work"],
                "digital_marketing_example": self.few_shot_examples["Digital Marketing"]
            }
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

    def test_github_models_api(self):
        """Test GitHub Models API directly to verify it works"""
        try:
            endpoint = "https://models.github.ai/inference"
            model = "openai/gpt-4o-mini"

            client = OpenAIClient(
                base_url=endpoint,
                api_key=self.github_token,
            )

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": "Say hello in one sentence.",
                    }
                ],
                temperature=0.7,
                top_p=1.0,
                model=model,
                max_tokens=50
            )

            test_response = response.choices[0].message.content
            logger.info(f"GitHub Models API test successful: {test_response}")
            return True

        except Exception as e:
            logger.error(f"GitHub Models API test failed: {e}")
            return False


# Alternative approach using direct OpenAI client instead of LangChain
class DirectLinkedInPostGenerator:
    """Alternative implementation using direct OpenAI client"""

    def __init__(self, github_token: str):
        self.github_token = github_token
        self.client = OpenAIClient(
            base_url="https://models.github.ai/inference",
            api_key=github_token,
        )
        self.model_name = "openai/gpt-4o-mini"

    def generate_post(self, topic: str, language: str = "English") -> str:
        """Generate LinkedIn post using direct OpenAI client"""

        system_prompt = """You are a professional LinkedIn content creator and social media expert. Your task is to generate engaging LinkedIn posts that drive meaningful engagement and professional networking.

Create LinkedIn posts that:
✅ START with an engaging hook (emoji, question, or bold statement)
✅ INCLUDE personal insights, experiences, or observations
✅ USE bullet points or numbered lists for key points
✅ ADD 6-8 relevant hashtags at the end
✅ INCLUDE a call-to-action or engaging question
✅ KEEP it professional but conversational
✅ LENGTH: 150-300 words
✅ MAKE it shareable and discussion-worthy

TONE: Professional yet approachable, thought-provoking, and authentic"""

        user_prompt = f"Create a LinkedIn post about '{topic}' in {language} language."

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating post: {e}")
            raise


# Flask Web Application
app = Flask(__name__)
CORS(app)

# Global generator instance
generator = None


def initialize_generator():
    """Initialize the LinkedIn post generator"""
    global generator

    # Get GitHub token from environment variables (.env file or system env)
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        logger.warning("GitHub token not found in environment variables.")
        logger.info("Please create a .env file with: GITHUB_TOKEN=your_token")
        logger.info("Or set it as system environment variable")
        logger.info("You can get a GitHub token at: https://github.com/settings/tokens")
        logger.info("Running in demo mode...")
        return None

    try:
        logger.info("Initializing LinkedIn Post Generator...")

        # Try the LangChain approach first
        try:
            generator = LinkedInPostGenerator(github_token)
            logger.info("Generator initialized successfully with LangChain and GitHub Models API")
        except Exception as e:
            logger.warning(f"LangChain approach failed: {e}")
            logger.info("Trying direct OpenAI client approach...")

            # Fallback to direct OpenAI client
            generator = DirectLinkedInPostGenerator(github_token)
            logger.info("Generator initialized successfully with direct OpenAI client")

        return generator

    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        logger.error("This might be due to:")
        logger.error("1. Invalid GitHub token")
        logger.error("2. GitHub Models API access issues")
        logger.error("3. Network connectivity problems")
        logger.error("4. Missing dependencies")
        logger.info("Falling back to demo mode...")
        return None


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_post():
    """API endpoint to generate LinkedIn post"""
    global generator

    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        language = data.get('language', 'English')

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400

        if not generator:
            # Demo mode - return sample content
            demo_content = generate_demo_post(topic, language)
            return jsonify({
                'content': demo_content,
                'topic': topic,
                'language': language,
                'demo_mode': True
            })

        # Generate post using the available generator
        if isinstance(generator, LinkedInPostGenerator):
            post = generator.generate_post(topic, language)
            return jsonify({
                'content': post.content,
                'topic': post.topic,
                'language': post.language,
                'hashtags': post.hashtags,
                'generated_at': post.generated_at
            })
        else:
            # Direct generator
            content = generator.generate_post(topic, language)
            return jsonify({
                'content': content,
                'topic': topic,
                'language': language,
                'generated_at': datetime.now().isoformat()
            })

    except Exception as e:
        logger.error(f"Error in generate_post: {e}")
        return jsonify({'error': str(e)}), 500


def generate_demo_post(topic: str, language: str) -> str:
    """Generate demo post when API is not available"""
    demo_posts = {
        "AI": """🤖 The AI revolution is here, and it's transforming how we work!

Just implemented an AI solution that boosted our team's productivity by 40%. Here's what I learned:

✅ AI amplifies human creativity, doesn't replace it
✅ The right data matters more than the fanciest model
✅ Ethical AI isn't optional—it's essential
✅ Continuous learning is the new competitive advantage

The future belongs to those who can blend human intuition with AI capabilities.

What's your experience with AI in your industry? Share your thoughts! 👇

#AI #ArtificialIntelligence #Innovation #MachineLearning #FutureOfWork #Technology #DigitalTransformation #Productivity""",

        "Remote Work": """🏠 Remote work taught me more about leadership than any office ever could.

After 3 years of leading distributed teams, here are my key insights:

🎯 Trust is the ultimate productivity multiplier
🤝 Async communication leads to better decisions
🌍 Global talent access is a game-changer
⚖️ Flexibility breeds loyalty and high performance
📈 Results matter more than face time

The companies still fighting remote work are missing the biggest talent opportunity in decades.

What's been your biggest remote work breakthrough?

#RemoteWork #Leadership #WorkLifeBalance #DistributedTeams #FutureOfWork #Management #Productivity #Innovation"""
    }

    # Return topic-specific demo or generate generic one
    if topic in demo_posts:
        return demo_posts[topic]
    else:
        return f"""🌟 Just had an incredible insight about {topic}!

After exploring this space extensively, I've discovered some fascinating trends that are reshaping how we approach {topic}.

Key observations:
✅ Innovation is accelerating at unprecedented speed
✅ Collaboration is more crucial than ever
✅ Data-driven decisions separate winners from followers
✅ Continuous adaptation is the new normal

The {topic} landscape is evolving rapidly, and early adopters will have a massive advantage.

What's your take on the future of {topic}? I'd love to hear your perspectives!

#{topic.replace(' ', '')} #Innovation #Leadership #Strategy #Growth #BusinessDevelopment #ProfessionalDevelopment #Success"""


def test_chain_with_topics():
    """Test the chain with different topics"""
    global generator

    if not generator:
        logger.warning("Generator not initialized. Running demo mode.")
        return

    test_topics = [
        "Artificial Intelligence",
        "Remote Work",
        "Digital Marketing",
        "Data Science",
        "Leadership"
    ]

    print("\n" + "=" * 50)
    print("TESTING LINKEDIN POST GENERATOR")
    print("=" * 50)

    for topic in test_topics:
        try:
            print(f"\n🔄 Generating post for: {topic}")

            if isinstance(generator, LinkedInPostGenerator):
                post = generator.generate_post(topic)
                print(f"✅ Generated successfully!")
                print(f"📝 Content preview: {post.content[:100]}...")
                print(f"🏷️  Hashtags: {post.hashtags}")
            else:
                content = generator.generate_post(topic)
                print(f"✅ Generated successfully!")
                print(f"📝 Content preview: {content[:100]}...")

            print("-" * 50)

        except Exception as e:
            print(f"❌ Error generating post for {topic}: {e}")


def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    logger.info("Received shutdown signal, cleaning up...")
    sys.exit(0)


def run_flask_app():
    """Run Flask app with proper error handling"""
    try:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        port = int(os.getenv("PORT", 5000))
        host = os.getenv("HOST", "127.0.0.1")

        logger.info(f"Starting Flask app on {host}:{port}")

        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        raise


if __name__ == "__main__":
    # Initialize generator
    generator = initialize_generator()

    # Test with sample topics
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_chain_with_topics()
    else:
        # Run Flask app
        run_flask_app()