import requests
import gradio as gr
import logging
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests.exceptions import Timeout
from urllib.request import urlopen, Request
import json
from huggingface_hub import InferenceClient
import random
import time
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
import os
from dotenv import load_dotenv
import certifi
import requests
from newspaper import Article
import PyPDF2
import io
import requests
import random
import datetime
from groq import Groq
import os
from mistralai import Mistral
from dotenv import load_dotenv
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np
from math import log
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple
import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Automatically get the current year
CURRENT_YEAR = datetime.datetime.now().year

# Load environment variables from a .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SearXNG instance details
SEARXNG_URL = os.getenv("SEARXNG_URL")
SEARXNG_KEY = os.getenv("SEARXNG_KEY")


logger.info(f"SearXNG URL: {SEARXNG_URL}")
logger.info(f"SearXNG Key: {SEARXNG_KEY}") 


# ... other environment variables ...
CUSTOM_LLM = os.getenv("CUSTOM_LLM")
CUSTOM_LLM_DEFAULT_MODEL = os.getenv("CUSTOM_LLM_DEFAULT_MODEL")

logger.info(f"CUSTOM_LLM: {CUSTOM_LLM}")
logger.info(f"CUSTOM_LLM_DEFAULT_MODEL: {CUSTOM_LLM_DEFAULT_MODEL}")

# Define the fetch_custom_models function here
def fetch_custom_models():
    if not CUSTOM_LLM:
        return []
    try:
        response = requests.get(f"{CUSTOM_LLM}/api/tags")  # Ollama endpoint for listing models
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]  # Ollama returns model names directly
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return []

# Fetch custom models and determine the default model
custom_models = fetch_custom_models()
all_models = ["huggingface", "groq", "mistral"] + custom_models

# Determine the default model
default_model = CUSTOM_LLM_DEFAULT_MODEL if CUSTOM_LLM_DEFAULT_MODEL in all_models else "groq"

logger.info(f"Default model selected: {default_model}")

# Use the environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    "mistralai/Mistral-Small-Instruct-2409",
    token=HF_TOKEN,
)

# Default API key for examples (replace with a dummy value or leave empty)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Mistral client
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Initialize the similarity model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Create a base class for AI models
class AIModel(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        pass

# Step 2: Implement specific classes for each AI model
class HuggingFaceModel(AIModel):
    def __init__(self, client):
        self.client = client

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class GroqModel(AIModel):
    def __init__(self, client):
        self.client = client

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class MistralModel(AIModel):
    def __init__(self, client):
        self.client = client

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = self.client.chat.complete(
            model="open-mistral-nemo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

# Step 3: Use a factory pattern to create model instances
class CustomModel(AIModel):
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = os.getenv("CUSTOM_LLM", "http://localhost:11434")

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        try:
            # Convert messages to Ollama format
            prompt = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in messages
            ])

            response = requests.post(
                f"{self.base_url}/api/generate",  # Ollama endpoint
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            response.raise_for_status()
            
            # Handle Ollama's streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_response += chunk['response']
                    if chunk.get('done', False):
                        break
                        
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama model: {e}")
            return f"Error: Unable to generate response from Ollama model. {str(e)}"

class AIModelFactory:
    @staticmethod
    def create_model(model_name: str, client: Any = None) -> AIModel:
        if model_name == "huggingface":
            return HuggingFaceModel(client)
        elif model_name == "groq":
            return GroqModel(client)
        elif model_name == "mistral":
            return MistralModel(client)
        elif CUSTOM_LLM and model_name in fetch_custom_models():
            return CustomModel(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

def determine_query_type(query: str, chat_history: str, ai_model: AIModel) -> str:
    system_prompt = """You are Sentinel, an intelligent AI agent tasked with determining whether a user query requires a web search or can be answered using your existing knowledge base. Your knowledge cutoff date is 2023, and the current year is 2024. Your task is to analyze the query and decide on the appropriate action.

    Instructions for Sentinel:
    1. If the query is a general conversation starter, greeting, or can be answered with information from 2023 or earlier, classify it as "knowledge_base".
    2. If the query requires information from 2024, up-to-date news, current events, or real-time data, classify it as "web_search".
    3. For queries about ongoing events, trends, or situations that likely have significant updates in 2024, classify as "web_search".
    4. Consider the chat history when making your decision.
    5. Respond with ONLY "knowledge_base" or "web_search".

    Instructions for users (include this in your first interaction):
    "Hello! I'm Sentinel, your AI assistant. I can help you with various tasks and answer your questions. Here's how to get the best results:
    - My knowledge base is current up to 2023. For information up to that year, I can answer directly.
    - For any information, events, or data from 2024 onwards, I'll need to search the web for the most up-to-date results.
    - If you're asking about ongoing situations or need the very latest information, please mention that you need current data.
    - Feel free to ask follow-up questions or request clarification on any topic.
    - If you're unsure whether I need to search, you can ask 'Do you need to search the web for this?'

    How can I assist you today?"

    Examples:
    - "Hi, how are you?" -> "knowledge_base"
    - "What were the major events of 2023?" -> "knowledge_base"
    - "What's the latest news in the US?" -> "web_search"
    - "Can you explain quantum computing?" -> "knowledge_base"
    - "What are the current stock prices for Apple?" -> "web_search"
    - "Who won the 2024 Super Bowl?" -> "web_search"
    - "What were the key findings of the 2022 climate report?" -> "knowledge_base"
    """

    user_prompt = f"""
    Chat history:
    {chat_history}

    Current query: {query}

    Determine if this query requires a web search or can be answered from the knowledge base.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = ai_model.generate_response(
            messages=messages,
            max_tokens=10,
            temperature=0.2
        )
        decision = response.strip().lower()
        return "web_search" if decision == "web_search" else "knowledge_base"
    except Exception as e:
        logger.error(f"Error determining query type: {e}")
        return "web_search"  # Default to web search if there's an error

def generate_ai_response(query: str, chat_history: str, ai_model: AIModel, temperature: float) -> str:
    system_prompt = """You are a helpful AI assistant. Provide a concise and informative response to the user's query based on your existing knowledge. Do not make up information or claim to have real-time data."""

    user_prompt = f"""
    Chat history:
    {chat_history}

    Current query: {query}

    Please provide a response to the query.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = ai_model.generate_response(
            messages=messages,
            max_tokens=500,
            temperature=temperature
        )
        return response
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I apologize, but I'm having trouble generating a response at the moment. Please try again later."


# Set up a session with retry mechanism
def requests_retry_session(
    retries=0,
    backoff_factor=0.1,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def scrape_pdf_content(url, max_chars=3000, timeout=5):
    try:
        logger.info(f"Scraping PDF content from: {url}")
        
        # Download the PDF file
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
        
        # Extract text from all pages
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
        
        # Limit the content to max_chars
        return content[:max_chars] if content else ""
    except requests.Timeout:
        logger.error(f"Timeout error while scraping PDF content from {url}")
        return ""
    except Exception as e:
        logger.error(f"Error scraping PDF content from {url}: {e}")
        return ""

def scrape_with_newspaper(url):
    if url.lower().endswith('.pdf'):
        return scrape_pdf_content(url)
    
    logger.info(f"Starting to scrape with Newspaper3k: {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Combine title and text
        content = f"Title: {article.title}\n\n"
        content += article.text
        
        # Add publish date if available
        if article.publish_date:
            content += f"\n\nPublish Date: {article.publish_date}"
        
        # Add authors if available
        if article.authors:
            content += f"\n\nAuthors: {', '.join(article.authors)}"
        
        # Add top image URL if available
        if article.top_image:
            content += f"\n\nTop Image URL: {article.top_image}"
        
        return content
    except Exception as e:
        logger.error(f"Error scraping {url} with Newspaper3k: {e}")
        return ""

def rephrase_query(chat_history, query, model, temperature=0.2) -> str:
    system_prompt = """You are a highly intelligent and context-aware conversational assistant. Your tasks are as follows:

1. Determine if the new query is a continuation of the previous conversation or an entirely new topic.

2. For both continuations and new topics:
   a. **Entity Identification and Quotation**:
      - Analyze the user's query to identify the main entities (e.g., organizations, brands, products, locations).
      - For each identified entity, enclose ONLY the entity itself in double quotes within the query.
      - If no identifiable entities are found, proceed without adding quotes.
   b. **Query Preservation**:
      - Maintain the entire original query, including any parts after commas or other punctuation.
      - Do not remove or truncate any part of the original query.

3. If it's a continuation:
   - Incorporate relevant information from the context to make the query more specific and contextual.
   - Ensure that entities from the previous context are properly quoted if they appear in the rephrased query.

4. For both continuations and new topics:
   - First, check if the query contains words indicating current information (e.g., "today", "now", "current", "latest"):
     - If present, do NOT add any date operators to the query
   - Otherwise, if the query mentions a specific time period (e.g., a quarter, year, or date range):
     - Add appropriate "after: " operators to the end of the rephrased query.
     - Use the format "after: YYYY" for date ranges.
   - If no specific time period is mentioned and no current-time indicators are present:
     - Append "after: {CURRENT_YEAR}" to the end of the rephrased query.
   - Do not use quotes or the "+" operator when adding dates.

5. **Output**:
   - Return ONLY the rephrased query, ensuring it is concise, clear, and contextually accurate.
   - Do not include any additional commentary or explanation.

### Example Scenarios

**Scenario 1: Query About Current Information**
- **User Query**: "What's the stock price of Apple today?"
- **Rephrased Query**: "What's the stock price of \"Apple\" today"

**Scenario 2: New Topic with Specific Quarter**
- **User Query**: "How did Bank of America perform during Q2 2024?"
- **Rephrased Query**: "How did \"Bank of America\" perform during Q2 2024 after: 2024"

**Scenario 3: Continuation with Date Range**
- **Previous Query**: "What were Apple's sales figures for 2023?"
- **User Query**: "How about for the first half of 2024?"
- **Rephrased Query**: "How about \"Apple\"'s sales figures for the first half of 2024 after: 2024"

**Scenario 4: Current Status Query**
- **User Query**: "What is the current market share of Toyota and Honda in the US?"
- **Rephrased Query**: "What is the current market share of \"Toyota\" and \"Honda\" in the \"US\""

**Scenario 5: Current Status Query**
- **User Query**: "Bank of America Q2 2024 earnings?"
- **Rephrased Query**: "\"Bank of America\" Q2 2024 earnings after: 2024""
"""

    # Create the user prompt with the chat history and current query
    user_prompt = f"""Conversation context: {chat_history}
New query: {query}
Current year: {CURRENT_YEAR}
Rephrased query:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        logger.info(f"Sending rephrasing request to {model} with temperature {temperature}")
        
        if model == "groq":
            response = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                max_tokens=150,
                temperature=temperature,
                top_p=0.9,
                presence_penalty=1.2,
                stream=False
            )
            rephrased_question = response.choices[0].message.content.strip()
            
        elif model == "mistral":
            response = mistral_client.chat.complete(
                model="open-mistral-nemo",
                messages=messages,
                max_tokens=150,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )
            rephrased_question = response.choices[0].message.content.strip()
            
        elif CUSTOM_LLM and model in fetch_custom_models():
            # Create CustomModel instance for Ollama
            custom_model = CustomModel(model)
            rephrased_question = custom_model.generate_response(
                messages=messages,
                max_tokens=150,
                temperature=temperature
            )
            
        else:  # huggingface
            response = client.chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=temperature,
                frequency_penalty=1.4,
                top_p=0.9
            )
            rephrased_question = response.choices[0].message.content.strip()

        # Remove surrounding quotes if present
        if (rephrased_question.startswith('"') and rephrased_question.endswith('"')) or \
           (rephrased_question.startswith("'") and rephrased_question.endswith("'")):
            rephrased_question = rephrased_question[1:-1].strip()

        logger.info(f"Rephrased Query (cleaned): {rephrased_question}")
        return rephrased_question
        
    except Exception as e:
        logger.error(f"Error rephrasing query with {model} LLM: {e}")
        return query  # Fallback to original query if rephrasing fails

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # term frequency saturation parameter
        self.b = b    # length normalization parameter
        self.corpus_size = 0
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_vectors = []
        
    def fit(self, corpus: List[str]):
        """
        Fit BM25 parameters to the corpus
        
        Args:
            corpus: List of document strings
        """
        self.corpus_size = len(corpus)
        
        # Calculate document lengths and average document length
        self.doc_lengths = []
        for doc in corpus:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
        self.avgdl = sum(self.doc_lengths) / self.corpus_size
        
        # Calculate document frequencies
        df = Counter()
        self.doc_vectors = []
        
        for doc in corpus:
            words = doc.lower().split()
            doc_words = set(words)
            for word in doc_words:
                df[word] += 1
            self.doc_vectors.append(Counter(words))
        
        # Calculate inverse document frequency
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = log((self.corpus_size - freq + 0.5) / (freq + 0.5))
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Calculate BM25 scores for the query against all documents
        
        Args:
            query: Query string
            
        Returns:
            numpy array of scores for each document
        """
        scores = np.zeros(self.corpus_size)
        query_words = query.lower().split()
        
        for word in query_words:
            if word not in self.idf:
                continue
                
            qi = self.idf[word]
            for idx, doc_vector in enumerate(self.doc_vectors):
                if word not in doc_vector:
                    continue
                    
                score = (qi * doc_vector[word] * (self.k1 + 1) /
                        (doc_vector[word] + self.k1 * (1 - self.b + self.b * 
                         self.doc_lengths[idx] / self.avgdl)))
                scores[idx] += score
                
        return scores

def prepare_documents_for_bm25(documents: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Prepare documents for BM25 ranking by combining title and content
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Tuple of (document texts, original documents)
    """
    doc_texts = []
    valid_documents = []
    
    for doc in documents:
        try:
            # Get title and content with default empty strings if missing
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            # Skip documents with no content and title
            if not (title.strip() or content.strip()):
                logger.warning(f"Skipping document with no title or content: {doc}")
                continue
                
            # Combine title and content for better matching
            doc_text = f"{title} {content}".strip()
            doc_texts.append(doc_text)
            valid_documents.append(doc)
            
        except Exception as e:
            logger.warning(f"Error processing document {doc}: {e}")
            continue
            
    if not valid_documents:
        raise ValueError("No valid documents found with required fields")
        
    return doc_texts, valid_documents

def rerank_documents(query: str, documents: List[Dict], 
                    similarity_threshold: float = 0.95, max_results: int = 5) -> List[Dict]:
    try:
        if not documents:
            logger.warning("No documents to rerank.")
            return documents
            
        # Validate input documents
        if not all(isinstance(doc, dict) for doc in documents):
            raise ValueError("All documents must be dictionaries")
            
        # Step 1: Prepare documents for BM25
        doc_texts, valid_docs = prepare_documents_for_bm25(documents)
        
        if not valid_docs:
            logger.warning("No valid documents after preparation.")
            return documents[:max_results]
            
        # Verify all documents have summaries for semantic scoring
        valid_docs = [doc for doc in valid_docs if 'summary' in doc and doc['summary'].strip()]
        
        if not valid_docs:
            logger.warning("No documents with valid summaries found.")
            return documents[:max_results]
            
        # Step 2: Initialize and fit BM25
        bm25 = BM25()
        bm25.fit(doc_texts)
        
        # Step 3: Get BM25 scores
        bm25_scores = bm25.get_scores(query)
        
        # Step 4: Get semantic similarity scores
        query_embedding = similarity_model.encode(query, convert_to_tensor=True)
        doc_summaries = [doc['summary'] for doc in valid_docs]
        doc_embeddings = similarity_model.encode(doc_summaries, convert_to_tensor=True)
        semantic_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Step 5: Combine scores (normalize first)
        if len(bm25_scores) > 1:
            bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        else:
            bm25_scores_norm = bm25_scores
            
        if len(semantic_scores) > 1:
            semantic_scores_norm = (semantic_scores - torch.min(semantic_scores)) / (torch.max(semantic_scores) - torch.min(semantic_scores))
        else:
            semantic_scores_norm = semantic_scores
        
        # Combine scores with weights (0.4 for BM25, 0.6 for semantic similarity)
        combined_scores = 0.4 * bm25_scores_norm + 0.6 * semantic_scores_norm.numpy()
        
        # Create scored documents with combined scores
        scored_documents = list(zip(valid_docs, combined_scores))
        
        # Sort by combined score (descending)
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        # Filter similar documents
        filtered_docs = []
        added_contents = []
        
        for doc, score in scored_documents:
            if score < 0.3:  # Minimum relevance threshold
                continue
                
            # Check similarity with already selected documents
            doc_embedding = similarity_model.encode(doc['summary'], convert_to_tensor=True)
            is_similar = False
            
            for content in added_contents:
                content_embedding = similarity_model.encode(content, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(doc_embedding, content_embedding)
                if similarity > similarity_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                filtered_docs.append(doc)
                added_contents.append(doc['summary'])
            
            if len(filtered_docs) >= max_results:
                break
        
        logger.info(f"Reranked and filtered to {len(filtered_docs)} unique documents using BM25 and semantic similarity.")
        return filtered_docs
        
    except Exception as e:
        logger.error(f"Error during reranking documents: {e}")
        return documents[:max_results]  # Fallback to first max_results documents if reranking fails

def compute_similarity(text1, text2):
    # Encode the texts
    embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
    embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
    
    return cosine_similarity.item()

def is_content_unique(new_content, existing_contents, similarity_threshold=0.8):
    for existing_content in existing_contents:
        similarity = compute_similarity(new_content, existing_content)
        if similarity > similarity_threshold:
            return False
    return True

def assess_relevance_and_summarize(llm_client, query, document, model, temperature=0.2) -> str:
    system_prompt = """You are a world-class AI assistant specializing in news analysis and document summarization. Your task is to provide a comprehensive and detailed summary of the given document that captures its key points and relevance to the user's query."""

    user_prompt = f"""
Query: {query}

Document Title: {document['title']}
Document Content:
{document['content'][:1000]}  # Limit to first 1000 characters for efficiency

Instructions:
1. Provide a detailed summary that captures the unique aspects of this document. Include:
   - Key facts and figures
   - Dates of events or announcements
   - Names of important entities mentioned
   - Any metrics or changes reported
   - The potential impact or significance of the content
2. Focus on aspects that are most relevant to the user's query
3. Ensure the summary is distinctive and highlights what makes this particular document unique
4. Include any specific context that helps understand the document's significance

Your response should be in the following format:
Summary: [Your detailed summary]

Remember to:
- Highlight the most important information first
- Include specific numbers, dates, and facts when available
- Connect the information to the user's query where relevant
- Focus on what makes this document unique or noteworthy
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        if model == "groq":
            response = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                max_tokens=500,
                temperature=temperature,
                top_p=0.9,
                presence_penalty=1.2,
                stream=False
            )
            summary = response.choices[0].message.content.strip()
            
        elif model == "mistral":
            response = mistral_client.chat.complete(
                model="open-mistral-nemo",
                messages=messages,
                max_tokens=500,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )
            summary = response.choices[0].message.content.strip()
            
        elif CUSTOM_LLM and model in fetch_custom_models():
            # Create CustomModel instance for Ollama
            custom_model = CustomModel(model)
            summary = custom_model.generate_response(
                messages=messages,
                max_tokens=500,
                temperature=temperature
            )
            
        else:  # huggingface
            response = client.chat_completion(
                messages=messages,
                max_tokens=500,
                temperature=temperature,
                frequency_penalty=1.4,
                top_p=0.9
            )
            summary = response.choices[0].message.content.strip()
        
        # Clean up the summary if needed
        if summary.startswith("Summary: "):
            summary = summary[9:].strip()
            
        return f"Relevant: Yes\nSummary: {summary}"
        
    except Exception as e:
        error_msg = f"Error summarizing with {model} LLM: {str(e)}"
        logger.error(error_msg)
        return f"Relevant: Yes\nSummary: Error occurred while summarizing the document: {str(e)}"

def scrape_full_content(url, max_chars=3000, timeout=5, use_pydf2=True):
    try:
        logger.info(f"Scraping full content from: {url}")
        
        # Check if the URL ends with .pdf
        if url.lower().endswith('.pdf'):
            if use_pydf2:
                return scrape_pdf_content(url, max_chars, timeout)
            else:
                logger.info(f"Skipping PDF document: {url}")
                return None
        
        # Use Newspaper3k for non-PDF content
        content = scrape_with_newspaper(url)
        
        # Limit the content to max_chars
        return content[:max_chars] if content else ""
    except requests.Timeout:
        logger.error(f"Timeout error while scraping full content from {url}")
        return ""
    except Exception as e:
        logger.error(f"Error scraping full content from {url}: {e}")
        return ""

def llm_summarize(json_input, model, temperature=0.2):
    system_prompt = """You are Sentinel, a world-class AI model who is expert at searching the web and answering user's queries. You are also an expert at summarizing web pages or documents and searching for content in them."""
    user_prompt = f"""
Please provide a comprehensive summary based on the following JSON input:
{json_input}
Instructions:
1. Analyze the query and the provided documents.
2. Write a detailed, long, and complete research document that is informative and relevant to the user's query based on provided context (the context consists of search results containing a brief description of the content of that page).
3. You must use this context to answer the user's query in the best way possible. Use an unbiased and journalistic tone in your response. Do not repeat the text.
4. Use an unbiased and professional tone in your response.
5. Do not repeat text verbatim from the input.
6. Provide the answer in the response itself.
7. You can use markdown to format your response.
8. Use bullet points to list information where appropriate.
9. Cite the answer using [number] notation along with the appropriate source URL embedded in the notation.
10. Place these citations at the end of the relevant sentences.
11. You can cite the same sentence multiple times if it's relevant to different parts of your answer.
12. Make sure the answer is not short and is informative.
13. Your response should be detailed, informative, accurate, and directly relevant to the user's query."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        if model == "groq":
            response = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                max_tokens=5500,
                temperature=temperature,
                top_p=0.9,
                presence_penalty=1.2,
                stream=False
            )
            return response.choices[0].message.content.strip()
        elif model == "mistral":
            response = mistral_client.chat.complete(
                model="open-mistral-nemo",
                messages=messages,
                max_tokens=10000,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )
            return response.choices[0].message.content.strip()
        elif CUSTOM_LLM and model in fetch_custom_models():
            # Create CustomModel instance for Ollama
            custom_model = CustomModel(model)
            response = custom_model.generate_response(
                messages=messages,
                max_tokens=1000,
                temperature=temperature
            )
            return response
        else:  # huggingface
            response = client.chat_completion(
                messages=messages,
                max_tokens=10000,
                temperature=temperature,
                frequency_penalty=1.4,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in LLM summarization: {e}")
        return "Error: Unable to generate a summary. Please try again."

def search_and_scrape(
    query: str,
    chat_history: str,
    ai_model: AIModel,
    num_results: int = 10,
    max_chars: int = 1500,
    time_range: str = "",
    language: str = "en",
    category: str = "general",
    engines: List[str] = [],
    safesearch: int = 2,
    method: str = "GET",
    llm_temperature: float = 0.2,
    timeout: int = 5,
    model: str = "huggingface",
    use_pydf2: bool = True
):
    try:
        # Step 1: Rephrase the Query
        rephrased_query = rephrase_query(chat_history, query, model, temperature=llm_temperature)
        logger.info(f"Rephrased Query: {rephrased_query}")

        if not rephrased_query or rephrased_query.lower() == "not_needed":
            logger.info("No need to perform search based on the rephrased query.")
            return "No search needed for the provided input."

        # Step 2: Perform search
        # Search query parameters
        params = {
            'q': rephrased_query,
            'format': 'json',
            'time_range': time_range,
            'language': language,
            'category': category,
            'engines': ','.join(engines),
            'safesearch': safesearch
        }

        # Remove empty parameters
        params = {k: v for k, v in params.items() if v != ""}

        # If no engines are specified, set default engines
        if 'engines' not in params:
            params['engines'] = 'google'  # Default to 'google' or any preferred engine
            logger.info("No engines specified. Defaulting to 'google'.")

        # Headers for SearXNG request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': 'https://shreyas094-searxng-local.hf.space',
            'Referer': 'https://shreyas094-searxng-local.hf.space/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }

        scraped_content = []
        page = 1
        while len(scraped_content) < num_results:
            # Update params with current page
            params['pageno'] = page

            # Send request to SearXNG
            logger.info(f"Sending request to SearXNG for query: {rephrased_query} (Page {page})")
            session = requests_retry_session()

            try:
                if method.upper() == "GET":
                    response = session.get(SEARXNG_URL, params=params, headers=headers, timeout=10, verify=certifi.where())
                else:  # POST
                    response = session.post(SEARXNG_URL, data=params, headers=headers, timeout=10, verify=certifi.where())
                
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Error during SearXNG request: {e}")
                return f"An error occurred during the search request: {e}"

            search_results = response.json()
            logger.debug(f"SearXNG Response: {search_results}")

            results = search_results.get('results', [])
            if not results:
                logger.warning(f"No more results returned from SearXNG on page {page}.")
                break

            for result in results:
                if len(scraped_content) >= num_results:
                    break
        
                url = result.get('url', '')
                title = result.get('title', 'No title')
        
                if not is_valid_url(url):
                    logger.warning(f"Invalid URL: {url}")
                    continue
        
                try:
                    logger.info(f"Processing content from: {url}")
                    
                    content = scrape_full_content(url, max_chars, timeout, use_pydf2)
                    
                    if content is None:  # This means it's a PDF and use_pydf2 is False
                        continue
                    
                    if not content:
                        logger.warning(f"Failed to scrape content from {url}")
                        continue
                    
                    scraped_content.append({
                        "title": title,
                        "url": url,
                        "content": content,
                        "scraper": "pdf" if url.lower().endswith('.pdf') else "newspaper"
                    })
                    logger.info(f"Successfully scraped content from {url}. Total scraped: {len(scraped_content)}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error scraping {url}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error while scraping {url}: {e}")

            page += 1

        if not scraped_content:
            logger.warning("No content scraped from search results.")
            return "No content could be scraped from the search results."

        logger.info(f"Successfully scraped {len(scraped_content)} documents.")

        # Step 4: Assess relevance, summarize, and check for uniqueness
        relevant_documents = []
        unique_summaries = []
        for doc in scraped_content:
            assessment = assess_relevance_and_summarize(client, rephrased_query, doc, model, temperature=llm_temperature)
            relevance, summary = assessment.split('\n', 1)

            if relevance.strip().lower() == "relevant: yes":
                summary_text = summary.replace("Summary: ", "").strip()
                
                if is_content_unique(summary_text, unique_summaries):
                    relevant_documents.append({
                        "title": doc['title'],
                        "url": doc['url'],
                        "summary": summary_text,
                        "scraper": doc['scraper']
                    })
                    unique_summaries.append(summary_text)
                else:
                    logger.info(f"Skipping similar content: {doc['title']}")

        if not relevant_documents:
            logger.warning("No relevant and unique documents found.")
            return "No relevant and unique news found for the given query."

        # Step 5: Rerank documents based on similarity to query
        reranked_docs = rerank_documents(rephrased_query, relevant_documents, similarity_threshold=0.95, max_results=num_results)
        
        if not reranked_docs:
            logger.warning("No documents remained after reranking.")
            return "No relevant news found after filtering and ranking."
        
        logger.info(f"Reranked and filtered to top {len(reranked_docs)} unique, related documents.")

        # Step 5: Scrape full content for top documents (up to num_results)
        for doc in reranked_docs[:num_results]:
            full_content = scrape_full_content(doc['url'], max_chars)
            doc['full_content'] = full_content
    
        # Prepare JSON for LLM
        llm_input = {
            "query": query,
            "documents": [
                {
                    "title": doc['title'],
                    "url": doc['url'],
                    "summary": doc['summary'],
                    "full_content": doc['full_content']
                } for doc in reranked_docs[:num_results]
            ]
        }

        # Step 6: LLM Summarization
        llm_summary = llm_summarize(json.dumps(llm_input), model, temperature=llm_temperature)
        
        return llm_summary

    except Exception as e:
        logger.error(f"Unexpected error in search_and_scrape: {e}")
        return f"An unexpected error occurred during the search and scrape process: {e}"

# Helper function to get the appropriate client for each model
def get_client_for_model(model: str) -> Any:
    if model == "huggingface":
        return InferenceClient("mistralai/Mistral-Small-Instruct-2409", token=HF_TOKEN)
    elif model == "groq":
        return Groq(api_key=GROQ_API_KEY)
    elif model == "mistral":
        return Mistral(api_key=MISTRAL_API_KEY)
    elif CUSTOM_LLM and (model in fetch_custom_models() or model == CUSTOM_LLM_DEFAULT_MODEL):
        return None  # CustomModel doesn't need a client
    else:
        raise ValueError(f"Unsupported model: {model}")


def chat_function(message: str, history: List[Tuple[str, str]], only_web_search: bool, num_results: int, max_chars: int, time_range: str, language: str, category: str, engines: List[str], safesearch: int, method: str, llm_temperature: float, model: str, use_pydf2: bool):
    chat_history = "\n".join([f"{role}: {msg}" for role, msg in history])
    
    # Create the appropriate AI model
    ai_model = AIModelFactory.create_model(model, get_client_for_model(model))
    
    if only_web_search:
        query_type = "web_search"
    else:
        query_type = determine_query_type(message, chat_history, ai_model)
    
    if query_type == "knowledge_base":
        response = generate_ai_response(message, chat_history, ai_model, llm_temperature)
    else:  # web_search
        gr.Info("Initiating Web Search")
        yield "Request you to sit back and relax until I scrape the web for up-to-date information"
        response = search_and_scrape(
            query=message,
            chat_history=chat_history,
            ai_model=ai_model,
            num_results=num_results,
            max_chars=max_chars,
            time_range=time_range,
            language=language,
            category=category,
            engines=engines,
            safesearch=safesearch,
            method=method,
            llm_temperature=llm_temperature,
            model=model,
            use_pydf2=use_pydf2
        )
    
    yield response

iface = gr.ChatInterface(
    chat_function,
    title="Web Scraper for News with Sentinel AI",
    description="Ask Sentinel any question. It will search the web for recent information or use its knowledge base as appropriate.",
    theme=gr.Theme.from_hub("allenai/gradio-theme"),
    additional_inputs=[
        gr.Checkbox(label="Only do web search", value=True),  # Add this line
        gr.Slider(5, 20, value=3, step=1, label="Number of initial results"),
        gr.Slider(500, 10000, value=1500, step=100, label="Max characters to retrieve"),
        gr.Dropdown(["", "day", "week", "month", "year"], value="week", label="Time Range"),
        gr.Dropdown(["", "all", "en", "fr", "de", "es", "it", "nl", "pt", "pl", "ru", "zh"], value="en", label="Language"),
        gr.Dropdown(["", "general", "news", "images", "videos", "music", "files", "it", "science", "social media"], value="general", label="Category"),
        gr.Dropdown(
            ["google", "bing", "duckduckgo", "baidu", "yahoo", "qwant", "startpage"],
            multiselect=True,
            value=["google", "duckduckgo", "bing", "qwant"],
            label="Engines"
        ),
        gr.Slider(0, 2, value=2, step=1, label="Safe Search Level"),
        gr.Radio(["GET", "POST"], value="GET", label="HTTP Method"),
        gr.Slider(0, 1, value=0.2, step=0.1, label="LLM Temperature"),
        gr.Dropdown(all_models, value=default_model, label="LLM Model"),
        gr.Checkbox(label="Use PyPDF2 for PDF scraping", value=True),
    ],
    additional_inputs_accordion=gr.Accordion("⚙️ Advanced Parameters", open=True),
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    chatbot=gr.Chatbot(
        show_copy_button=True,
        likeable=True,
        layout="bubble",
        height=500,
    )
)

if __name__ == "__main__":
    logger.info("Starting the SearXNG Scraper for News using ChatInterface with Advanced Parameters")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
