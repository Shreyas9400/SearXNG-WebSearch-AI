# SearXNG-WebSearch-AI
AI powered Chatbot with real time updates.
# Web Scraper for Financial News with Sentinel AI

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
   - [Search Engine Integration](#search-engine-integration)
   - [AI Models Integration](#ai-models-integration)
   - [Content Processing](#content-processing)
3. [Key Features](#key-features)
   - [Intelligent Query Processing](#intelligent-query-processing)
   - [Content Analysis](#content-analysis)
   - [Search Optimization](#search-optimization)
4. [Architecture](#architecture)
   - [User Interface (UI)](#user-interface-ui)
   - [Query Processing](#query-processing)
   - [Search Engine](#search-engine)
   - [Content Analysis](#content-analysis)
   - [Ranking System](#ranking-system)
   - [Response Generation](#response-generation)
   - [Core Classes](#core-classes)
5. [Main Functions](#main-functions)
6. [API Integration](#api-integration)
8. [Advanced Parameters](#advanced-parameters)

## 1. Overview
This application is a sophisticated web scraper and AI-powered chat interface specifically designed for financial news analysis. It combines web scraping capabilities with multiple Language Learning Models (LLMs) to provide intelligent, context-aware responses to user queries about financial information.

## 2. Core Components

### Search Engine Integration
- Uses SearXNG as the primary search meta-engine
- Supports multiple search engines (Google, Bing, DuckDuckGo, etc.)
- Implements custom retry mechanisms and timeout handling

### AI Models Integration
- Supports multiple LLM providers: Hugging Face (Mistral-Small-Instruct), Groq (Llama-3.1-70b), Mistral AI (Open-Mistral-Nemo)
- Implements semantic similarity using Sentence-Transformer

### Content Processing
- PDF processing with PyPDF2
- Web content scraping with Newspaper3k
- BM25 ranking algorithm implementation
- Document deduplication and relevance assessment

## 3. Key Features

### Intelligent Query Processing
- Query type determination (knowledge base vs. web search)
- Query rephrasing for optimal search results
- Entity recognition
- Time-aware query modification

### Content Analysis
- Relevance assessment
- Content summarization
- Semantic similarity comparison
- Document deduplication
- Priority-based content ranking

### Search Optimization
- Custom retry mechanism
- Rate limiting
- Error handling
- Content filtering and validation

## 4. Architecture

### User Interface (UI)
- You start by interacting with a Gradio Chat Interface.

### Query Processing
- Your query is sent to the Query Analysis (QA) section.
- The system then determines the type of query (DT).
- If it's a type that can use a Knowledge Base, it generates an AI response (KB).
- If it requires web searching, it rephrases the query (QR) for web search.
- The system extracts the entity domain (ED) from the rephrased query.

### Search Engine
- The extracted entity domain is sent to the SearXNG Search Engine (SE).
- The search engine returns the search results (SR).

### Content Analysis
- The search results are processed by web scraping (WS).
- If the content is in PDF format, it is scraped using PDF Scraping (PDF).
- If in HTML format, it's scraped using Newspaper3k Scraping (NEWS).
- Relevant content is summarized (DS) and checked for uniqueness (UC).

### Ranking System
- Content is ranked (DR) based on:
  - **BM25 Scoring (BM):** A scoring method to rank documents.
  - **Semantic Similarity (SS):** How similar the content is to the query.
- The scores are combined (CS) to produce a final ranking (FR).

### Response Generation
- The final ranking is summarized again (FS) to create a final summary.
- The AI-generated response (KB) and final summary (FS) are combined to form the final response.

### Completion
- The final response is sent back to the Gradio Chat Interface (UI) for you to see.

### Core Classes
- **BM25:** Custom implementation for document ranking
- **Search and Scrape Pipeline:** Handles query processing, web search, content scraping, document analysis, and content summarization.

## 5. Main Functions
- **`determine_query_type(query, chat_history, llm_client)`**: Determines whether to use knowledge base or web search based on context.
- **`search_and_scrape(query, chat_history, ...)`**: Main function for web search and content aggregation.
- **`rerank_documents_with_priority(query, documents, entity_domain, ...)`**: Hybrid ranking using BM25 and semantic similarity.
- **`llm_summarize(json_input, model, temperature)`**: Generates summaries using the specified LLM and handles citation and formatting.
## 6. API Integration
- **Required API Keys**: Hugging Face, Groq, Mistral, SearXNG
- **Environment Variables Setup**: Use dotenv to load environment variables
## 8. Advanced Parameters
| **Parameter**            | **Description**                                               | **Range/Options**                          | **Default**   | **Usage**                                                     |
|--------------------------|---------------------------------------------------------------|--------------------------------------------|---------------|---------------------------------------------------------------|
| **Number of Results**     | Number of search results retrieved.                           | 5 to 20                                    | 5             | Controls number of links/articles fetched from web searches.   |
| **Maximum Characters**    | Limits characters per document processed.                     | 500 to 10,000                              | 3000          | Truncates long documents, focusing on relevant information.    |
| **Time Range**            | Specifies the time period for search results.                 | day, week, month, year                     | month         | Filters results based on recent or historical data.            |
| **Language Selection**    | Filters search results by language.                           | `en`, `fr`, `es`, etc.                     | `en`          | Retrieves content in a specified language.                     |
| **LLM Temperature**       | Controls randomness in responses from LLM.                    | 0.0 to 1.0                                 | 0.2           | Low values for factual responses; higher for creative ones.    |
| **Search Engines**        | Specifies search engines used for scraping.                   | Google, Bing, DuckDuckGo, etc.             | All engines   | Choose specific search engines for better or private results.  |
| **Safe Search Level**     | Filters explicit/inappropriate content.                       | 0: No filter, 1: Moderate, 2: Strict       | 2 (Strict)    | Ensures family-friendly or professional content.               |
| **Model Selection**       | Chooses the LLM for summaries or responses.                   | Mistral, GPT-4, Groq                       | Varies        | Select models based on performance or speed.                   |
| **PDF Processing Toggle** | Enables/disables PDF document processing.                     | `True` (process) or `False` (skip)         | `False`       | Processes PDFs, useful for reports but may slow down speed.    |
