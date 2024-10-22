# SearXNG-WebSearch-AI

AI powered Chatbot with real time updates.

# Web Scraper for Financial News with Sentinel AI

## Interface

<img src="https://github.com/user-attachments/assets/6eb31020-dde8-4b89-9666-46bda0b4b4ad" width="1200" alt="Interface Image">

## Demo

https://github.com/user-attachments/assets/37b2c9a2-be0b-46fb-bf6d-628d7ec78e1d

## Setup Instructions

### Prerequisites

Ensure you have the following installed before proceeding:

- Python 3.8 or higher
- Git (for cloning the repository)
- Virtualenv (recommended)

### Step 1: Clone the Repository

First, clone the repository to your local machine.

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name

### Step 2: Set Up a Virtual Environment (Optional but Recommended)

It's a good practice to create a virtual environment for your project to isolate dependencies.

# Create a virtual environment

    python -m venv venv

# Activate the virtual environment

# On Windows:

    venv\Scripts\activate

# On macOS/Linux:

    source venv/bin/activate

### Step 3: Install the Required Dependencies

Install the necessary dependencies specified in the requirements.txt file.

    pip install -r requirements.txt

### Step 4: Set Up Environment Variables

You will need API keys for various integrations used in the project (e.g., Hugging Face, SearXNG). Create a .env file in the root directory of your project and add the necessary environment variables.

Add your API keys in the following format:

HF_TOKEN=your_huggingface_api_token
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
SEARXNG_KEY=your_searxng_secret_key

### Step 5: Running the App

Once you've installed the dependencies and set up your API keys, run the app using:

    python app.py

### Step 6: Launch the App

The app will be hosted locally by default. Open your browser and go to: http://127.0.0.1:7860/

## Setting Up SearXNG Instance

To run your own instance of SearXNG, follow these steps.

### Step 1: Install Dependencies

Ensure you have `docker` and `docker-compose` installed on your system. If not, follow the respective installation guides:

- [Install Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

### Step 2: Clone the SearXNG Repository

Clone the SearXNG Docker repository to your local machine:

    ```bash
    git clone https://github.com/searxng/searxng-docker.git
    cd searxng-docker

### Step 3: Configure SearXNG

You can configure the SearXNG settings by editing the settings.yml file. Adjust parameters like search engines, safe search settings, etc.

To edit the settings, run:

    nano settings.yml

**very important add the "Json" to formats**

    formats:
        - html
        - json

### Step 4: Start the SearXNG Instance

Once the configuration is set, start the SearXNG instance using Docker Compose:

    docker-compose up -d

This will start the SearXNG service in detached mode, meaning it will run in the background.

### Step 5: Access the SearXNG Instance

After the instance starts, you can access SearXNG via your browser at:

    http://localhost:8080

### Step 6: Configure the App to Use the Local SearXNG Instance

If you're integrating SearXNG with another application, you need to point the application to your local SearXNG instance. Update your .env file with the following:

    SEARXNG_KEY=http://localhost:8080

Now, your application will send search requests to your locally hosted SearXNG instance.

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
7. [Advanced Parameters](#advanced-parameters)

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

| **Parameter**             | **Description**                         | **Range/Options**                | **Default** | **Usage**                                               |
| ------------------------------- | --------------------------------------------- | -------------------------------------- | ----------------- | ------------------------------------------------------------- |
| **Number of Results**     | Number of search results retrieved.           | 5 to 20                                | 5                 | Controls number of links/articles fetched from web searches.  |
| **Maximum Characters**    | Limits characters per document processed.     | 500 to 10,000                          | 3000              | Truncates long documents, focusing on relevant information.   |
| **Time Range**            | Specifies the time period for search results. | day, week, month, year                 | month             | Filters results based on recent or historical data.           |
| **Language Selection**    | Filters search results by language.           | `en`, `fr`, `es`, etc.           | `en`            | Retrieves content in a specified language.                    |
| **LLM Temperature**       | Controls randomness in responses from LLM.    | 0.0 to 1.0                             | 0.2               | Low values for factual responses; higher for creative ones.   |
| **Search Engines**        | Specifies search engines used for scraping.   | Google, Bing, DuckDuckGo, etc.         | All engines       | Choose specific search engines for better or private results. |
| **Safe Search Level**     | Filters explicit/inappropriate content.       | 0: No filter, 1: Moderate, 2: Strict   | 2 (Strict)        | Ensures family-friendly or professional content.              |
| **Model Selection**       | Chooses the LLM for summaries or responses.   | Mistral, GPT-4, Groq                   | Varies            | Select models based on performance or speed.                  |
| **PDF Processing Toggle** | Enables/disables PDF document processing.     | `True` (process) or `False` (skip) | `False`         | Processes PDFs, useful for reports but may slow down speed.   |

## Docker Setup and Usage

This project uses Docker and Docker Compose for easy setup and deployment. Follow these steps to get the application running:

### Prerequisites

- Docker
- Docker Compose

### Setup

1. Clone this repository to your local machine.
2. Navigate to the project directory:

   ```
   cd path/to/project
   ```
3. Create a `.env` file from the example:

   ```
   cp .env-example .env
   ```
4. Open the `.env` file in a text editor and modify the values as needed for your local setup. Pay special attention to:

   - `PORT`: The port on which the application will run (default is 5000)
   - `SEARXNG_KEY`: The URL for your SearXNG instance (default is http://localhost:8080)

### Building and Running

1. Build and start the Docker containers:

   ```
   docker-compose up --build
   ```

   This command will build the Docker image and start the container. The `--build` flag ensures that the image is rebuilt if there are any changes.
2. The application should now be running and accessible at `http://localhost:5000` (or whatever port you specified in the `.env` file).
3. To stop the application, press `Ctrl+C` in the terminal where docker-compose is running.
4. To run the containers in detached mode (in the background):

   ```
   docker-compose up -d
   ```
5. To stop and remove the containers when running in detached mode:

   ```
   docker-compose down
   ```

### Rebuilding

If you make changes to the application code or Dockerfile, you'll need to rebuild the Docker image:

```
docker-compose up --build
```

### Viewing Logs

If running in detached mode, you can view the logs with:

```
docker-compose logs -f
```

### Troubleshooting

- If you encounter any issues, ensure that the required ports are not being used by other applications.
- Check that all necessary environment variables are correctly set in your `.env` file.
- Verify that Docker and Docker Compose are correctly installed and up to date.

For more detailed information on Docker Compose commands, refer to the [official Docker Compose documentation](https://docs.docker.com/compose/).