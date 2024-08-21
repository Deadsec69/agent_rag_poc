
# POC FastAPI Project

## Overview

This project is a Proof of Concept (POC) built using FastAPI. The main objective is to create an API that interacts with an AI agent to perform web searches and generate content based on the user's queries. The AI agent is trained to write blogs and chat with users using a knowledge base that includes URLs, files (PDFs, and text), and other data sources.

## Features

- **AI Agent Interaction:** Users can interact with an AI agent through a designated endpoint.
- **Content Generation:** The AI agent can generate content based on user queries using various data sources such as PDFs, text files, and URLs.
- **Real-time Response:** The content generation process is optimized to start within 1-2 seconds, and responses are streamed back to the user as they are generated.
- **Markdown Output:** All responses from the AI agent are returned in Markdown format.

## File Structure

```
POC_Task/
│
├── app.py                  # Main FastAPI application file
├── make_requests.py        # Script to handle request making to the AI agent and other services
├── models.py               # Data models and classes used in the application
├── requirements.txt        # Python dependencies required for the project
├── db/
│   ├── chroma.sqlite3       # SQLite database file
│   └── ...                 # Additional database binary files
├── txts/
│   └── oberoi.txt           # Text file used as part of the knowledge base
├── pdfs/
│   ├── Hotel-Brochure-Vivanta-New-Delhi-Dwarka.pdf  # PDF document used as part of the knowledge base
│   └── 16256-114065-m28695768.pdf                  # Another PDF document used as part of the knowledge base
└── __pycache__/           # Compiled Python files for caching
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- FastAPI
- Uvicorn (ASGI server)
- Other dependencies listed in `requirements.txt`
- Add OpenAI API key in app.py line 22
### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-link>
   cd POC_Task
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the API:**
   - The FastAPI application should now be running at `http://127.0.0.1:8000`.
   - You can interact with the API through the provided endpoints.

## Usage

- **Interacting with the AI Agent:**
  - Send a POST request to the `/query` endpoint with your query.
  - The AI agent will respond in Markdown format.

- **Content Generation:**
  - The agent uses data from the PDFs and text files in the `pdfs/` and `txts/` directories to generate content based on user queries.
