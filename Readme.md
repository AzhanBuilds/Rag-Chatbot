
🤖 RAG Chatbot

Description: An AI-powered Retrieval-Augmented Generation chatbot that retrieves relevant information from documents using FAISS and Sentence Transformers, then uses the Gemini API to generate answers through a Flask web interface.

🌐 Live Demo

Try the chatbot online:

Live Demo: https://your-live-demo-url.com

📌 Project Overview

RAG Chatbot is an AI-powered chatbot built using Retrieval-Augmented Generation (RAG).

The chatbot can answer user questions using information from provided documents. Instead of directly asking the AI model to generate an answer, the system first searches the available documents for relevant information and then provides that information to the AI model to generate the final response.

This makes the chatbot more useful for answering questions based on specific knowledge or documents.

🎯 Project Objective

The main objective of this project is to build a chatbot that can:

Accept questions from users
Search information from provided documents
Find the most relevant content
Use the retrieved content as context
Generate a natural-language answer using Gemini
Display the answer through a web interface
🧠 What is RAG?

RAG stands for Retrieval-Augmented Generation.

It combines two main processes:

1. Retrieval

The system searches the available documents and retrieves information that is relevant to the user's question.

2. Generation

The retrieved information is given to the Gemini AI model along with the user's question. Gemini uses this context to generate the final answer.

Simple Formula
RAG = Retrieval + Context + Generation
🔄 Complete Project Workflow

The project works in two stages.

Stage 1: Document Processing
Document
   ↓
Load Document
   ↓
Split Text into Chunks
   ↓
Create Embeddings
   ↓
Store Embeddings in FAISS
   ↓
Save Vector Store
Step 1: Load Documents

The documents are stored inside the documents folder.

Example:

documents/
└── ai.txt

The application reads the text from the document.

Step 2: Split Text

Large documents are divided into smaller pieces called chunks.

This makes it easier to search for specific information.

Large Document
      ↓
 ┌─────────┐
 │ Chunk 1 │
 ├─────────┤
 │ Chunk 2 │
 ├─────────┤
 │ Chunk 3 │
 └─────────┘
Step 3: Create Embeddings

Each chunk is converted into a numerical representation called an embedding.

The project uses:

all-MiniLM-L6-v2

from Sentence Transformers.

Embeddings help the system compare the meaning of the user's question with the meaning of document chunks.

Step 4: Store in FAISS

The embeddings are stored in a FAISS index.

FAISS is used for fast similarity searching.

The vector store contains:

vectorstore/
├── index.faiss
└── chunks.pkl

index.faiss contains the vector index.

chunks.pkl contains the original document chunks.

💬 Stage 2: Question Answering

When the user asks a question, the following process takes place:

User Question
      ↓
Flask Application
      ↓
Create Question Embedding
      ↓
FAISS Similarity Search
      ↓
Retrieve Relevant Chunks
      ↓
Create Context
      ↓
Gemini API
      ↓
Generate Answer
      ↓
Flask
      ↓
Display Answer
Step 1: User Asks a Question

For example:

What is Artificial Intelligence?

The question is received by the Flask application.

Step 2: Convert Question into Embedding

The Sentence Transformer model converts the question into an embedding.

User Question
      ↓
Sentence Transformer
      ↓
Question Embedding
Step 3: Search FAISS

The question embedding is compared with the embeddings stored in FAISS.

FAISS finds the most similar document chunks.

Step 4: Retrieve Relevant Information

The most relevant chunks are retrieved from the document.

This is the Retrieval part of RAG.

Step 5: Send Context to Gemini

The user's question and retrieved information are sent to the Gemini API.

Question
   +
Retrieved Context
   ↓
Gemini API
Step 6: Generate the Answer

Gemini generates a natural-language response using the retrieved information.

Step 7: Display the Answer

The response is returned to Flask and displayed on the chatbot webpage.

# 🏗️ Project Architecture

```
                    ┌──────────────────┐
                    │      User        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Flask Web App    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  RAG Pipeline    │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │ Sentence        │     │     FAISS       │
        │ Transformer     │     │ Vector Search   │
        └─────────────────┘     └────────┬────────┘
                                         │
                                         ▼
                                Relevant Context
                                         │
                                         ▼
                                ┌─────────────────┐
                                │   Gemini API    │
                                └────────┬────────┘
                                         │
                                         ▼
                                Generated Answer
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Chat Interface  │
                                └─────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

- Python
- Flask

### Frontend

- HTML5
- CSS3
- JavaScript

### AI / RAG

- Retrieval-Augmented Generation (RAG)
- Text Embeddings
- Vector Similarity Search
- Large Language Model (LLM)

### Tools

- Git
- GitHub
- Python Virtual Environment (venv)
- Markdown
