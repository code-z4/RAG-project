# Customer Service RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot using ChromaDB and Ollama.

The chatbot uses a customer service Q&A knowledge base, converts it into embeddings, stores the embeddings in ChromaDB, retrieves relevant information, and generates responses using the Llama 3.2 model running locally through Ollama.

## Technologies Used

- Python
- ChromaDB
- Sentence Transformers
- Ollama
- Llama 3.2 1B

## Workflow

Document -> Chunking -> Embeddings -> ChromaDB -> Retrieval -> Ollama -> Response

