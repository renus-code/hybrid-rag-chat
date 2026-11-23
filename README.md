# Hybrid RAG Chat Application

A Retrieval-Augmented Generation (RAG) application built using LangChain, Groq, ChromaDB, BM25, and FlashRank.

This application uses **Hybrid Search** to retrieve relevant information from local text documents, applies **Maximal Marginal Relevance (MMR)** for diverse vector retrieval, and uses **FlashRank** to rerank retrieved documents before generating answers using an LLM.

## Features

- **Document Loading:** Loads `.txt` documents from a local directory.
- **Text Splitting:** Uses `RecursiveCharacterTextSplitter` to divide documents into manageable chunks with overlapping content.
- **Semantic Search:** Uses FastEmbed embeddings and ChromaDB for vector-based retrieval.
- **Keyword Search:** Uses BM25 to retrieve documents based on keyword relevance.
- **Hybrid Retrieval:** Combines BM25 and vector search using `EnsembleRetriever`.
- **MMR Retrieval:** Improves retrieval diversity by reducing redundant results.
- **Reranking:** Uses FlashRank to rerank retrieved documents and select the most relevant results.
- **Contextual Compression:** Uses `ContextualCompressionRetriever` to refine retrieval results.
- **LLM Integration:** Uses Groq's GPT-OSS-20B model for answer generation.
- **Source Attribution:** Includes source filenames in the generated context.
- **Interactive UI:** Provides a Streamlit interface for asking questions.

## Architecture

```text
              Local Text Documents
                       |
                       v
                Document Loading
                       |
                       v
                 Text Splitting
                       |
                       v
                Document Chunks
                       |
              +--------+--------+
              |                 |
              v                 v
         BM25 Search       Vector Search
                                |
                         ChromaDB + FastEmbed
                                |
                              MMR
              |                 |
              +--------+--------+
                       |
                       v
                Ensemble Retriever
                  (Hybrid Search)
                       |
                       v
                  FlashRank
                   Reranker
                       |
                       v
             Contextual Compression
                       |
                       v
                Retrieved Context
                       |
                       v
                 Groq LLM
                       |
                       v
                Generated Answer
                       |
                       v
                  Streamlit UI
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Groq GPT-OSS-20B |
| LLM Framework | LangChain |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | EnsembleRetriever |
| Vector Search Strategy | MMR |
| Reranking | FlashRank |
| Compression | ContextualCompressionRetriever |
| User Interface | Streamlit |

## Project Structure

```text
hybrid-rag-chat/
│
├── new_articles/
│   └── *.txt
│
├── rag_pipeline.py
├── app.py
├── requirements.txt
├── .env
└── README.md
```

## Configuration

The main retrieval and model configurations are defined in `rag_pipeline.py`.

| Parameter | Value |
|---|---|
| Chunk Size | 1000 |
| Chunk Overlap | 200 |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| LLM Model | openai/gpt-oss-20b |
| Reranker Model | ms-marco-MiniLM-L-12-v2 |
| Retrieval K | 20 |
| Fetch K | 40 |
| Rerank Top N | 4 |

## How Hybrid Retrieval Works

The application combines two retrieval approaches:

**1. BM25 Retrieval**

Uses keyword-based search to identify documents containing relevant terms from the user's query.

**2. Vector Retrieval**

Converts documents and queries into embeddings and uses semantic similarity to identify relevant content. MMR helps retrieve diverse results.

**3. Ensemble Retrieval**

Combines BM25 and vector retrieval using equal weights:

```python
weights=[0.5, 0.5]
```

**4. FlashRank Reranking**

The retrieved documents are reranked using `ms-marco-MiniLM-L-12-v2`, retaining the top four results for the generation stage.

## Answer Generation

The retrieved documents are formatted with their source filenames and passed to the Groq LLM.

The application follows a context-restricted prompt:

- Answers must be based only on the retrieved context.
- Unsupported questions should receive a predefined fallback response.
- Answers should be concise and factual.
- Source filenames are included in the context for attribution.