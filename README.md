# AutoArchitect

RAG system for asking natural language questions about a codebase. Point it at a GitHub repo, it indexes the code, and you can query it in plain English. Answers include file paths and line numbers.

## How It Works

1. **Ingest** - clone a repo, filter out junk (node_modules, .git, lock files, etc.)
2. **Chunk** - split files into overlapping 50-line windows
3. **Embed** - run chunks through a bi-encoder to get vectors, store in ChromaDB
4. **Rerank** - bi-encoder pulls 20 candidates, cross-encoder rescores and keeps the top 5
5. **Generate** - feed the top chunks to a local LLM (Ollama) which writes an answer with citations

## Setup

You need Python 3.10+, Git, and [Ollama](https://ollama.com).

```bash
pip install sentence-transformers chromadb requests
ollama pull llama3.2
```

## Usage

Build the index (once per repo):
```bash
python ingest.py
python chunk.py
python embed.py
```

Then ask questions:
```bash
python generate.py
```

```
Ask a question about the codebase (or 'quit' to exit): How is HTTP/2 configured?

HTTP/2 is configured by instantiating a client with `http2=True`:

    client = httpx.AsyncClient(http2=True)

This configuration is available on both the Client and AsyncClient classes.
(Source: docs/http2.md, lines 1-50; httpx/_transports/default.py, lines 1-50)
```

## Project Structure

```
ingest.py       Clone repo, filter and read source files
chunk.py        Split files into overlapping chunks
embed.py        Embed chunks with bi-encoder, store in ChromaDB
rerank.py       Two-stage retrieval (bi-encoder + cross-encoder)
generate.py     Send retrieved chunks to LLM, print answer
```

## Models

- **all-MiniLM-L6-v2** - bi-encoder for embedding (~80MB)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** - cross-encoder for reranking (~80MB)
- **llama3.2** - local LLM for answer generation (~2GB)

## Configuration

Each file has constants at the top: chunk size, overlap, which extensions to index, which model to use, etc.