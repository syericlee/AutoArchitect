"""
AutoArchitect — LLM Generation

Takes retrieved chunks and sends them to a local LLM (Ollama)
to generate answers with file/line citations.
"""

import requests
from rerank import load_collection, load_models, search_and_rerank

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

SYSTEM_PROMPT = """You are a code assistant that answers questions about a codebase.
Using ONLY the code snippets provided below, answer the question.
Cite specific file paths and line numbers in your answer.
If the snippets don't contain enough information to answer, say so."""


def build_context(chunks: list) -> str:
    """Join the retrieved chunks into a single string for the prompt."""
    snippets = []
    for c in chunks:
        snippet = f"[File: {c['file_path']} | Lines: {c['start_line']}-{c['end_line']}]\n{c['document']}"
        snippets.append(snippet)

    return "\n\n".join(snippets)


def ask_llm(query: str, context: str) -> str:
    """Send the prompt to Ollama and return the answer."""
    prompt = f"{SYSTEM_PROMPT}\n\nSNIPPETS:\n{context}\n\nQUESTION: {query}"

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    })

    return response.json()["response"]


def main(bi_encoder, cross_encoder, collection):
    while True:
        query = input("\nAsk a question about the codebase (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        chunks = search_and_rerank(query, bi_encoder, cross_encoder, collection)
        context = build_context(chunks)
        answer = ask_llm(query, context)
        print(f"\n{answer}")


if __name__ == "__main__":
    collection = load_collection()
    bi_encoder, cross_encoder = load_models()
    main(bi_encoder, cross_encoder, collection)