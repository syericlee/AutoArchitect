"""
AutoArchitect — Embedding and Vector Index

Encodes chunks into vectors and stores them in ChromaDB.
Reads chunks.json, writes to chroma_db/.
"""

import json
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "httpx_codebase"


def load_model(model_name: str) -> SentenceTransformer:
    """Load the bi-encoder model and return it."""
    model = SentenceTransformer(model_name)
    return model


def build_index(chunks: list, model: SentenceTransformer) -> chromadb.Collection:
    """Embed all chunks and store them in ChromaDB."""
    texts = [chunk["content"] for chunk in chunks]
    vectors = model.encode(texts)

    client = chromadb.PersistentClient("./chroma_db")
    existing = [c.name for c in client.list_collections()]

    # Wipe and rebuild to avoid duplicates
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    
    # Cosine similarity matches how sentence-transformers vectors are trained
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=vectors.tolist(),
        metadatas=[{
            "file_path": chunks[i]["file_path"],
            "start_line": chunks[i]["start_line"],
            "end_line": chunks[i]["end_line"],
            "language": chunks[i]["language"]
        } for i in range(len(chunks))],
        documents=[chunks[i]["content"] for i in range(len(chunks))]
    )

    return collection


def search(query: str, collection: chromadb.Collection, model: SentenceTransformer, n_results: int = 5, verbose: bool = True):
    """Search ChromaDB for chunks most similar to the query."""
    query_vector = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )

    if verbose:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            # Convert distance to similarity (0 = unrelated, 1 = identical)
            similarity = 1 - distance
            file_path = metadata["file_path"]
            lines = f"{metadata['start_line']}-{metadata['end_line']}"
            print(f"#{i + 1} [{similarity:.3f}]  {file_path}  (lines {lines})")

    return results

if __name__ == "__main__":
    with open("chunks.json", "r") as file:
        chunks = json.load(file)

    model = load_model(MODEL_NAME)
    collection = build_index(chunks, model)
    search("Where is authentication implemented?", collection, model)