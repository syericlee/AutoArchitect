"""
AutoArchitect — Retrieval with Reranking

Two-stage retrieval: bi-encoder finds 20 candidates fast,
cross-encoder rescores them precisely, returns the top 5.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from embed import search

# Configuration
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "httpx_codebase"
CANDIDATES = 20    # How many the bi-encoder retrieves
TOP_N = 5          # How many the cross-encoder keeps

def load_collection() -> chromadb.Collection:
    """Connect to the existing ChromaDB database."""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(COLLECTION_NAME)

def load_models() -> tuple[SentenceTransformer, CrossEncoder]:
    """Load both the bi-encoder and cross-encoder."""
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return bi_encoder, cross_encoder


def rerank(query: str, candidates: list, cross_encoder: CrossEncoder, top_n: int = TOP_N) -> list:
    """Score each candidate with the cross-encoder, return top_n sorted by relevance."""

    pairs = [[query, candidate] for candidate in candidates]
    scores = cross_encoder.predict(pairs)
    
    # Keep track of original positions so we can grab the right metadata later
    scores_with_indices = list(zip(scores, range(len(scores))))
    scores_with_indices.sort(key=lambda x: x[0], reverse=True)

    return scores_with_indices[:top_n]


def search_and_rerank(query: str, bi_encoder: SentenceTransformer, cross_encoder: CrossEncoder, collection: chromadb.Collection) -> list:
    """Bi-encoder retrieves candidates, cross-encoder picks the best ones."""

    results = search(query, collection, bi_encoder, n_results=CANDIDATES, verbose=False)

    candidates = [results["documents"][0][i] for i in range(len(results["ids"][0]))]
    
    top_candidates = rerank(query, candidates, cross_encoder)

    final_results = []
    for score, index in top_candidates:
        metadata = results["metadatas"][0][index]
        document = results["documents"][0][index]
        file_path = metadata["file_path"]
        lines = f"{metadata['start_line']}-{metadata['end_line']}"
        print(f"[{score:.3f}]  {file_path}  (lines {lines})")

        final_results.append({
            "score": float(score),
            "document": document,
            "file_path": file_path,
            "start_line": metadata["start_line"],
            "end_line": metadata["end_line"],
        })

    return final_results

if __name__ == "__main__":
    collection = load_collection()
    bi_encoder, cross_encoder = load_models()
    query = "Where is auth implemented?"
    search_and_rerank(query, bi_encoder, cross_encoder, collection)