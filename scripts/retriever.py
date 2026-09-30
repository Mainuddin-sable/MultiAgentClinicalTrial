"""
retriever.py

Queries the 'regulatory_guidelines' ChromaDB collection built by
build_vector_index.py. This is the RAG lookup the Regulatory Agent
calls when it needs grounded evidence -- it never invents regulatory
requirements itself, it retrieves the actual text here and reasons
over that.

Usage as a library (this is how the Regulatory Agent will use it):

    from retriever import retrieve_regulatory_evidence

    results = retrieve_regulatory_evidence("informed consent requirements for minors")
    for r in results:
        print(r["source_type"], r["source_file"], r["distance"])
        print(r["text"])

Run directly for a quick manual test with a few example queries:
    python scripts\\retriever.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).resolve().parent.parent / "vector_store"
COLLECTION_NAME = "regulatory_guidelines"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # must match build_vector_index.py exactly

# Loaded once, lazily, so importing this module doesn't eat a few seconds
# every time -- only the first call to retrieve_regulatory_evidence() pays
# the model-loading cost.
_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_regulatory_evidence(query: str, k: int = 5) -> list[dict]:
    """Returns the top-k regulatory chunks most relevant to `query`.

    Each result: {rank, text, source_type, source_file, chunk_index, distance}
    Lower distance = more relevant (this is cosine/L2 distance, not similarity).
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    raw = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for rank, (text, metadata, distance) in enumerate(
        zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]), start=1
    ):
        results.append({
            "rank": rank,
            "text": text,
            "source_type": metadata["source_type"],
            "source_file": metadata["source_file"],
            "chunk_index": metadata["chunk_index"],
            "distance": round(distance, 4),
        })
    return results


def print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("-" * 60)
    if not results:
        print("  (no results -- did you run build_vector_index.py yet?)")
        return
    for r in results:
        preview = r["text"][:200].replace("\n", " ")
        print(f"  [{r['rank']}] {r['source_type']} / {r['source_file']} "
              f"(chunk {r['chunk_index']}, distance {r['distance']})")
        print(f"      {preview}...")


if __name__ == "__main__":
    # A few example queries relevant to a diabetes drug protocol, to sanity
    # check that retrieval is actually pulling relevant regulatory text.
    test_queries = [
        "informed consent requirements for clinical trial participants",
        "statistical principles for determining sample size",
        "general considerations for planning a clinical study",
        "safety monitoring and adverse event reporting",
    ]

    for q in test_queries:
        results = retrieve_regulatory_evidence(q, k=3)
        print_results(q, results)