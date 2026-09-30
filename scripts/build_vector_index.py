"""
build_vector_index.py

Chunks and embeds every document under data/raw/ (eCFR HTML, ICH/FDA PDFs,
ClinicalTrials.gov JSON) and stores them in two local ChromaDB collections.

Run from the project root (with the venv active):
    python scripts\build_vector_index.py

Install requirements first:
    pip install chromadb beautifulsoup4 langchain-text-splitters pypdf sentence-transformers
"""

import json
from collections import Counter
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---- Config ----
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DB_DIR = Path(__file__).resolve().parent.parent / "vector_store"

REGULATORY_SOURCE_TYPES = {"ecfr", "ich", "fda"}
REGULATORY_COLLECTION = "regulatory_guidelines"
REFERENCE_COLLECTION = "trial_examples"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SOURCE_MAP = {
    "ecfr": {"dir": DATA_DIR / "ecfr", "type": "html"},
    "ich": {"dir": DATA_DIR / "ich", "type": "pdf"},
    "fda": {"dir": DATA_DIR / "fda" / "clinical_trial_guidance", "type": "pdf"},
    "clinicaltrials": {"dir": DATA_DIR / "clinicaltrials", "type": "json"},
}


def extract_html_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_json_records(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    studies = data.get("studies", data if isinstance(data, list) else [])
    records = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        nct_id = ident.get("nctId", "unknown")
        desc = protocol.get("descriptionModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        text_parts = [
            ident.get("briefTitle", ""),
            desc.get("briefSummary", ""),
            desc.get("detailedDescription", ""),
            eligibility.get("eligibilityCriteria", ""),
        ]
        records.append((nct_id, "\n".join(p for p in text_parts if p)))
    return records


def load_documents() -> list[dict]:
    docs = []
    for source_type, cfg in SOURCE_MAP.items():
        folder, ftype = cfg["dir"], cfg["type"]

        if not folder.exists():
            print(f"[skip] {folder} does not exist yet")
            continue

        pattern = {"html": "*.html", "pdf": "*.pdf", "json": "*.json"}[ftype]
        files = list(folder.glob(pattern))

        if not files:
            print(f"[skip] {folder} has no {ftype} files yet")
            continue

        for f in files:
            try:
                if ftype == "html":
                    docs.append({"source_type": source_type, "source_file": f.name,
                                 "record_id": f.stem, "text": extract_html_text(f)})
                elif ftype == "pdf":
                    docs.append({"source_type": source_type, "source_file": f.name,
                                 "record_id": f.stem, "text": extract_pdf_text(f)})
                elif ftype == "json":
                    for record_id, text in extract_json_records(f):
                        docs.append({"source_type": source_type, "source_file": f.name,
                                     "record_id": record_id, "text": text})
            except Exception as e:
                print(f"[error] failed to process {f}: {e}")

    return docs


def print_source_breakdown(docs: list[dict]) -> None:
    counts = Counter(d["source_type"] for d in docs)
    for source_type, count in sorted(counts.items()):
        print(f"    {source_type}: {count} document(s)/record(s)")


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in docs:
        if not doc["text"].strip():
            continue
        for i, piece in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "id": f"{doc['source_type']}_{doc['record_id']}_{i}",
                "text": piece,
                "metadata": {
                    "source_type": doc["source_type"],
                    "source_file": doc["source_file"],
                    "record_id": doc["record_id"],
                    "chunk_index": i,
                },
            })
    return chunks


def build_index(chunks: list[dict], collection_name: str, model: SentenceTransformer):
    if not chunks:
        print(f"[skip] no chunks for collection '{collection_name}'")
        return

    MAX_BATCH_SIZE = 4000  # stay comfortably under Chroma's ~5461 hard limit

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks into '{collection_name}' with {EMBEDDING_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    for start in range(0, len(chunks), MAX_BATCH_SIZE):
        end = start + MAX_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  added batch {start}-{min(end, len(chunks))}")

    print(f"Stored {len(chunks)} chunks in collection '{collection_name}' at {DB_DIR}")


if __name__ == "__main__":
    print(f"Scanning {DATA_DIR} ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} source documents/records:")
    print_source_breakdown(documents)

    if not documents:
        print("No documents found. Make sure your downloader scripts have "
              "populated data/raw/ subfolders before running this script.")
        raise SystemExit(1)

    chunks = chunk_documents(documents)
    print(f"Produced {len(chunks)} chunks")

    regulatory_chunks = [c for c in chunks if c["metadata"]["source_type"] in REGULATORY_SOURCE_TYPES]
    reference_chunks = [c for c in chunks if c["metadata"]["source_type"] not in REGULATORY_SOURCE_TYPES]

    model = SentenceTransformer(EMBEDDING_MODEL)
    build_index(regulatory_chunks, REGULATORY_COLLECTION, model)
    build_index(reference_chunks, REFERENCE_COLLECTION, model)