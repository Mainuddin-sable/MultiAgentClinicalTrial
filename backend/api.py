"""
api.py

FastAPI backend that exposes the LangGraph review pipeline over HTTP, so
the frontend (frontend/index.html) can call it directly instead of using
mock data.

Run from the project root (venv active):
    uvicorn backend.api:app --reload --port 8000

Then open frontend/index.html in a browser and click "Run review" --
it POSTs to http://localhost:8000/review.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the scripts/ folder importable regardless of where uvicorn is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from review_graph import build_graph  # noqa: E402

app = FastAPI(title="Clinical Trial Protocol Review API")

# Allow the static frontend (opened directly as a file, or served from any
# localhost port) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ReviewRequest(BaseModel):
    protocol_text: str


@app.post("/review")
def run_review(request: ReviewRequest):
    graph = _get_graph()
    final_state = graph.invoke({"protocol_text": request.protocol_text})
    return final_state


@app.get("/health")
def health():
    return {"status": "ok"}