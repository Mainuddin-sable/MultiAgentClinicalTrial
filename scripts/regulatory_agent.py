"""
regulatory_agent.py

The Regulatory Agent: checks whether a clinical trial protocol adequately
addresses standard ICH/FDA/eCFR regulatory requirements. Uses retriever.py
for RAG grounding and Gemini for reasoning -- it never invents a
regulatory requirement; every finding must cite retrieved evidence.

Run directly for a demo against a sample protocol description:
    python scripts\\regulatory_agent.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import retrieve_regulatory_evidence

load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")  # override in .env if this changes again

# Deliberately a small, explicit checklist for v1 -- not "review the whole
# protocol in one vague pass." Each item becomes one grounded, auditable
# check, matching how the Biostat Agent uses hard numbers instead of vibes.
REGULATORY_CHECKLIST = [
    {"id": "informed_consent", "topic": "informed consent process and documentation requirements"},
    {"id": "safety_monitoring", "topic": "safety monitoring and adverse event reporting obligations"},
    {"id": "statistical_justification", "topic": "statistical principles for trial design and sample size justification"},
    {"id": "study_design_quality", "topic": "general considerations for clinical study design and quality"},
]

SYSTEM_PROMPT = """You are the Regulatory Agent in a clinical-trial protocol review system.

You check whether a protocol adequately addresses a specific regulatory topic. You are given:
1. The protocol text (or relevant excerpt)
2. Retrieved regulatory evidence chunks (the ONLY source of regulatory requirements you may use)

Rules:
- Base your assessment ONLY on the retrieved evidence provided. Never invent or assume a regulatory requirement that isn't in the evidence.
- If the evidence doesn't clearly cover the topic, say so honestly rather than guessing.
- Cite evidence by its [chunk N] label, exactly as given.
- Respond with ONLY valid JSON, no markdown fences, no preamble, matching this schema:
{
  "status": "pass" | "issue" | "insufficient_evidence",
  "explanation": "2-4 sentences",
  "cited_chunks": [1, 2]
}
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Check that .env exists in the project "
            "root and contains GOOGLE_API_KEY=your_key."
        )
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0)


def _extract_text(content) -> str:
    """Gemini 3.x can return .content as a list of content blocks (e.g.
    including internal reasoning parts) instead of a plain string. Pull
    the actual text out regardless of which shape comes back."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _format_evidence(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[chunk {i}] (source: {c['source_type']} / {c['source_file']})\n{c['text']}")
    return "\n\n".join(lines)


def check_topic(protocol_text: str, topic: str, llm: ChatGoogleGenerativeAI) -> dict:
    evidence = retrieve_regulatory_evidence(topic, k=4)
    evidence_block = _format_evidence(evidence)

    user_prompt = f"""Regulatory topic to check: {topic}

PROTOCOL TEXT:
{protocol_text}

RETRIEVED REGULATORY EVIDENCE:
{evidence_block}

Assess whether the protocol adequately addresses this topic, based only on the evidence above."""

    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt),
    ])

    raw = _extract_text(response.content).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"status": "parse_error", "explanation": raw, "cited_chunks": []}

    # Attach the real source metadata for cited chunks, so a finding
    # traces back to an actual document, not just a bare number.
    cited_sources = []
    for idx in parsed.get("cited_chunks", []):
        if isinstance(idx, int) and 1 <= idx <= len(evidence):
            e = evidence[idx - 1]
            cited_sources.append({"source_file": e["source_file"], "chunk_index": e["chunk_index"]})

    return {
        "topic": topic,
        "status": parsed.get("status", "parse_error"),
        "explanation": parsed.get("explanation", ""),
        "cited_sources": cited_sources,
    }


def run_regulatory_agent(protocol_text: str) -> dict:
    """Runs every checklist item against the protocol. Returns the
    structure this agent contributes to the shared LangGraph state."""
    llm = _get_llm()
    findings = []
    for item in REGULATORY_CHECKLIST:
        result = check_topic(protocol_text, item["topic"], llm)
        result["check_id"] = item["id"]
        findings.append(result)
    return {"agent": "regulatory", "findings": findings}


if __name__ == "__main__":
    # Placeholder protocol text -- your diabetes example from the deck.
    # Swap this out once the PDF protocol parser exists; for now it lets
    # you test the agent end to end.
    sample_protocol = """
Study Title: A Randomized, Placebo-Controlled Trial of Drug X for Type 2 Diabetes

Design: 50 participants, randomized 1:1 to Drug X (n=25) or placebo (n=25).
Primary Endpoint: Change in fasting blood glucose (mg/dL) from baseline to Week 12.
Population: Adults aged 18-65 with confirmed Type 2 Diabetes Mellitus.
Statistical Plan: Two-sample t-test comparing mean glucose reduction between arms,
alpha = 0.05, target power = 80%. Expected effect size: 20 mg/dL difference,
assumed standard deviation 25 mg/dL.
Safety: Adverse events will be recorded by investigators during study visits.
Consent: Written informed consent will be obtained from all participants prior
to enrollment.
"""

    print("Running Regulatory Agent on sample protocol...\n")
    result = run_regulatory_agent(sample_protocol)

    for finding in result["findings"]:
        print(f"\n=== {finding['check_id']} ===")
        print(f"Status: {finding['status']}")
        print(f"Explanation: {finding['explanation']}")
        if finding["cited_sources"]:
            print("Cited sources:")
            for s in finding["cited_sources"]:
                print(f"  - {s['source_file']} (chunk {s['chunk_index']})")