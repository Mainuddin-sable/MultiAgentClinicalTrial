"""
review_graph.py

The LangGraph orchestration layer. Runs the Regulatory, Biostatistics, and
Clinical agents against a protocol, merges their findings into shared
state, and produces a final decision: approved, or needs revision.

SCOPE NOTE (read this): this version does NOT yet auto-rewrite the
protocol and loop back. It runs each agent once, aggregates findings, and
decides. A real "revision cycle" -- where a Revision Agent rewrites the
protocol based on findings and the graph re-runs automatically -- is a
separate, later piece of work. Building that honestly requires its own
agent and its own testing, not just a loop-back edge bolted on here.

Also note: the Biostatistics Agent still needs its numeric inputs passed
explicitly (proposed sample size, expected effect, SD) rather than
extracted from protocol_text -- that's what the document parser will do
once it exists. For now this graph hardcodes the same demo values used
in biostat_agent.py directly, matching the sample protocol.

Run directly for a demo:
    python scripts\\review_graph.py
"""

from typing import TypedDict
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from regulatory_agent import run_regulatory_agent, GEMINI_MODEL
from biostat_agent import run_biostat_agent
from clinical_agent import run_clinical_agent
from retriever import retrieve_regulatory_evidence
from grounding_verifier import verify_grounding

load_dotenv()


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found. Check your .env file.")
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0)


class ReviewState(TypedDict):
    protocol_text: str
    regulatory_findings: list[dict]
    biostat_result: dict
    clinical_findings: list[dict]
    decision: str
    issues_summary: list[str]


def regulatory_node(state: ReviewState) -> dict:
    result = run_regulatory_agent(state["protocol_text"])
    return {"regulatory_findings": result["findings"]}


def biostat_node(state: ReviewState) -> dict:
    # Hardcoded to match the sample protocol for now -- see SCOPE NOTE above.
    result = run_biostat_agent(
        proposed_n_per_group=25,
        expected_effect=20,
        expected_sd=25,
        alpha=0.05,
        target_power=0.80,
    )
    return {"biostat_result": result}


def clinical_node(state: ReviewState) -> dict:
    result = run_clinical_agent(state["protocol_text"])
    return {"clinical_findings": result["findings"]}


def verify_node(state: ReviewState) -> dict:
    """Independently re-checks each regulatory finding's citation before it
    reaches the final report. A finding that fails verification is NOT
    silently dropped -- it stays visible, but flagged, so a human reviewer
    sees exactly what the system itself isn't confident about."""
    llm = _get_llm()
    verified_findings = []

    for f in state["regulatory_findings"]:
        f = dict(f)  # don't mutate the original
        if f["status"] not in ("issue", "pass"):
            f["grounding_verified"] = None
            verified_findings.append(f)
            continue

        evidence = retrieve_regulatory_evidence(f["topic"], k=4)
        texts = [e["text"] for e in evidence]
        verification = verify_grounding(f["explanation"], texts, llm)

        f["grounding_verified"] = verification.get("grounded")
        f["verification_reasoning"] = verification.get("reasoning", "")
        verified_findings.append(f)

    return {"regulatory_findings": verified_findings}


def aggregate_node(state: ReviewState) -> dict:
    """Deterministic decision logic -- NOT an LLM judging the other LLMs.
    A protocol needs revision if any check anywhere came back 'issue'."""
    issues_summary = []

    for f in state["regulatory_findings"]:
        if f["status"] == "issue":
            line = f"[Regulatory] {f['topic']}: {f['explanation']}"
            if f.get("grounding_verified") is False:
                line += "\n   ⚠️ CITATION NOT INDEPENDENTLY VERIFIED -- review this one manually."
            issues_summary.append(line)

    biostat = state["biostat_result"]
    if biostat["status"] == "issue":
        issues_summary.append(f"[Biostatistics] {biostat['explanation']}")

    for f in state["clinical_findings"]:
        if f["status"] in ("issue", "partial"):
            issues_summary.append(f"[Clinical] {f['label']}: {f['explanation']}")

    decision = "needs_revision" if issues_summary else "approved"

    return {"decision": decision, "issues_summary": issues_summary}


def build_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("regulatory", regulatory_node)
    graph.add_node("biostat", biostat_node)
    graph.add_node("clinical", clinical_node)
    graph.add_node("verify", verify_node)
    graph.add_node("aggregate", aggregate_node)

    # Fan out: all three agents run off the same starting state.
    graph.add_edge(START, "regulatory")
    graph.add_edge(START, "biostat")
    graph.add_edge(START, "clinical")

    # Regulatory findings get independently verified before aggregation.
    graph.add_edge("regulatory", "verify")

    # Fan in: aggregate only runs once verification, biostat, and clinical
    # have all completed.
    graph.add_edge("verify", "aggregate")
    graph.add_edge("biostat", "aggregate")
    graph.add_edge("clinical", "aggregate")

    graph.add_edge("aggregate", END)

    return graph.compile()


if __name__ == "__main__":
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

    print("Running full multi-agent review via LangGraph...\n")
    app = build_graph()
    final_state = app.invoke({"protocol_text": sample_protocol})

    print(f"\n{'=' * 60}")
    print(f"FINAL DECISION: {final_state['decision'].upper()}")
    print(f"{'=' * 60}\n")

    if final_state["issues_summary"]:
        print(f"{len(final_state['issues_summary'])} issue(s) found:\n")
        for i, issue in enumerate(final_state["issues_summary"], start=1):
            print(f"{i}. {issue}\n")
    else:
        print("No issues found across any agent.")