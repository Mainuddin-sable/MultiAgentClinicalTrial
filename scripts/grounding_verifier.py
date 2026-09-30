"""
grounding_verifier.py

Independent grounding/citation verification layer for the Regulatory
Agent. Addresses RAG's best-known failure mode: an LLM confidently
citing a source that doesn't actually support what it claims.

This runs a SEPARATE LLM call, framed purely as a fact-checker with no
stake in the original claim, to independently check whether each cited
chunk genuinely supports the finding it's attached to -- before that
finding reaches the final report.

This is the project's actual novel contribution beyond "agents + RAG":
most systems retrieve and trust. This one retrieves, generates a claim,
and then independently re-checks whether the claim is actually supported
by what was retrieved.

Run directly for a two-part demo:
  1. Verifies a real Regulatory Agent finding against its real evidence
     (should come back grounded=true -- it's genuinely supported).
  2. Deliberately swaps in an UNRELATED evidence chunk for the same claim
     and verifies that instead (should come back grounded=false -- this
     is the "catch a hallucinated/mismatched citation" demo).

    python scripts\\grounding_verifier.py
"""

import json
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import retrieve_regulatory_evidence
from regulatory_agent import run_regulatory_agent, GEMINI_MODEL

load_dotenv()

VERIFIER_SYSTEM_PROMPT = """You are an independent fact-checker reviewing another AI system's citation, not its overall conclusion.

The CLAIM below asserts that certain regulatory requirements exist (e.g.
"the guideline requires X"), and separately comments on whether a protocol
meets them. You are NOT checking whether the protocol meets those
requirements -- the source text can never confirm that, since it is
regulatory guideline text, not the protocol itself. Ignore any part of the
claim describing what the protocol does or doesn't include.

Check ONLY this: does the SOURCE TEXT actually establish the regulatory
requirement(s) the claim attributes to it?

Use ONLY the source text provided -- not your own outside knowledge of
regulations. If the source text is about a different topic entirely, or
doesn't establish the specific requirement being cited, mark it as NOT
grounded.

Respond with ONLY valid JSON, no markdown fences, no preamble:
{
  "grounded": true | false,
  "reasoning": "1-2 sentences explaining your judgment"
}
"""


def _extract_text(content) -> str:
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


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found. Check your .env file.")
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0)


def verify_grounding(claim: str, source_texts: list[str], llm: ChatGoogleGenerativeAI) -> dict:
    """The core verification call. Deliberately a separate, independently
    framed LLM call -- not a re-use of the original agent's prompt/context
    -- so it isn't just rubber-stamping its own earlier output."""
    combined_source = "\n\n---\n\n".join(source_texts)
    user_prompt = f"""CLAIM:
{claim}

SOURCE TEXT:
{combined_source}

Does the source text actually support the claim?"""

    response = llm.invoke([
        ("system", VERIFIER_SYSTEM_PROMPT),
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
        parsed = {"grounded": None, "reasoning": f"parse_error: {raw}"}

    return parsed


if __name__ == "__main__":
    llm = _get_llm()

    print("=" * 60)
    print("PART 1: Verifying a REAL finding (should come back grounded)")
    print("=" * 60)

    sample_protocol = """
Study Title: A Randomized, Placebo-Controlled Trial of Drug X for Type 2 Diabetes
Consent: Written informed consent will be obtained from all participants prior
to enrollment.
"""
    result = run_regulatory_agent(sample_protocol)
    consent_finding = next(f for f in result["findings"] if f["check_id"] == "informed_consent")

    print(f"\nClaim: {consent_finding['explanation']}\n")

    # Re-retrieve the same evidence this finding actually used. cited_sources
    # only stores file/chunk metadata, not full text, so we fetch it fresh.
    real_evidence = retrieve_regulatory_evidence(
        "informed consent process and documentation requirements", k=4
    )
    real_texts = [e["text"] for e in real_evidence]

    verification = verify_grounding(consent_finding["explanation"], real_texts, llm)
    print(f"Verification result: {json.dumps(verification, indent=2)}")

    print("\n" + "=" * 60)
    print("PART 2: Same claim, but with an UNRELATED chunk swapped in")
    print("(the deliberate hallucination-catching demo)")
    print("=" * 60)

    unrelated_evidence = retrieve_regulatory_evidence(
        "statistical power calculation sample size formula", k=1
    )
    fake_texts = [unrelated_evidence[0]["text"]]

    print(f"\nClaim (same as above): {consent_finding['explanation'][:150]}...\n")
    print(f"But 'evidence' is actually from: {unrelated_evidence[0]['source_file']} (about statistics, not consent)\n")

    fake_verification = verify_grounding(consent_finding["explanation"], fake_texts, llm)
    print(f"Verification result: {json.dumps(fake_verification, indent=2)}")

    print("\n" + "=" * 60)
    if verification.get("grounded") is True and fake_verification.get("grounded") is False:
        print("SUCCESS: verifier correctly passed real grounding and caught the mismatch.")
    else:
        print("Check the results above manually -- one or both didn't behave as expected.")
    print("=" * 60)