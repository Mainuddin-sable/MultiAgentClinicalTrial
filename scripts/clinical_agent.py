"""
clinical_agent.py

The Clinical Agent: checks whether a protocol's clinical design makes
sense -- population/eligibility, intervention/comparator, endpoint
relevance, and safety monitoring from a clinical (not regulatory-citation)
standpoint.

Unlike the Regulatory Agent, there's no single authoritative external
source to retrieve from here -- clinical design soundness is closer to
expert judgment. To keep this agent auditable rather than a vague "does
this look okay" pass, each check is graded against an explicit, hardcoded
rubric of what a well-designed protocol should include for that item.

Run directly for a demo against the sample protocol:
    python scripts\\clinical_agent.py
"""

import json
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

CLINICAL_CHECKLIST = [
    {
        "id": "population_eligibility",
        "label": "Patient population & eligibility criteria",
        "rubric": (
            "A well-specified population/eligibility section should include: clear inclusion "
            "criteria (diagnosis confirmation method, relevant disease severity/stage), clear "
            "exclusion criteria (contraindications, confounding conditions, comorbidities that "
            "could bias results), and justification for any age range or demographic restriction."
        ),
    },
    {
        "id": "intervention_comparator",
        "label": "Intervention & comparator appropriateness",
        "rubric": (
            "The intervention should be clearly specified (dose, route, frequency, duration). "
            "The comparator (placebo or active control) should be justified as appropriate for "
            "the condition and standard of care -- e.g. is placebo ethical here, or should this "
            "be an active-comparator trial?"
        ),
    },
    {
        "id": "endpoint_relevance",
        "label": "Primary endpoint clinical relevance",
        "rubric": (
            "The primary endpoint should be clinically meaningful to patients/practice (not just "
            "statistically convenient), objectively measurable, and assessed on a clearly defined "
            "timeline appropriate to the condition and mechanism of the intervention."
        ),
    },
    {
        "id": "safety_monitoring_clinical",
        "label": "Safety monitoring procedures (clinical adequacy)",
        "rubric": (
            "From a clinical standpoint (not just regulatory reporting obligations): are there "
            "defined monitoring visits/intervals, pre-specified stopping rules or dose-modification "
            "criteria, and monitoring appropriate to the known or plausible risk profile of the "
            "intervention?"
        ),
    },
]

SYSTEM_PROMPT = """You are the Clinical Agent in a clinical-trial protocol review system.

You assess whether a protocol's design is clinically sound for ONE specific
aspect at a time, graded strictly against the rubric you are given for that
aspect. Do not introduce criteria outside the rubric. Do not be vague --
if the protocol is missing something the rubric calls for, say exactly what
is missing.

Respond with ONLY valid JSON, no markdown fences, no preamble, matching:
{
  "status": "pass" | "issue" | "partial",
  "explanation": "2-4 sentences, specific to what the rubric asked for"
}
"""


def _extract_text(content) -> str:
    """Gemini 3.x can return .content as a list of content blocks instead
    of a plain string. Pull the actual text out regardless of shape."""
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


def check_item(protocol_text: str, item: dict, llm: ChatGoogleGenerativeAI) -> dict:
    user_prompt = f"""Aspect being checked: {item['label']}

RUBRIC FOR THIS ASPECT:
{item['rubric']}

PROTOCOL TEXT:
{protocol_text}

Assess this protocol against the rubric above, for this aspect only."""

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
        parsed = {"status": "parse_error", "explanation": raw}

    return {
        "check_id": item["id"],
        "label": item["label"],
        "status": parsed.get("status", "parse_error"),
        "explanation": parsed.get("explanation", ""),
    }


def run_clinical_agent(protocol_text: str) -> dict:
    llm = _get_llm()
    findings = [check_item(protocol_text, item, llm) for item in CLINICAL_CHECKLIST]
    return {"agent": "clinical", "findings": findings}


if __name__ == "__main__":
    # Same sample protocol used by the other two agents, for consistency.
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

    print("Running Clinical Agent on sample protocol...\n")
    result = run_clinical_agent(sample_protocol)

    for finding in result["findings"]:
        print(f"\n=== {finding['label']} ===")
        print(f"Status: {finding['status']}")
        print(f"Explanation: {finding['explanation']}")