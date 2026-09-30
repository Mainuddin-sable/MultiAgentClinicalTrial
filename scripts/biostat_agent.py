"""
biostat_agent.py

The Biostatistics Agent: checks whether a protocol's proposed sample size
is adequate to detect its stated treatment effect. The actual power/
sample-size math is done by statsmodels (deterministic, real statistics)
-- the LLM's only job is to interpret and write up numbers it is handed,
never to compute or invent them itself.

Run directly for a demo against the sample protocol's stats plan:
    python scripts\\biostat_agent.py
"""

import json
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from statsmodels.stats.power import TTestIndPower

load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = """You are the Biostatistics Agent in a clinical-trial protocol review system.

You are given a protocol's proposed sample size AND a set of numbers already
computed by a Python statistical library (not by you). Your job is ONLY to
interpret and explain these numbers in plain language for a reviewer -- you
must NOT recompute, alter, or invent any numeric value. Use the numbers
exactly as given.

Respond with ONLY valid JSON, no markdown fences, no preamble, matching:
{
  "status": "pass" | "issue",
  "explanation": "2-4 sentences, referencing the actual numbers given"
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


def compute_power_analysis(
    proposed_n_per_group: int,
    expected_effect: float,
    expected_sd: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> dict:
    """The actual deterministic calculation -- this is the part that must
    never be handed to an LLM to guess. Returns both directions:
    (a) power actually achieved at the protocol's proposed sample size, and
    (b) the sample size that WOULD be required to hit the target power."""
    cohens_d = expected_effect / expected_sd
    analysis = TTestIndPower()

    achieved_power = analysis.power(
        effect_size=cohens_d, nobs1=proposed_n_per_group, alpha=alpha, ratio=1.0
    )
    required_n_per_group = analysis.solve_power(
        effect_size=cohens_d, alpha=alpha, power=target_power, ratio=1.0
    )

    return {
        "cohens_d": round(cohens_d, 3),
        "proposed_n_per_group": proposed_n_per_group,
        "achieved_power": round(float(achieved_power), 3),
        "target_power": target_power,
        "required_n_per_group": int(round(required_n_per_group)),
        "alpha": alpha,
    }


def run_biostat_agent(
    proposed_n_per_group: int,
    expected_effect: float,
    expected_sd: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> dict:
    stats = compute_power_analysis(proposed_n_per_group, expected_effect, expected_sd, alpha, target_power)

    llm = _get_llm()
    user_prompt = f"""Protocol's proposed sample size: {stats['proposed_n_per_group']} participants per group
Expected treatment effect: {expected_effect}
Expected standard deviation: {expected_sd}
Significance level (alpha): {stats['alpha']}
Target power: {stats['target_power']}

PYTHON-COMPUTED RESULTS (use these exact numbers, do not recompute):
- Cohen's d (standardized effect size): {stats['cohens_d']}
- Statistical power actually achieved at the proposed sample size: {stats['achieved_power']}
- Sample size per group that WOULD be required to reach {int(stats['target_power']*100)}% power: {stats['required_n_per_group']}

Assess whether the proposed sample size is statistically adequate."""

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
        "agent": "biostat",
        "check_id": "sample_size_power",
        "status": parsed.get("status", "parse_error"),
        "explanation": parsed.get("explanation", ""),
        "computed_stats": stats,
    }


if __name__ == "__main__":
    # Matches the diabetes example: 25 per arm, expected effect 20 mg/dL,
    # assumed SD 25 mg/dL, alpha 0.05, target power 80%.
    # NOTE: these values are hardcoded here for now. Once the protocol PDF
    # parser exists, it will extract these automatically instead of you
    # typing them in.
    print("Running Biostatistics Agent on sample protocol's stats plan...\n")
    result = run_biostat_agent(
        proposed_n_per_group=25,
        expected_effect=20,
        expected_sd=25,
        alpha=0.05,
        target_power=0.80,
    )

    print(f"Status: {result['status']}")
    print(f"Explanation: {result['explanation']}\n")
    print("Computed stats (from Python, not the LLM):")
    for k, v in result["computed_stats"].items():
        print(f"  {k}: {v}")