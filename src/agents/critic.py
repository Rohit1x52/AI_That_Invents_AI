import json
import os
import re
import logging
from typing import Dict, Any, List, Optional

# =========================
# LLM Prompt
# =========================
CRITIC_PROMPT = """
You are a senior deep learning researcher.

Evaluate the following neural network architecture.

You MUST return a valid JSON object only.

Return:
{
  "scores": {
    "efficiency": 0-10,
    "expressiveness": 0-10,
    "training_stability": 0-10,
    "hardware_friendliness": 0-10,
    "overall": 0-10
  },
  "verdict": "poor | acceptable | promising",
  "findings": [string],
  "suggestions": [
    {"action": string, "target": string, "reason": string}
  ],
  "explanation": string
}

Architecture Blueprint:
{blueprint}
"""

# =========================
# Heuristic Critic (Fast, Deterministic)
# =========================
def _heuristic_critic(
    bp: Dict[str, Any],
    dna: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    stages = bp.get("stages", [])
    depth = sum(int(s.get("depth", 1)) for s in stages)
    max_filters = max((int(s.get("filters", 0)) for s in stages), default=0)

    params = (metrics or {}).get("params", 0)
    acc = (metrics or {}).get("val_acc", 0.0)
    latency = (metrics or {}).get("latency_cpu_ms")

    findings: List[str] = []
    suggestions: List[Dict[str, str]] = []

    score = 6.0

    # ---- Structural reasoning
    if depth > 20:
        findings.append("Excessive depth may hurt latency and stability")
        suggestions.append({"action": "reduce_depth", "target": "stages", "reason": "too deep"})
        score -= 1.0

    if max_filters > 512:
        findings.append("Very wide layers may be overkill")
        suggestions.append({"action": "reduce_filters", "target": "late_stages", "reason": "over-parameterized"})
        score -= 1.0

    has_bottleneck = any(s.get("type") == "bottleneck_block" for s in stages)
    if has_bottleneck:
        findings.append("Bottleneck blocks improve parameter efficiency")
        score += 1.0

    # ---- Metric-based reasoning
    if params and params > 1_500_000 and acc < 0.6:
        findings.append("High parameter count without accuracy gain")
        suggestions.append({"action": "shrink", "target": "global", "reason": "inefficient capacity"})
        score -= 1.5

    if latency and latency > 20:
        findings.append("Latency exceeds reasonable CPU budget")
        suggestions.append({"action": "prefer_shallow", "target": "global", "reason": "latency constraint"})
        score -= 1.0

    if acc >= 0.75:
        findings.append("Architecture performs well on validation")
        suggestions.append({"action": "exploit_neighborhood", "target": "global", "reason": "promising design"})
        score += 2.0

    # ---- Normalize
    score = max(0.0, min(10.0, score))

    verdict = (
        "promising" if score >= 7.5 else
        "acceptable" if score >= 4.5 else
        "poor"
    )

    scores = {
        "efficiency": int(round(score)),
        "expressiveness": int(round(min(10, score + 1))),
        "training_stability": int(round(score)),
        "hardware_friendliness": int(round(min(10, score + 1))),
        "overall": int(round(score)),
    }

    return {
        "scores": scores,
        "verdict": verdict,
        "findings": findings,
        "suggestions": suggestions,
        "explanation": "Heuristic evaluation based on structure, metrics, and efficiency trade-offs."
    }

# =========================
# LLM Critic (Groq / OpenAI)
# =========================
def _llm_critic(bp: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key available")

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = CRITIC_PROMPT.format(
            blueprint=json.dumps(bp, indent=2)
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
        )

        content = response.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            raise ValueError("LLM did not return valid JSON")

        return json.loads(match.group(0))
        
    except ImportError:
         raise RuntimeError("groq package not installed")


# =========================
# Public API
# =========================
def critique_blueprint(
    bp: Dict[str, Any],
    dna: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    mode: str = "heuristic",
) -> Dict[str, Any]:

    if mode == "heuristic":
        return _heuristic_critic(bp, dna=dna, metrics=metrics)

    if mode == "llm":
        try:
            return _llm_critic(bp)
        except Exception as e:
            logging.warning(f"[Critic] LLM failed, falling back to heuristic: {e}")
            return _heuristic_critic(bp, dna=dna, metrics=metrics)

    raise ValueError(f"Unknown critic mode: {mode}")