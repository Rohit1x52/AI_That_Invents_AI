import json
import os
from groq import Groq

CRITIQUE_PROMPT = """
You are an AI Architecture Auditor.

You are given a CNN blueprint in JSON form. Evaluate it in the following dimensions:

1. **Accuracy Potential**  
   How strong is the architecture likely to be? (0-10)

2. **Efficiency (Params/FLOPs)**  
   How lightweight and efficient is it? (0-10)

3. **Latency Expectation**  
   Estimate whether it will run fast on CPU & GPU. (0-10)

4. **Innovation / Novelty**  
   How unique or meaningful are the architectural choices? (0-10)

5. **Design Flaws**  
   Any obvious mistakes? (e.g., bottlenecks too heavy, too many filters, missing BN, etc.)

6. **Overall Score**  
   Weighted score (0-10). Your formula.

Finally, propose **one improved blueprint** in JSON (only edit key parts, not full rewrite).

Return format **EXACTLY**:

{
  "scores": { ... },
  "analysis": "...",
  "improved_blueprint": { ... }
}
"""

def groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=key)

def self_critique_blueprint(bp: dict) -> dict:
    prompt = CRITIQUE_PROMPT + "\nBlueprint:\n" + json.dumps(bp, indent=2)

    client = groq_client()

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )

    text = resp.choices[0].message.content

    import re
    m = re.search(r"(\{.*\})", text, re.S)
    if not m:
        raise RuntimeError("Did not receive JSON critique")

    return json.loads(m.group(1))
