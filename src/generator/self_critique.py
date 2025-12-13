import json
import os
import re
import logging
from typing import Dict, Any, Optional

try:
    from groq import Groq
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False

CRITIQUE_PROMPT = """
You are an expert Neural Architecture Search (NAS) Auditor.

Your Goal: Analyze the provided CNN Blueprint and produce an IMPROVED version.

Evaluation Criteria:
1. Accuracy Potential (0-10)
2. Efficiency (Params/FLOPs) (0-10)
3. Latency Expectation (0-10)
4. Innovation (0-10)
5. Flaws (List specific bottlenecks or design errors)

Output Format:
You must return ONLY valid JSON. Do not include preamble text.
{
  "scores": { "accuracy": 8, "efficiency": 7, "latency": 9, "innovation": 5, "overall": 7.5 },
  "analysis": "Brief analysis of flaws...",
  "improved_blueprint": { ... full valid blueprint json ... }
}
"""

_GROQ_CLIENT = None

def get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        key = os.environ.get("GROQ_API_KEY")
        if key and _HAS_GROQ:
            _GROQ_CLIENT = Groq(api_key=key)
    return _GROQ_CLIENT

def extract_json_content(text: str) -> str:
    # 1. Try Markdown Code Block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    
    # 2. Try greedy brace matching
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
        
    return text

def self_critique_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    client = get_groq_client()
    
    # Fallback if no API key or client
    if not client:
        logging.warning("Groq client unavailable. Returning dummy critique.")
        return {
            "scores": {"overall": 5.0},
            "analysis": "No LLM critique available (Missing API Key).",
            "improved_blueprint": bp 
        }

    full_prompt = f"{CRITIQUE_PROMPT}\n\nCURRENT BLUEPRINT:\n{json.dumps(bp, indent=2)}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a JSON-speaking AI architect."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2, # Low temp for valid JSON
            max_tokens=2000,
            response_format={"type": "json_object"} # Force JSON mode if supported
        )
        
        raw_text = response.choices[0].message.content
        cleaned_json = extract_json_content(raw_text)
        
        data = json.loads(cleaned_json)
        
        # Validate structure
        if "improved_blueprint" not in data:
            data["improved_blueprint"] = bp 
            
        return data

    except Exception as e:
        logging.error(f"Critique generation failed: {e}")
        return {
            "scores": {"overall": 0.0},
            "analysis": f"Error during critique: {str(e)}",
            "improved_blueprint": bp
        }