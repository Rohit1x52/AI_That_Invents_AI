import json, os, random
from typing import Dict, List
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

from src.generator.heuristic import sample_candidates
from src.generator.self_critique import self_critique_blueprint

PROMPT_TEMPLATE = """
Task: {task}
Dataset: {dataset}
Constraints: {constraints}
Guidance: prefer {preference} architectures.

Return ONLY a JSON list of blueprints.
Each blueprint must contain:
- name
- input_shape
- num_classes
- backbone
- stages (list of dicts)
"""

def parse_constraints(cstr: Dict) -> str:
    if not cstr:
        return "none"
    return ", ".join([f"{k}={v}" for k,v in cstr.items()])

def generate_with_template(task: str, dataset: str, constraints: Dict,
                           n: int = 4, seed: int = 42, preference: str = "balanced"):

    ds_map = {
        "cifar10": {"input_shape":[3,32,32], "num_classes":10},
        "imagenet": {"input_shape":[3,224,224], "num_classes":1000}
    }
    ds = ds_map.get(dataset, {"input_shape":[3,32,32], "num_classes":10})

    seed_bp = {
        "name": "seed",
        "input_shape": ds["input_shape"],
        "num_classes": ds["num_classes"],
        "backbone": "convnet",
        "stages":[
            {"type":"conv_block", "filters":32, "depth":1, "kernel":3},
            {"type":"conv_block", "filters":64, "depth":1, "kernel":3}
        ]
    }

    candidates = sample_candidates(seed_bp, n=n, seed=seed, mutation_mode=preference)
    return candidates

def generate_with_groq(prompt: str, n: int = 4):
    try:
        from groq import Groq
    except Exception:
        raise RuntimeError(
            "groq package not installed. Install with: pip install groq"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment variables.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200
    )

    text = response.choices[0].message.content

    import re
    m = re.search(r"(\[.*\])", text, re.S)
    if not m:
        raise RuntimeError("LLM did not return a valid JSON list of blueprints.")

    return json.loads(m.group(1))

def generate(task: str,
             dataset: str,
             constraints: Dict,
             n: int = 4,
             mode: str = "template",
             seed: int = 42,
             preference: str = "balanced"):

    prompt = PROMPT_TEMPLATE.format(
        task=task,
        dataset=dataset,
        constraints=parse_constraints(constraints),
        preference=preference
    )

    if mode == "template":
        return generate_with_template(task, dataset, constraints, n=n, seed=seed, preference=preference)

    elif mode == "groq":
        return generate_with_groq(prompt, n=n)

    else:
        raise ValueError(f"Unknown mode: {mode}")

def generate_and_critique(task, dataset, constraints, n=4, seed=42, preference="balanced"):
    """
    Generate blueprints → LLM critique → return:
    - raw blueprints
    - critiques
    - improved versions
    """
    originals = generate(task, dataset, constraints, n=n, mode="groq")

    results = []

    for bp in originals:
        critique = self_critique_blueprint(bp)
        scored = {
            "original": bp,
            "scores": critique.get("scores", {}),
            "analysis": critique.get("analysis", ""),
            "improved_blueprint": critique.get("improved_blueprint", {})
        }
        results.append(scored)

    return results
