import json
import os
import re
import time
import random
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

from src.generator.heuristic import sample_candidates
from src.generator.self_critique import self_critique_blueprint

# --- Improved Prompt Engineering ---
# explicit schema definition helps the LLM output valid blueprints
BLUEPRINT_SCHEMA_DOC = """
JSON Format Guide:
{
  "name": "string",
  "input_shape": [C, H, W],
  "num_classes": int,
  "backbone": "convnet",
  "stages": [
    { "type": "conv_block"|"bottleneck_block", "filters": int, "depth": int, "kernel": 3|5, "stride": 1|2 }
  ]
}
"""

PROMPT_TEMPLATE = """
You are a Neural Architecture Search expert.
Task: Design {n} diverse neural network architectures for:
- Task: {task}
- Dataset: {dataset}
- Constraints: {constraints}
- Preference: {preference} (e.g. 'balanced', 'high_accuracy', 'low_latency')

{schema}

Rules:
1. Return ONLY a valid JSON list of blueprints. No conversational text.
2. Ensure input_shape matches the dataset (e.g. [3,32,32] for CIFAR, [3,224,224] for ImageNet).
3. Use 'stride': 2 in early stages to downsample and reduce FLOPs.
4. Scale depth and filters according to the 'Preference'.

Output:
"""

def extract_json_from_text(text: str) -> List[Dict]:
    """Robustly extracts JSON list from LLM markdown response."""
    # 1. Try finding markdown code block
    match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # 2. Try finding the first '[' and last ']'
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end+1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: simple cleanup
        text = text.replace("'", '"').replace("True", "true").replace("False", "false")
        return json.loads(text)

def get_smart_seed(dataset: str) -> Dict[str, Any]:
    """Returns a tailored seed based on dataset complexity."""
    if "imagenet" in dataset.lower():
        return {
            "name": "seed_heavy",
            "input_shape": [3, 224, 224],
            "num_classes": 1000,
            "backbone": "convnet",
            "stages": [
                {"type": "conv_block", "filters": 64, "depth": 1, "stride": 2},
                {"type": "bottleneck_block", "filters": 128, "depth": 2, "stride": 2},
                {"type": "bottleneck_block", "filters": 256, "depth": 2, "stride": 2}
            ]
        }
    else: # CIFAR / MNIST
        return {
            "name": "seed_light",
            "input_shape": [3, 32, 32],
            "num_classes": 10,
            "backbone": "convnet",
            "stages": [
                {"type": "conv_block", "filters": 32, "depth": 1, "stride": 1},
                {"type": "conv_block", "filters": 64, "depth": 2, "stride": 2},
                {"type": "conv_block", "filters": 64, "depth": 2, "stride": 1}
            ]
        }

# --- Generation Logic ---

def generate_with_template(
    task: str, 
    dataset: str, 
    constraints: Dict,
    n: int = 4, 
    seed: int = 42, 
    preference: str = "balanced"
) -> List[Dict]:
    
    seed_bp = get_smart_seed(dataset)
    
    # We map "preference" to mutation modes in heuristic.py
    # If constraints include params_max, we pass that context implicitly via mutation mode
    mode = preference
    if constraints.get("params", "").startswith("<"):
        mode = "shrink"
    
    candidates = sample_candidates(seed_bp, n=n, seed=seed, mutation_mode=mode)
    return candidates

def generate_with_llm(prompt: str, n: int = 4, provider: str = "groq") -> List[Dict]:
    """Abstracted LLM caller."""
    try:
        if provider == "groq":
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key: return [] # Fail gracefully
            
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return extract_json_from_text(response.choices[0].message.content)
            
        # Add OpenAI/Anthropic blocks here...
        
    except Exception as e:
        print(f"LLM Generation Failed: {e}")
        return []

def generate(
    task: str,
    dataset: str,
    constraints: Dict,
    n: int = 4,
    mode: str = "template",
    seed: int = 42,
    preference: str = "balanced"
) -> List[Dict]:
    
    if mode == "template":
        return generate_with_template(task, dataset, constraints, n, seed, preference)

    elif mode == "groq":
        constraints_str = ", ".join([f"{k}={v}" for k,v in constraints.items()])
        prompt = PROMPT_TEMPLATE.format(
            n=n,
            task=task,
            dataset=dataset,
            constraints=constraints_str,
            preference=preference,
            schema=BLUEPRINT_SCHEMA_DOC
        )
        # Attempt LLM generation
        blueprints = generate_with_llm(prompt, n=n, provider="groq")
        
        # Fallback to template if LLM fails
        if not blueprints:
            print("LLM failed or returned empty. Falling back to template generation.")
            return generate_with_template(task, dataset, constraints, n, seed, preference)
            
        return blueprints

    else:
        raise ValueError(f"Unknown mode: {mode}")

def generate_and_critique(
    task: str, 
    dataset: str, 
    constraints: Dict, 
    n: int = 4, 
    seed: int = 42, 
    preference: str = "balanced"
) -> List[Dict]:
    """
    Full pipeline: Dream -> Criticize -> Refine
    """
    # 1. Dream
    originals = generate(task, dataset, constraints, n=n, mode="groq", seed=seed, preference=preference)
    
    results = []
    for bp in originals:
        # 2. Criticize & Refine
        try:
            critique = self_critique_blueprint(bp)
            
            # If the improved version is actually better (logic inside self_critique), use it
            final_bp = critique.get("improved_blueprint") or bp
            
            scored = {
                "blueprint": final_bp,
                "original_blueprint": bp,
                "critique_score": critique.get("scores", {}).get("overall", 0),
                "critique_notes": critique.get("analysis", "")
            }
            results.append(scored)
        except Exception as e:
            # Fallback if critique crashes
            results.append({"blueprint": bp, "error": str(e)})

    return results