import json, argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.generator.llm_generator import generate_and_critique

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="image classification")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--params_max", type=int, default=500000)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    constraints = {"params_max": args.params_max}

    results = generate_and_critique(
        task=args.task,
        dataset=args.dataset,
        constraints=constraints,
        n=args.n
    )

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
