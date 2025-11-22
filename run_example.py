"""Example script to demonstrate training a model from blueprint"""
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.trainer.train import train_one_model

def main():
    print("=" * 60)
    print("AI That Invents AI - Training Example")
    print("=" * 60)
    
    # Check blueprint file
    bp_path = Path("examples/blueprints/blueprint_convnet.json")
    if not bp_path.exists():
        print(f"❌ ERROR: Blueprint not found: {bp_path}")
        print(f"   Current directory: {Path.cwd()}")
        return 1
    
    print(f"✓ Found blueprint: {bp_path}")
    
    # Load blueprint
    try:
        bp = json.loads(bp_path.read_text())
        print(f"✓ Blueprint loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load blueprint: {e}")
        return 1
    
    # Create config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")
    
    cfg = OmegaConf.create({
        "blueprint": bp,
        "data": {"root": "./data"},
        "train": {
            "use_synthetic": True, 
            "batch_size": 128, 
            "epochs": 3, 
            "lr": 0.01, 
            "patience": 2
        },
        "device": device,
        "log_dir": "./logs/convnet_exp",
        "seed": 42
    })
    
    print("\n" + "-" * 60)
    print("Configuration:")
    print("-" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("-" * 60)
    
    # Train model
    print("\n🚀 Starting training...")
    try:
        res = train_one_model(dict(cfg))
        
        print("\n" + "=" * 60)
        print("✅ Training finished successfully!")
        print("=" * 60)
        print("\nResults:")
        print(json.dumps(res, indent=2))
        print("\n" + "=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
