import json
import os
import time
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

try:
    import mlflow
    _MLFLOW = True
except Exception:
    _MLFLOW = False
    
from src.codegen.blueprint import Blueprint
from src.codegen.renderer import render_blueprint
from src.codegen.validator import validate_blueprint_dict
from src.eval.metrics import count_parameters
from src.eval.latency import measured_latency_device_list
from src.eval.flops_utils import compute_flops

WORKSPACE_SCREENSHOT = "/mnt/data/744ed4e5-3a58-4991-bb3e-9c16d315b62f.png"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_run_config(log_dir: Path, cfg: Dict[str, Any]):
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

def train_one_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize cfg
    cfg = dict(cfg)  # shallow copy
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log_dir = Path(cfg.get("log_dir", "./logs/exp")).expanduser()
    seed = int(cfg.get("seed", 42))

    # reproducibility
    set_seed(seed)

    # Save config for reproducibility
    save_run_config(log_dir, cfg)

    # MLflow run (optional)
    if _MLFLOW:
        mlflow.start_run()
        mlflow.log_params({
            "device": device_str,
            "seed": seed,
            "batch_size": train_cfg.get("batch_size"),
            "epochs": train_cfg.get("epochs"),
            "lr": train_cfg.get("lr"),
        })

    # Build blueprint & validate
    bp_dict = cfg["blueprint"]
    try:
        validate_blueprint_dict(bp_dict, device="cpu")
    except AssertionError as e:
        raise RuntimeError(f"Blueprint validation failed: {e}")

    bp = Blueprint.from_dict(bp_dict)
    model = render_blueprint(bp)
    model.to(device)

    # Count params
    params = count_parameters(model)

    # Data loaders: synthetic fast default or CIFAR
    use_synth = train_cfg.get("use_synthetic", True)
    batch_size = int(train_cfg.get("batch_size", 128))
    epochs = int(train_cfg.get("epochs", 3))
    patience = int(train_cfg.get("patience", 3))
    lr = float(train_cfg.get("lr", 0.01))

    if use_synth:
        # tiny synthetic dataset
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, n, shape, num_classes):
                self.n = n
                self.shape = shape
                self.num_classes = num_classes
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                x = torch.randn(self.shape)
                y = torch.randint(0, self.num_classes, (1,)).item()
                return x, y

        train_ds = SyntheticDataset(2048, tuple(bp.input_shape), bp.num_classes)
        val_ds = SyntheticDataset(512, tuple(bp.input_shape), bp.num_classes)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        import torchvision
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        data_root = data_cfg.get("root", "./data")
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        valset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimizer / loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training loop with early stopping
    best_val = -1.0
    best_ckpt = None
    counter = 0

    log_dir.mkdir(parents=True, exist_ok=True)

    total_steps = epochs * max(1, len(train_loader))
    step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            step += 1

        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += float(criterion(outputs, labels).item())
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
        val_acc = (correct / total) if total > 0 else 0.0

        # Print progress
        avg_train_loss = running_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_acc: {val_acc:.4f}")

        # Log
        if _MLFLOW:
            mlflow.log_metric("val_acc", float(val_acc), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)

        # Checkpoint if improved
        if val_acc > best_val:
            best_val = val_acc
            best_ckpt = str(log_dir / "best.pth")
            try:
                torch.save(model.state_dict(), best_ckpt)
            except Exception as e:
                # Do not fail training if save fails; warn instead
                print("Warning: failed to save checkpoint:", e)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # early stop
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # After training: measure latency & FLOPs (best-effort)
    try:
        model_cpu = render_blueprint(bp)  # fresh model for CPU measurement to avoid device sync issues
    except Exception:
        model_cpu = model

    latency_results = None
    try:
        latency_results = measured_latency_device_list(model_cpu, bp.input_shape, devices=("cpu", "cuda"))
    except Exception:
        latency_results = {"cpu": None, "cuda": None}

    flops = None
    try:
        flops = compute_flops(model_cpu, tuple(bp.input_shape))
    except Exception:
        flops = None

    # Log final metrics
    result = {
        "best_val_acc": float(best_val),
        "best_checkpoint": best_ckpt,
        "params": int(params),
        "latency_ms": latency_results,
        "flops": int(flops) if flops is not None else None,
        "log_dir": str(log_dir),
        "seed": seed
    }

    if _MLFLOW:
        mlflow.log_metrics({"best_val_acc": float(best_val)})
        # Optionally log artifacts
        try:
            if best_ckpt:
                mlflow.log_artifact(best_ckpt)
        except Exception:
            pass
        mlflow.end_run()

    return result


# CLI convenience
def main():
    import argparse
    parser = argparse.ArgumentParser(prog="train_one_model")
    parser.add_argument("--blueprint", type=str, default="examples/blueprints/blueprint_convnet.json")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="./logs/exp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bp = json.load(open(args.blueprint))
    cfg = {
        "blueprint": bp,
        "train": {"use_synthetic": args.use_synthetic, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        "device": args.device,
        "log_dir": args.log_dir,
        "seed": args.seed
    }
    res = train_one_model(cfg)
    print("Run result:", res)

if __name__ == "__main__":
    main()
