import json
import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast # Key for speedup

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

# Local imports
try:
    from src.codegen.blueprint import Blueprint
    from src.codegen.renderer import render_blueprint
    from src.codegen.validator import validate_blueprint_dict
    from src.eval.metrics import count_parameters
    # We can lazily import these to avoid circular dependency issues
except ImportError:
    pass

# Setup Logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("trainer")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Slower but reproducible

def get_data_loaders(cfg: Dict, input_shape, num_classes):
    """
    Factory for data loaders. 
    Supports 'synthetic' for fast NAS debugging and 'cifar10' for real proxy tasks.
    """
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    
    batch_size = int(train_cfg.get("batch_size", 128))
    use_synthetic = train_cfg.get("use_synthetic", False)
    
    if use_synthetic:
        # Fast, fake data to test if architecture runs without crashing
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, n): self.n = n
            def __len__(self): return self.n
            def __getitem__(self, idx):
                return torch.randn(*input_shape), torch.randint(0, num_classes, (1,)).item()
        
        train_loader = torch.utils.data.DataLoader(SyntheticDataset(2048), batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(SyntheticDataset(512), batch_size=batch_size)
        return train_loader, val_loader

    else:
        # Real Data (CIFAR-10 Example)
        import torchvision
        import torchvision.transforms as T
        
        # Standard augmentation for higher accuracy
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_tfm = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*stats),
        ])
        val_tfm = T.Compose([T.ToTensor(), T.Normalize(*stats)])
        
        root = data_cfg.get("root", "./data")
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tfm)
        valset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=val_tfm)
        
        return (
            torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        )

def train_one_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Setup
    cfg = dict(cfg)
    train_cfg = cfg.get("train", {})
    log_dir = Path(cfg.get("log_dir", "./logs/exp")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    
    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    
    # Save run config
    with open(log_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # 2. Build Model
    bp_dict = cfg["blueprint"]
    try:
        bp = Blueprint.from_dict(bp_dict)
        model = render_blueprint(bp)
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        # Return partial result indicating failure
        return {"status": "FAILED", "error": str(e), "log_dir": str(log_dir)}

    # 3. Data & Optimizer
    train_loader, val_loader = get_data_loaders(cfg, bp.input_shape, bp.num_classes)
    
    lr = float(train_cfg.get("lr", 0.05)) # Higher default for Cosine
    epochs = int(train_cfg.get("epochs", 5))
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision Scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 4. Training Loop
    best_acc = 0.0
    best_ckpt = None
    start_time = time.time()
    
    logger.info(f"Start Training: {epochs} epochs | {device_str} | AMP={device.type=='cuda'}")

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = criterion(logits, y)
                
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
            
            scheduler.step()
            
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    logits = model(x)
                    val_loss += criterion(logits, y).item()
                    _, preds = logits.max(1)
                    correct += preds.eq(y).sum().item()
                    total += y.size(0)
            
            val_acc = correct / total if total > 0 else 0.0
            
            logger.info(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.3f} | Acc={val_acc:.2%}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_ckpt = str(log_dir / "best_model.pth")
                torch.save(model.state_dict(), best_ckpt)
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM Detected! Clearing cache.")
            torch.cuda.empty_cache()
            return {"status": "FAILED", "error": "OOM", "best_val_acc": 0.0}
        else:
            raise e

    total_time = time.time() - start_time
    params = sum(p.numel() for p in model.parameters())
    
    return {
        "status": "COMPLETED",
        "best_val_acc": float(best_acc),
        "best_val_loss": float(val_loss),
        "best_checkpoint": best_ckpt,
        "params": params,
        "train_time_sec": total_time,
        "log_dir": str(log_dir)
    }