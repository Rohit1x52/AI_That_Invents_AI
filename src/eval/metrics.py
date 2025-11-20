import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def accuracy(outputs, targets):
    _, preds = outputs.max(1)
    return (preds == targets).float().mean().item()
