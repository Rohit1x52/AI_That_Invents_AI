import argparse, json, torch, numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.metrics import confusion_matrix, classification_report
from src.codegen.blueprint import Blueprint
from src.codegen.renderer import render_blueprint
import torchvision.transforms as T
import torchvision

def load_checkpoint(ckpt_path, bp):
    model = render_blueprint(Blueprint.from_dict(bp))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def cifar10_test_loader(root="./data", batch_size=128):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader

def evaluate(model, loader, device="cpu"):
    model.to(device)
    ys, preds = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            _, p = out.max(1)
            ys.extend(y.numpy().tolist())
            preds.extend(p.cpu().numpy().tolist())
    return ys, preds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--blueprint", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    bp = json.load(open(args.blueprint))
    model = load_checkpoint(args.ckpt, bp)
    loader = cifar10_test_loader()
    ys, preds = evaluate(model, loader, device=args.device)
    print(classification_report(ys, preds))
    print("Confusion matrix:")
    print(confusion_matrix(ys, preds))

if __name__ == "__main__":
    main()
