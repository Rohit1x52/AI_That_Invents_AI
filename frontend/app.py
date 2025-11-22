import streamlit as st
import json
from pathlib import Path
import sys
import os
import torch

# Ensure src is importable
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules (Phase-1)
from src.codegen.blueprint import Blueprint
from src.codegen.renderer import render_blueprint
from src.codegen.validator import validate_blueprint_dict
from src.eval.metrics import count_parameters
from src.eval.latency import measured_latency_device_list
from src.eval.flops_utils import compute_flops

# Paths
EXAMPLES_DIR = Path("examples/blueprints")
LOG_DIR = Path("logs/frontend")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Small helpers
def list_blueprints():
    return sorted([p for p in EXAMPLES_DIR.glob("*.json")])

def load_blueprint(path: Path):
    return json.loads(path.read_text())

st.set_page_config(page_title="AI-Inventor Phase1 Demo", layout="wide")

# Header
st.title("🧬 AI That Invents AI — Phase 1 MVP Demo")
st.markdown("Interactive demo: blueprint → model → validate → train → evaluate")

# Sidebar controls
with st.sidebar:
    st.header(" Controls")
    
    # Blueprint selection
    bpaths = list_blueprints()
    if not bpaths:
        st.error("No blueprints found in examples/blueprints. Add JSON files and reload.")
        st.stop()
    
    selection = st.selectbox(" Choose blueprint", bpaths, format_func=lambda p: p.name)
    
    st.divider()
    st.subheader(" Configuration")
    use_synth = st.checkbox("Use synthetic data (fast)", value=True)
    epochs = st.number_input("Epochs", min_value=1, max_value=10, value=2)
    batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64, step=8)
    
    device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    device = st.selectbox("Device", options=device_options, index=1 if torch.cuda.is_available() else 0)
    
    st.divider()
    st.subheader(" Actions")
    run_validate = st.button(" Validate Blueprint", use_container_width=True)
    run_render = st.button(" Render Model", use_container_width=True)
    run_train = st.button(" Quick Train", use_container_width=True)
    run_latency = st.button("⏱ Measure Latency", use_container_width=True)
    run_flops = st.button(" Compute FLOPs", use_container_width=True)

# Load and display blueprint JSON
try:
    bp_json = load_blueprint(selection)
    st.subheader(" Blueprint JSON")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.json(bp_json)
    with col2:
        st.metric("Input Shape", str(bp_json.get("input_shape", "N/A")))
        st.metric("Classes", bp_json.get("num_classes", "N/A"))
        st.metric("Layers", len(bp_json.get("layers", [])))
except Exception as e:
    st.error(f"Failed to load blueprint: {e}")
    st.stop()

# Validation
if run_validate:
    with st.spinner("Validating blueprint..."):
        try:
            meta = validate_blueprint_dict(bp_json, device="cpu")
            st.success(" Validation passed!")
            if meta:
                st.json(meta)
        except AssertionError as e:
            st.error(f" Validation failed: {e}")
        except Exception as e:
            st.error(f" Unexpected error during validation: {e}")

# Render
if run_render:
    with st.spinner("Rendering model..."):
        try:
            bp = Blueprint.from_dict(bp_json)
            model = render_blueprint(bp)
            
            st.subheader(" Model Architecture")
            st.code(str(model), language="python")
            
            param_count = int(count_parameters(model))
            st.metric("Total Parameters", f"{param_count:,}")
            st.info(f"Model size: ~{param_count * 4 / (1024**2):.2f} MB (float32)")
        except Exception as e:
            st.error(f" Render failed: {e}")
            st.exception(e)

# Quick train
if run_train:
    st.subheader(" Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Training in progress..."):
        try:
            bp = Blueprint.from_dict(bp_json)
            model = render_blueprint(bp)
            model.to(device)
            # Lightweight synthetic training if requested
            if use_synth:
                # tiny synthetic dataset
                class Synth:
                    def __init__(self, n=1024, shape=tuple(bp.input_shape), num_classes=bp.num_classes):
                        self.n = n; self.shape = shape; self.num_classes = num_classes
                    def __len__(self): return self.n
                    def __getitem__(self, idx):
                        import torch
                        return torch.randn(self.shape), torch.randint(0, self.num_classes, (1,)).item()
                from torch.utils.data import DataLoader
                train_loader = DataLoader(Synth(1024, tuple(bp.input_shape), bp.num_classes),
                                          batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(Synth(256, tuple(bp.input_shape), bp.num_classes),
                                        batch_size=batch_size, shuffle=False)
            else:
                import torchvision.transforms as T
                import torchvision
                transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
                trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
                valset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
                from torch.utils.data import DataLoader
                train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

            # Training loop (very small)
            import torch.nn as nn, torch.optim as optim
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            best_val = 0.0
            metrics_placeholder = st.empty()
            
            for epoch in range(epochs):
                # Update progress
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                
                model.train()
                total_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                
                # Validation
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
                        correct += preds.eq(labels).sum().item()
                        total += labels.size(0)
                
                val_acc = correct / total if total > 0 else 0.0
                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # Display metrics
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Train Loss", f"{avg_train_loss:.4f}")
                    col2.metric("Val Loss", f"{avg_val_loss:.4f}")
                    col3.metric("Val Accuracy", f"{val_acc:.2%}")
                
                if val_acc > best_val:
                    best_val = val_acc
                    ckpt_path = LOG_DIR / f"{selection.stem}_best.pth"
                    torch.save(model.state_dict(), ckpt_path)
            progress_bar.progress(1.0)
            status_text.text("Training complete!")
            st.success(f" Training finished! Best validation accuracy: {best_val:.2%}")
            st.info(f" Checkpoint saved to: `{ckpt_path.name}`")
            st.balloons()
        except Exception as e:
            st.error(f" Training failed: {e}")
            st.exception(e)

# Latency
if run_latency:
    with st.spinner("Measuring latency (performing repeated inference calls)..."):
        try:
            bp = Blueprint.from_dict(bp_json)
            model = render_blueprint(bp)
            results = measured_latency_device_list(model, bp.input_shape, devices=("cpu", "cuda"))
            
            st.subheader("⏱ Latency Results")
            if isinstance(results, dict):
                for device_name, latency in results.items():
                    if latency is None:
                        st.metric(f"{device_name.upper()} Latency", "N/A")
                    else:
                        st.metric(f"{device_name.upper()} Latency", f"{latency:.2f} ms")
            else:
                st.json(results)
        except Exception as e:
            st.error(f" Latency measurement failed: {e}")
            st.exception(e)

# FLOPs
if run_flops:
    with st.spinner("Computing FLOPs (requires fvcore)..."):
        try:
            bp = Blueprint.from_dict(bp_json)
            model = render_blueprint(bp)
            flops = compute_flops(model, tuple(bp.input_shape))
            
            st.subheader("FLOPs Analysis")
            if flops:
                st.metric("Total FLOPs", f"{flops:,}")
                st.info(f"Approximately {flops / 1e9:.2f} GFLOPs")
            else:
                st.warning("FLOPs computation returned None. Make sure fvcore is installed.")
        except Exception as e:
            st.error(f" FLOPs computation failed: {e}")
            st.exception(e)

st.divider()
st.caption("🧬 AI That Invents AI - Phase 1 MVP Demo | Built with Streamlit")
st.caption("Use this interface for quick interactive exploration of neural architecture blueprints.")
st.caption("Made with ❤️ by the Rohit Ranjan Kumar")
