<div align="center">

# 🧬 AI That Invents AI

### Self-Evolving Neural Architecture System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Automated neural architecture synthesis through evolutionary design**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Examples](#-examples)

</div>

---

## 📋 Overview

**AI That Invents AI** is a self-evolving neural architecture system that automatically designs, validates, and optimizes deep learning models. Phase 1 establishes the foundation with a complete pipeline from blueprint specification to trained model evaluation.

### 🎯 Pipeline Flow

```
Blueprint (JSON/DSL) → Validation → PyTorch Model → Training → Evaluation → Metrics Report
```

---

## ✨ Features

### 🏗️ **1. Architecture DSL & Blueprint System**
- 📝 JSON-based blueprint format
- 🔧 Configurable stages (filters, depth, kernels)
- 🎯 Model head definition
- 📐 Input/output shape specification

### ⚙️ **2. Code Generation**
- 🔄 Blueprint → `torch.nn.Module` conversion
- 📦 Modular architecture under `src/codegen/`
- 🎨 ConvNet-style backbones (MVP)
- 🔌 Extensible for new architectures

### ✅ **3. Validation Pass**
- 🔍 Forward pass shape verification
- 📊 Parameter count validation
- 🛡️ Tensor dimension checks
- ✔️ Structural correctness enforcement

### 🚀 **4. Training Loop**
- ⚡ Fast 2–5 epoch runs
- 🖼️ CIFAR-10 and synthetic dataset support
- 🎓 SGD + CrossEntropy optimization
- 🛑 Early stopping mechanism

### 📊 **5. Evaluation Tools**

| Tool | Description |
|------|-------------|
| `metrics.py` | Parameter counting & accuracy helpers |
| `latency.py` | CPU & GPU measured latency |
| `flops_utils.py` | FLOPs estimation (via fvcore) |

### 📦 **6. Example Blueprints**
- `blueprint_convnet.json` - Standard ConvNet
- `blueprint_wideconv.json` - Wide architecture
- `blueprint_mixed.json` - Hybrid design

---

## 📁 Project Structure

```
AI_That_Invents_AI/
│
├── 📂 src/
│   ├── 🔧 codegen/          # Model generation engine
│   │   ├── blueprint.py     # Blueprint data structures
│   │   ├── renderer.py      # PyTorch model renderer
│   │   ├── validator.py     # Architecture validator
│   │   └── __init__.py
│   │
│   ├── 📝 dsl/              # Domain-specific language
│   │   ├── parser.py        # DSL parser
│   │   └── __init__.py
│   │
│   ├── 📊 eval/             # Evaluation utilities
│   │   ├── latency.py       # Latency measurement
│   │   ├── metrics.py       # Performance metrics
│   │   ├── flops_utils.py   # FLOPs computation
│   │   └── __init__.py
│   │
│   ├── 🎓 trainer/          # Training pipeline
│   │   ├── train.py         # Training logic
│   │   └── __init__.py
│   │
│   ├── 📋 spec/             # Specification schema
│   │   ├── spec_schema.json # JSON schema definitions
│   │   ├── parser.py        # Spec parser
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── 📚 examples/
│   └── blueprints/          # Example architectures
│       ├── blueprint_convnet.json
│       ├── blueprint_wideconv.json
│       └── blueprint_mixed.json
│
├── 🖥️ frontend/            # Web interface
│   └── app.py              # Streamlit application
│
├── 🧪 tests/               # Unit & integration tests
│   ├── unit/
│   └── conftest.py
│
├── 📓 notebooks/           # Jupyter notebooks
│   └── train_blueprint_mvp.ipynb
│
├── 🚀 run_example.py       # Quick start training script
├── 📖 README.md
├── 📦 requirements.txt
├── 🚫 .gitignore
└── 📄 .gitattributes
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Rohit1x52/AI_That_Invents_AI.git
cd AI_That_Invents_AI
```

2. **Create virtual environment**
```bash
python -m venv AIinventor
# Windows
.\AIinventor\Scripts\Activate.ps1
# Linux/Mac
source AIinventor/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎯 Quick Start

### Option 1: Run Training Script
```bash
# Windows
.\AIinventor\Scripts\python.exe run_example.py

# Linux/Mac
python run_example.py
```

This will:
- ✅ Load a blueprint from `examples/blueprints/`
- ✅ Validate the architecture
- ✅ Render the PyTorch model
- ✅ Train for 3 epochs on synthetic data
- ✅ Display metrics and save checkpoints

### Option 2: Interactive Web Interface
```bash
streamlit run frontend/app.py
```

Features:
- 🖱️ Interactive blueprint selection
- ⚙️ Configurable training parameters
- 📊 Real-time training progress
- 📈 Latency and FLOPs measurement
- 💾 Model checkpoint management

### Option 3: Using Jupyter Notebook
```bash
jupyter notebook notebooks/train_blueprint_mvp.ipynb
```

### Command Line Usage
```python
from src.codegen.blueprint import Blueprint
from src.codegen.renderer import render_blueprint
import json

# Load blueprint
with open("examples/blueprints/blueprint_convnet.json") as f:
    bp_dict = json.load(f)

# Create model
blueprint = Blueprint.from_dict(bp_dict)
model = render_blueprint(blueprint)

# Evaluate
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🏗️ Architecture

### Core Components

#### 🎨 **Blueprint System**
Defines neural architecture in declarative JSON format:
```json
{
  "input_shape": [3, 32, 32],
  "num_classes": 10,
  "layers": [
    {"type": "conv", "filters": 64, "kernel": 3},
    {"type": "pool", "size": 2}
  ]
}
```

#### 🔧 **Renderer**
Converts blueprints to executable PyTorch models with automatic shape inference and layer composition.

#### ✅ **Validator**
Ensures architectural integrity through forward pass simulation and constraint checking.

#### 📊 **Evaluator**
Measures model quality across multiple dimensions:
- **Accuracy**: Classification performance
- **Latency**: Inference speed (CPU/GPU)
- **FLOPs**: Computational complexity
- **Parameters**: Model size

---

## 📊 Examples

### Train a Model
See `notebooks/train_blueprint_mvp.ipynb` for complete example with:
- ✅ Blueprint loading
- ✅ Model rendering
- ✅ CIFAR-10 training
- ✅ Metrics logging with MLflow

---

## 🛣️ Roadmap

### Phase 1: MVP ✅ (Current)
- Blueprint → Model pipeline
- Basic training & evaluation
- Example architectures

### Phase 2: Evolution Engine 🔄 (Next)
- Genetic algorithm for architecture search
- Multi-objective optimization
- Population management

### Phase 3: Advanced Features 📈 (Future)
- Distributed training support
- Neural architecture search (NAS)
- Hardware-aware optimization
- Automated hyperparameter tuning

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🙏 Acknowledgments

- Built with PyTorch
- Inspired by neural architecture search research
- FLOPs computation via fvcore
- MLflow for experiment tracking
- Streamlit for web interface

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by Rohit Ranjan Kumar

</div>
