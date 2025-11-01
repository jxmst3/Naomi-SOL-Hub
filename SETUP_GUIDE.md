# Setup Guide: AI 3D Design & Printing Agent

Complete step-by-step installation and configuration for the design agent on Linux, macOS, and Windows.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start (5 min)](#quick-start-5-min)
3. [Detailed Installation](#detailed-installation)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum
- **Python**: 3.10 or newer
- **RAM**: 8GB (16GB recommended)
- **Disk Space**: 20GB (50GB+ recommended)
- **OS**: Linux, macOS, or Windows with WSL2

### GPU Acceleration (Optional)
- **NVIDIA GPU**: RTX 2060 or better with 6GB+ VRAM
- **CUDA**: 11.8 or 12.1
- **cuDNN**: 8.x

### External Tools
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install git ffmpeg slic3r python3-dev python3-venv

# macOS
brew install git ffmpeg slic3r

# Windows
# Download and install:
# - Git: https://git-scm.com/download/win
# - FFmpeg: https://ffmpeg.org/download.html
# - slic3r: https://slic3r.org/download
# - Python 3.11+: https://www.python.org/downloads/
```

---

## Quick Start (5 min)

```bash
# 1. Clone or download the package
git clone https://github.com/yourusername/design-agent-local.git
cd design-agent-local

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Test GUI
python3 design_agent_local.py --open_gui

# 5. Try first generation
python3 design_agent_local.py \
  --backend shap_e \
  --prompt "a small robot"
```

---

## Detailed Installation

### 1. Prerequisites

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
  python3.11 python3.11-venv python3.11-dev \
  git wget curl \
  build-essential cmake \
  libopenblas-dev liblapack-dev \
  ffmpeg \
  slic3r \
  libgl1-mesa-glx libglib2.0-0
```

#### macOS
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 git cmake ffmpeg slic3r openblas lapack
```

#### Windows (with WSL2)
```powershell
# Enable WSL2
wsl --install

# In WSL terminal, follow Ubuntu instructions
```

### 2. Python Virtual Environment

```bash
# Create environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Verify activation (should show "venv" in terminal)
which python3  # or 'where python' on Windows
```

### 3. Install Python Packages

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# If GPU acceleration desired (NVIDIA only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### 4. API Configuration

The agent requires API keys for cloud-based backends. Add to environment:

```bash
# OpenAI (for Shap-E)
export OPENAI_API_KEY="sk-..."

# Stability AI (for TripoSR)
export STABILITY_API_KEY="sk-..."

# OctoPrint (optional, for printing)
export OCTOPI_URL="http://192.168.1.100"
export OCTOPI_API_KEY="your_octoprint_api_key"
```

**Persist environment variables:**

Create `.env` file:
```bash
OPENAI_API_KEY=sk-...
STABILITY_API_KEY=sk-...
OCTOPI_URL=http://192.168.1.100
OCTOPI_API_KEY=...
```

Load before running:
```bash
set -a
source .env
set +a
python3 design_agent_local.py --open_gui
```

Or on Windows:
```cmd
for /f "tokens=*" %i in (type .env) do set %i
python design_agent_local.py --open_gui
```

### 5. External Tool Installation

#### slic3r (3D Printer Slicing)
```bash
# Ubuntu/Debian
sudo apt-get install slic3r

# macOS
brew install slic3r

# Verify
slic3r --version
```

#### COLMAP (Multi-view 3D Reconstruction)
```bash
# Ubuntu/Debian
sudo apt-get install colmap

# macOS
brew install colmap

# Verify
colmap --version
```

#### FFmpeg (Video/Audio Processing)
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify
ffmpeg -version
```

---

## Configuration

### 1. Print Settings

Edit generation script or create config file `config.json`:

```json
{
  "layer_height": 0.2,
  "nozzle_diameter": 0.4,
  "print_temperature": 205,
  "bed_temperature": 60,
  "infill_density": 20,
  "print_speed": 60
}
```

Load in code:
```python
import json
with open('config.json') as f:
    config_dict = json.load(f)
config = Config(**config_dict)
```

### 2. OctoPrint Integration

Configure OctoPrint URL and API key:

```bash
export OCTOPI_URL="http://octopi.local"  # or IP address
export OCTOPI_API_KEY="your_api_key"
```

Find API key in OctoPrint web UI: Settings > API > API Key

### 3. GPU/CUDA Configuration

```bash
# Check NVIDIA GPU
nvidia-smi

# Specify GPU
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Disable GPU (force CPU)
export CUDA_VISIBLE_DEVICES=""
```

### 4. Model Cache Directories

Models are cached to avoid re-downloading:

```bash
# Set cache locations
export HF_HOME="~/.cache/huggingface"
export TORCH_HOME="~/.cache/torch"

# Clear caches if needed
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
```

---

## Verification

### 1. System Check

```bash
python3 design_agent_local.py --check-system
```

Should output:
- Python version: 3.10+
- PyTorch version with CUDA support
- Git, FFmpeg, slic3r availability
- Disk space available

### 2. Test Each Backend

```bash
# Text-to-3D (Shap-E)
python3 design_agent_local.py \
  --backend shap_e \
  --prompt "cube"

# Image-to-3D (TripoSR)
python3 design_agent_local.py \
  --backend tripo \
  --image sample.png

# Concept Images (Stable Diffusion)
python3 design_agent_local.py \
  --backend diffusion \
  --prompt "futuristic chair"
```

### 3. Test GUI

```bash
python3 design_agent_local.py --open_gui
```

Should launch Tkinter window with:
- Backend selector dropdown
- Prompt input area
- Image file browser
- Generate button
- Status output

### 4. Run Unit Tests

```bash
pytest test_design_agent_local.py -v
pytest test_design_agent_local.py --cov=design_agent_local
```

Expected output: All tests pass ✓

### 5. Validate Slicing

```bash
python3 design_agent_local.py \
  --backend shap_e \
  --prompt "simple box" \
  --slice \
  --layer_height 0.2
```

Check output/*/model.gcode is valid (readable text file with G-code commands)

---

## Troubleshooting

### Installation Issues

#### "ModuleNotFoundError: No module named 'X'"
```bash
# Ensure virtualenv is activated
source venv/bin/activate

# Reinstall requirements
pip install --upgrade -r requirements.txt

# Try individual package
pip install torch --upgrade
```

#### "pip: command not found"
```bash
# Use python -m pip
python3 -m pip install -r requirements.txt
```

#### Incompatible CUDA/PyTorch
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch for specific CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size or inference steps
python3 design_agent_local.py \
  --backend diffusion \
  --prompt "test" \
  --batch_size 1

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Check memory usage
nvidia-smi  # GPU memory
free -h     # System RAM
```

#### Slow Generation
```bash
# Enable FP16 (faster, less precise)
export TORCH_DTYPE=float16

# Check GPU utilization
nvidia-smi -l 1  # Refresh every 1s

# Reduce model precision
# Edit config: "precision": "fp16"
```

#### "slic3r not found"
```bash
# Verify installation
which slic3r
slic3r --version

# If missing, install:
sudo apt-get install slic3r  # Ubuntu/Debian
brew install slic3r          # macOS

# Or download: https://slic3r.org/download
```

#### API Connection Failed
```bash
# Check internet connection
ping api.openai.com

# Verify API keys set
echo $OPENAI_API_KEY
echo $STABILITY_API_KEY

# Test API manually
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check firewall/proxy settings
```

#### GUI Won't Launch
```bash
# Check tkinter installation
python3 -m tkinter  # Should open test window

# Install tkinter if missing
sudo apt-get install python3-tk      # Ubuntu/Debian
brew install python-tk@3.11          # macOS

# Try CLI instead
python3 design_agent_local.py --prompt "test" --backend diffusion
```

### Debugging

#### Enable Verbose Logging
```bash
python3 design_agent_local.py \
  --verbose \
  --backend diffusion \
  --prompt "test"
```

Logs written to `design_agent.log`

#### Profile Performance
```bash
python3 -m cProfile -s cumulative design_agent_local.py \
  --backend diffusion \
  --prompt "test" > profile.txt
```

Review profile.txt for bottlenecks

#### Test Individual Components
```python
# Python interactive mode
python3

from design_agent_local import *

# Test torch
import torch
print(torch.cuda.is_available())

# Test diffusers
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Test mesh processing
from design_agent_local import MeshProcessor
processor = MeshProcessor(Config())
```

---

## Platform-Specific Notes

### Linux
- Most straightforward installation
- Full GPU/CUDA support
- systemd integration (see launch_agent.sh)

### macOS
- No NVIDIA GPU support (Metal acceleration only)
- Brew package manager highly recommended
- M1/M2 ARM support: Install arm64 PyTorch build

### Windows
- WSL2 strongly recommended
- Native Windows requires manual tool installation
- GPU support through WSL2 NVIDIA integration

### Docker/Container

Build Docker image:
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv git ffmpeg slic3r

WORKDIR /app
COPY . .

RUN python3.11 -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

CMD ["python3", "design_agent_local.py", "--open_gui"]
```

Run:
```bash
docker build -t design-agent .
docker run --gpus all -it design-agent
```

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Configure API keys
3. ✅ Run GUI or CLI
4. 📖 Read README.md for features
5. 🧪 Try examples in EXAMPLES.md
6. 🚀 Deploy to production

---

**Version**: 1.0  
**Last Updated**: 2025  
**Support**: See README.md troubleshooting section
