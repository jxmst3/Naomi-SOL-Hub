# Usage Examples: AI 3D Design & Printing Agent

Complete examples and tutorials for all major use cases.

---

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Backend Comparisons](#backend-comparisons)
3. [Advanced Workflows](#advanced-workflows)
4. [Integration Examples](#integration-examples)
5. [Troubleshooting Examples](#troubleshooting-examples)

---

## Basic Examples

### Example 1: Simple Text-to-3D

Generate a 3D model from text description:

```bash
python3 design_agent_local.py \
  --prompt "a small decorative vase" \
  --backend shap_e
```

**Output**: `output/shap_e/model_shap_e.glb`

### Example 2: Image-to-3D

Generate 3D model from a photograph:

```bash
python3 design_agent_local.py \
  --image product_photo.jpg \
  --backend tripo
```

**Output**: `output/tripo/model_tripo.glb`

### Example 3: Generate and Slice for Printing

Create design and generate G-code for 3D printer:

```bash
python3 design_agent_local.py \
  --prompt "miniature robot" \
  --backend shap_e \
  --slice \
  --layer_height 0.15 \
  --output ./prints
```

**Output**: 
- Refined mesh: `prints/shap_e/model_refined.stl`
- G-code: `prints/shap_e/model.gcode`
- Metadata: `prints/shap_e/metadata.json`

### Example 4: Multi-view 3D Reconstruction

Generate 3D model from multiple photographs:

```bash
python3 design_agent_local.py \
  --image_dir ./product_photos \
  --backend gaussian \
  --output ./models
```

**Requirements**: 15-50 images of object from different angles

**Output**: 
- PLY mesh: `models/gaussian/model.ply`
- COLMAP reconstruction: `work/gaussian/colmap/`
- Training logs: `work/gaussian/training/`

---

## Backend Comparisons

### Use Case 1: Quick Prototyping (Fastest)

**Task**: Rapidly generate concepts for brainstorming

```bash
# Use Stable Diffusion for concept images (fastest)
python3 design_agent_local.py \
  --prompt "futuristic chair design" \
  --backend diffusion

# Then use TripoSR for best single-image quality
python3 design_agent_local.py \
  --image output/diffusion/concept_0.png \
  --backend tripo
```

**Time**: ~30 seconds total  
**Cost**: $0.01  
**Quality**: ⭐⭐⭐⭐

### Use Case 2: Best Quality with Time Budget (1 hour)

**Task**: Generate highest-quality model for final product

```bash
# Start DreamFusion training
python3 design_agent_local.py \
  --prompt "luxury pen" \
  --backend dreamfusion \
  --output ./final_product
```

**Time**: 45-90 minutes  
**Cost**: $0  
**Quality**: ⭐⭐⭐⭐⭐

### Use Case 3: Photogrammetric Accuracy (Multi-view)

**Task**: Scan real object with high accuracy

```bash
# 1. Take 30-40 photos of object from all angles
# 2. Run reconstruction
python3 design_agent_local.py \
  --image_dir ./object_photos \
  --backend gaussian
```

**Time**: 10-15 minutes  
**Cost**: $0  
**Quality**: ⭐⭐⭐⭐⭐ (photogrammetric)  
**Accuracy**: ±0.1-0.5mm

### Use Case 4: Production Pipeline (Batch)

**Task**: Generate multiple models for inventory

```bash
#!/bin/bash

PROMPTS=(
  "coffee mug"
  "water bottle"
  "desk organizer"
  "phone stand"
)

for prompt in "${PROMPTS[@]}"; do
  python3 design_agent_local.py \
    --prompt "$prompt" \
    --backend shap_e \
    --slice \
    --output "./batch_output"
  
  echo "Generated: $prompt"
done
```

---

## Advanced Workflows

### Workflow 1: Iterative Design Refinement

Starting with text → refining through iteration:

```bash
# Step 1: Generate initial concept
python3 design_agent_local.py \
  --prompt "gaming headset" \
  --backend shap_e \
  --output ./iteration_1

# Step 2: Export and review the model
# [Manual review in Fusion 360 / Blender]

# Step 3: Refine prompt based on results
python3 design_agent_local.py \
  --prompt "gaming headset with large ear cups and RGB lights" \
  --backend shap_e \
  --output ./iteration_2

# Step 4: Compare results, pick best
# [Export comparison images]
```

### Workflow 2: Hybrid Generation (Text + Image)

Combine text description with reference image:

```bash
# 1. Generate concept images first
python3 design_agent_local.py \
  --prompt "luxury smartwatch with round face" \
  --backend diffusion \
  --output ./concept

# 2. Use best concept as reference for 3D
python3 design_agent_local.py \
  --image output/diffusion/concept_2.png \
  --backend tripo \
  --output ./final_model
```

### Workflow 3: Direct to Printer Pipeline

Complete workflow from text to print queue:

```bash
#!/bin/bash

DESIGN="cute desk organizer"
PRINTER_URL="http://octopi.local"
PRINTER_KEY="your_api_key"

echo "Step 1: Generating design..."
python3 design_agent_local.py \
  --prompt "$DESIGN" \
  --backend shap_e \
  --slice \
  --layer_height 0.2 \
  --octopi_url "$PRINTER_URL" \
  --octopi_key "$PRINTER_KEY" \
  --output ./ready_to_print

echo "Step 2: Design queued for printing!"

# Check printer status
curl -H "X-API-Key: $PRINTER_KEY" \
  "$PRINTER_URL/api/job"
```

### Workflow 4: Batch Processing with Configuration

Process multiple designs with saved config:

```bash
# Create config file
cat > batch_config.json << 'EOF'
{
  "backend": "shap_e",
  "output_dir": "batch_output",
  "slice": true,
  "layer_height": 0.2,
  "narrate": true
}
EOF

# Process designs from text file
while IFS= read -r prompt; do
  echo "Processing: $prompt"
  python3 design_agent_local.py \
    --config batch_config.json \
    --prompt "$prompt"
done < designs.txt
```

---

## Integration Examples

### Integration 1: Cura Slicer Integration

Use generated STL with Cura:

```bash
# 1. Generate model
python3 design_agent_local.py \
  --prompt "custom lithophane holder" \
  --backend shap_e \
  --output ./cura_import

# 2. Open in Cura
cura output/shap_e/model_refined.stl &

# 3. Configure print settings in Cura GUI
# 4. Export G-code from Cura
```

### Integration 2: Blender Post-processing

Refine generated model in Blender:

```bash
# 1. Generate model
python3 design_agent_local.py \
  --prompt "abstract sculpture" \
  --backend shap_e

# 2. Open in Blender
blender output/shap_e/model_refined.stl &

# 3. In Blender:
#    - Edit geometry
#    - Add modifiers
#    - Apply textures
#    - Export refined model

# 4. Use refined model for printing
python3 design_agent_local.py \
  --image refined_model_render.png \
  --backend tripo \
  --slice
```

### Integration 3: OctoPrint API Integration

Monitor and control prints:

```bash
#!/bin/bash

OCTOPI_URL="http://octopi.local"
API_KEY="your_api_key"

# 1. Generate and upload design
python3 design_agent_local.py \
  --prompt "test print" \
  --backend diffusion \
  --slice \
  --octopi_url "$OCTOPI_URL" \
  --octopi_key "$API_KEY"

# 2. Check print queue
curl -H "X-API-Key: $API_KEY" \
  "$OCTOPI_URL/api/files/local"

# 3. Start specific print job
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"command":"select","print":true}' \
  "$OCTOPI_URL/api/files/local/model.gcode"

# 4. Monitor print
curl -H "X-API-Key: $API_KEY" \
  "$OCTOPI_URL/api/job"
```

### Integration 4: Web Service Wrapper

Deploy as REST API:

```python
# api_server.py
from flask import Flask, request, jsonify
import design_agent_local
import threading

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_design():
    data = request.json
    prompt = data.get('prompt')
    backend = data.get('backend', 'shap_e')
    
    config = design_agent_local.Config(backend=backend)
    agent = design_agent_local.DesignAgent(config)
    
    def run_generation():
        result = agent.design_and_print(prompt=prompt)
        return result
    
    # Run in background
    thread = threading.Thread(target=run_generation)
    thread.start()
    
    return jsonify({"status": "generating", "job_id": thread.ident})

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    # Implementation to check job status
    return jsonify({"status": "processing"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run API server:
```bash
pip install flask
python3 api_server.py

# Call API
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test cube","backend":"shap_e"}'
```

### Integration 5: Docker Deployment

Run in containerized environment:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv git ffmpeg slic3r

WORKDIR /app
COPY . .

RUN python3.11 -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["bash", "-c", "source venv/bin/activate && python3 design_agent_local.py --open_gui"]
```

Build and run:
```bash
docker build -t design-agent .
docker run --gpus all -it -p 8080:8080 design-agent
```

---

## Troubleshooting Examples

### Troubleshooting 1: Out of Memory Error

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Use CPU or reduce model precision

```bash
# Use CPU
export CUDA_VISIBLE_DEVICES=""
python3 design_agent_local.py --prompt "test" --backend diffusion

# Or use smaller batch
python3 design_agent_local.py \
  --prompt "test" \
  --backend diffusion \
  --batch_size 1
```

### Troubleshooting 2: Slow Generation

**Symptom**: Generation taking 10+ minutes

**Solutions**:

```bash
# Check if using CPU (should be ~1-2s per step)
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, enable GPU:
# 1. Install NVIDIA drivers
# 2. Install CUDA 11.8+
# 3. Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use faster backend (TripoSR)
python3 design_agent_local.py \
  --image photo.jpg \
  --backend tripo
```

### Troubleshooting 3: API Authentication Failure

**Error**: `401 Unauthorized`

**Solution**: Verify API keys

```bash
# Check keys are set
echo $OPENAI_API_KEY
echo $STABILITY_API_KEY

# Test OpenAI API
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Test Stability API
curl -H "Authorization: Bearer $STABILITY_API_KEY" \
  https://api.stability.ai/v1/account/balance
```

### Troubleshooting 4: slic3r Not Found

**Error**: `slic3r not found`

**Solution**: Install slic3r

```bash
# Ubuntu/Debian
sudo apt-get install slic3r

# macOS
brew install slic3r

# Verify
slic3r --version

# Or specify full path
python3 design_agent_local.py \
  --backend shap_e \
  --prompt "test" \
  --slice \
  --slic3r_path /usr/bin/slic3r
```

### Troubleshooting 5: GUI Won't Launch

**Error**: `ModuleNotFoundError: No module named 'tkinter'`

**Solution**: Install tkinter

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk@3.11

# Then test
python3 -m tkinter  # Should open window

# Launch GUI
python3 design_agent_local.py --open_gui
```

---

## Performance Optimization

### Optimization 1: Enable FP16 (Faster)

Use half-precision floats for 2x speedup:

```bash
export TORCH_DTYPE=float16
python3 design_agent_local.py --prompt "test" --backend diffusion
```

### Optimization 2: Use Model Caching

First run caches models, subsequent runs are faster:

```bash
# First run (slow, downloads models ~4GB)
python3 design_agent_local.py --prompt "test" --backend diffusion

# Second run (fast, uses cache)
python3 design_agent_local.py --prompt "different design" --backend diffusion
```

### Optimization 3: Batch Multiple Generations

Generate several designs in one session:

```bash
#!/bin/bash

python3 << 'EOF'
import design_agent_local

config = design_agent_local.Config(backend=design_agent_local.Backend.SHAP_E)
agent = design_agent_local.DesignAgent(config)

prompts = ["cube", "sphere", "pyramid", "cylinder"]

for prompt in prompts:
    result = agent.design_and_print(prompt=prompt)
    print(f"{prompt}: {result['status']}")
EOF
```

---

## Advanced Configuration

### Custom Configuration File

Create custom config for specific use case:

```json
{
  "backend": "gaussian",
  "output_dir": "photogrammetry_output",
  "work_dir": "photogrammetry_work",
  "slice": false,
  "layer_height": 0.1,
  "nozzle_diameter": 0.4,
  "print_temperature": 210,
  "bed_temperature": 65,
  "verbose": true,
  "device": "cuda"
}
```

Use config:
```bash
python3 design_agent_local.py --config custom_config.json --image_dir ./photos
```

---

**Version**: 1.0  
**Last Updated**: 2025

For more examples, check the GitHub repository: https://github.com/yourusername/design-agent-local/examples/
