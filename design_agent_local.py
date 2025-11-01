#!/usr/bin/env python3
"""
Design Agent Local - AI-Powered 3D Design & Printing
Orchestrates text-to-3D, image-to-3D, and multi-view 3D generation with printing.
"""

import argparse
import os
import sys
import time
import torch
import json
import re
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

import numpy as np
from PIL import Image
import requests

# Optional imports with graceful degradation
try:
    import pymeshlab as pymesh
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False

try:
    from diffusers import DiffusionPipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('design_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Data Classes
# ============================================================================

class Backend(Enum):
    SHAP_E = "shap_e"
    TRIPO = "tripo"
    DREAMFUSION = "dreamfusion"
    GAUSSIAN = "gaussian"
    DIFFUSION = "diffusion"


@dataclass
class Config:
    """Configuration container for design agent."""
    backend: Backend = Backend.SHAP_E
    output_dir: str = "output"
    work_dir: str = "work"
    slice: bool = False
    narrate: bool = False
    layer_height: float = 0.2
    nozzle_diameter: float = 0.4
    print_temperature: int = 205
    bed_temperature: int = 60
    octopi_url: Optional[str] = None
    octopi_api_key: Optional[str] = None
    verbose: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Mesh Processing
# ============================================================================

class MeshProcessor:
    """Handles all mesh refinement, conversion, and optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MeshProcessor")
    
    def refine_mesh(self, input_file: str, output_file: str) -> bool:
        """
        Refine mesh: handle point clouds, fill holes, smooth, remesh.
        
        Args:
            input_file: Path to input mesh (STL, OBJ, PLY, GLB)
            output_file: Path to output mesh
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_PYMESHLAB:
            self.logger.warning("PyMeshLab not available; skipping mesh refinement")
            return False
        
        try:
            ms = pymesh.MeshSet()
            ms.load_new_mesh(input_file)
            
            # Check if point cloud
            if ms.current_mesh().face_number() == 0:
                self.logger.info("Detected point cloud; computing normals & Poisson reconstruction")
                ms.compute_normal_for_point_clouds(k=10)
                ms.generate_surface_reconstruction_screened_poisson(depth=8, samplespernode=1.5)
            
            # Refinements
            ms.close_holes(max_hole_size=100)
            ms.remove_duplicate_vertices()
            ms.smooth_laplacian_smooth(iterations=3, lambda_value=0.3)
            ms.remeshing_isotropic_explicit_remeshing(
                target_edge_length=self.config.nozzle_diameter * 2,
                iterations=10
            )
            
            ms.save_current_mesh(output_file)
            self.logger.info(f"Refined mesh saved to {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Mesh refinement failed: {e}")
            return False
    
    def convert_format(self, input_file: str, output_file: str) -> bool:
        """Convert mesh between formats (STL, OBJ, PLY, GLB)."""
        if not HAS_PYMESHLAB:
            self.logger.warning("PyMeshLab not available; cannot convert formats")
            return False
        
        try:
            ms = pymesh.MeshSet()
            ms.load_new_mesh(input_file)
            ms.save_current_mesh(output_file)
            self.logger.info(f"Converted {input_file} to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Format conversion failed: {e}")
            return False


# ============================================================================
# Slicing Engine
# ============================================================================

class SlicingEngine:
    """Handles G-code generation via slic3r."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SlicingEngine")
    
    def slice_to_gcode(self, stl_file: str, gcode_file: str) -> bool:
        """
        Slice STL to G-code using slic3r.
        
        Args:
            stl_file: Path to input STL
            gcode_file: Path to output G-code
            
        Returns:
            True if successful
        """
        try:
            # Check if slic3r is available
            result = subprocess.run(['which', 'slic3r'], capture_output=True)
            if result.returncode != 0:
                self.logger.error("slic3r not found. Install with: apt-get install slic3r")
                return False
            
            cmd = [
                'slic3r',
                stl_file,
                f'--layer-height={self.config.layer_height}',
                f'--nozzle-diameter={self.config.nozzle_diameter}',
                f'--bed-temperature={self.config.bed_temperature}',
                f'--temperature={self.config.print_temperature}',
                '-o', gcode_file
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"G-code generated: {gcode_file}")
            return True
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Slicing failed: {e.stderr.decode()}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during slicing: {e}")
            return False
    
    def upload_to_octoprint(self, gcode_file: str) -> bool:
        """Upload G-code to OctoPrint instance."""
        if not self.config.octopi_url or not self.config.octopi_api_key:
            self.logger.warning("OctoPrint URL or API key not configured")
            return False
        
        try:
            url = f"{self.config.octopi_url}/api/files/local"
            headers = {"X-API-Key": self.config.octopi_api_key}
            
            with open(gcode_file, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files, headers=headers, timeout=30)
            
            if response.status_code in [200, 201]:
                self.logger.info(f"Uploaded to OctoPrint: {os.path.basename(gcode_file)}")
                return True
            else:
                self.logger.error(f"OctoPrint upload failed: {response.text}")
                return False
        
        except Exception as e:
            self.logger.error(f"OctoPrint connection failed: {e}")
            return False


# ============================================================================
# Generation Backends (Abstract)
# ============================================================================

class GenerationBackend:
    """Base class for all generation backends."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate(self, **kwargs) -> Optional[str]:
        """
        Generate 3D model.
        
        Returns:
            Path to generated mesh file, or None if failed
        """
        raise NotImplementedError
    
    def _save_metadata(self, output_dir: str, metadata: Dict[str, Any]):
        """Save generation metadata."""
        meta_file = os.path.join(output_dir, 'metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)


# ============================================================================
# Specific Backends
# ============================================================================

class ShapEBackend(GenerationBackend):
    """Generate 3D models using OpenAI Shap-E API."""
    
    def generate(self, prompt: str, output_dir: str = None) -> Optional[str]:
        """Generate mesh from text prompt."""
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Generating Shap-E model: '{prompt}'")
            
            # Note: Requires OPENAI_API_KEY environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("OPENAI_API_KEY not set")
                return None
            
            headers = {"Authorization": f"Bearer {api_key}"}
            url = "https://api.openai.com/v1/models/shap-e/generate"
            
            payload = {
                "prompt": prompt,
                "guidance_scale": 15.0,
                "num_inference_steps": 64
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            
            if response.status_code != 200:
                self.logger.error(f"API error: {response.text}")
                return None
            
            result = response.json()
            
            # Save mesh (typically comes as GLB)
            mesh_path = os.path.join(output_dir, "model_shap_e.glb")
            mesh_data = requests.get(result['mesh_url']).content
            
            with open(mesh_path, 'wb') as f:
                f.write(mesh_data)
            
            self.logger.info(f"Shap-E model saved: {mesh_path}")
            
            self._save_metadata(output_dir, {
                "backend": "shap_e",
                "prompt": prompt,
                "model": mesh_path,
                "guidance_scale": 15.0,
                "num_steps": 64
            })
            
            return mesh_path
        
        except Exception as e:
            self.logger.error(f"Shap-E generation failed: {e}")
            return None


class TripoSRBackend(GenerationBackend):
    """Generate 3D models using Stability AI TripoSR API."""
    
    def generate(self, image_path: str, output_dir: str = None) -> Optional[str]:
        """Generate mesh from image."""
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Generating TripoSR model from: {image_path}")
            
            api_key = os.getenv("STABILITY_API_KEY")
            if not api_key:
                self.logger.error("STABILITY_API_KEY not set")
                return None
            
            # TripoSR API endpoint
            url = "https://api.stability.ai/v2beta/3d/stable-image-to-3d"
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                headers = {'Authorization': f'Bearer {api_key}'}
                response = requests.post(url, files=files, headers=headers, timeout=120)
            
            if response.status_code != 200:
                self.logger.error(f"TripoSR API error: {response.text}")
                return None
            
            mesh_path = os.path.join(output_dir, "model_tripo.glb")
            with open(mesh_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"TripoSR model saved: {mesh_path}")
            
            self._save_metadata(output_dir, {
                "backend": "tripo_sr",
                "source_image": image_path,
                "model": mesh_path
            })
            
            return mesh_path
        
        except Exception as e:
            self.logger.error(f"TripoSR generation failed: {e}")
            return None


class DreamFusionBackend(GenerationBackend):
    """Generate 3D models using DreamFusion (via subprocess)."""
    
    def generate(self, prompt: Optional[str] = None, image_path: Optional[str] = None,
                 output_dir: str = None) -> Optional[str]:
        """Generate mesh using DreamFusion (text or image conditioning)."""
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info("Cloning DreamFusion repository...")
            
            work_dir = Path(self.config.work_dir) / "dreamfusion"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            repo_url = "https://github.com/ashawkey/stable-dreamfusion.git"
            repo_path = work_dir / "stable-dreamfusion"
            
            if not repo_path.exists():
                subprocess.run(['git', 'clone', repo_url, str(repo_path)], 
                             check=True, capture_output=True)
            
            # Prepare command
            cmd = [
                'python', str(repo_path / 'main.py'),
                '--image', image_path if image_path else '',
                '--text', prompt if prompt else '',
                '--workspace', str(work_dir / 'workspace'),
                '--guidance_scale', '100',
                '--iters', '10000'
            ]
            
            # Remove empty args
            cmd = [arg for arg in cmd if arg]
            
            self.logger.info(f"Running DreamFusion: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, timeout=3600)
            
            # Locate output mesh
            mesh_pattern = list(work_dir.glob('workspace/*/*.obj'))
            if not mesh_pattern:
                mesh_pattern = list(work_dir.glob('workspace/*/*.glb'))
            
            if mesh_pattern:
                mesh_path = str(mesh_pattern[0])
                self.logger.info(f"DreamFusion model: {mesh_path}")
                
                self._save_metadata(output_dir, {
                    "backend": "dreamfusion",
                    "prompt": prompt,
                    "image": image_path,
                    "model": mesh_path,
                    "guidance_scale": 100,
                    "iterations": 10000
                })
                
                return mesh_path
            else:
                self.logger.error("DreamFusion output not found")
                return None
        
        except Exception as e:
            self.logger.error(f"DreamFusion generation failed: {e}")
            return None


class GaussianSplattingBackend(GenerationBackend):
    """Generate 3D models using multi-view Gaussian Splatting."""
    
    def generate(self, image_dir: str, output_dir: str = None) -> Optional[str]:
        """Generate mesh from multiple viewpoint images."""
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Generating Gaussian Splatting from: {image_dir}")
            
            work_dir = Path(self.config.work_dir) / "gaussian"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            repo_url = "https://github.com/graphdeco-inria/gaussian-splatting.git"
            repo_path = work_dir / "gaussian-splatting"
            
            if not repo_path.exists():
                subprocess.run(['git', 'clone', repo_url, str(repo_path)],
                             check=True, capture_output=True)
            
            # Prepare COLMAP for structure-from-motion
            self.logger.info("Running COLMAP SfM...")
            colmap_cmd = [
                'colmap', 'automatic_reconstructor',
                '--image_path', image_dir,
                '--workspace_path', str(work_dir / 'colmap')
            ]
            
            subprocess.run(colmap_cmd, timeout=1800)
            
            # Train Gaussian Splatting
            self.logger.info("Training Gaussian Splatting model...")
            train_cmd = [
                'python', str(repo_path / 'train.py'),
                '-s', str(work_dir / 'colmap'),
                '-m', str(work_dir / 'output'),
                '--iterations', '30000'
            ]
            
            subprocess.run(train_cmd, check=True, timeout=3600)
            
            # Export to mesh
            self.logger.info("Exporting to mesh...")
            export_cmd = [
                'python', str(repo_path / 'SIBR_viewers' / 'gaussian_viewer.py'),
                '-m', str(work_dir / 'output'),
                '--export_ply'
            ]
            
            subprocess.run(export_cmd, timeout=600)
            
            # Locate PLY output
            ply_files = list((work_dir / 'output').glob('**/*.ply'))
            if ply_files:
                mesh_path = str(ply_files[0])
                self.logger.info(f"Gaussian Splatting mesh: {mesh_path}")
                
                self._save_metadata(output_dir, {
                    "backend": "gaussian_splatting",
                    "image_dir": image_dir,
                    "model": mesh_path,
                    "iterations": 30000
                })
                
                return mesh_path
            else:
                self.logger.error("Gaussian Splatting output not found")
                return None
        
        except Exception as e:
            self.logger.error(f"Gaussian Splatting generation failed: {e}")
            return None


class StableDiffusionBackend(GenerationBackend):
    """Generate concept images using Stable Diffusion."""
    
    def generate(self, prompt: str, output_dir: str = None, num_images: int = 4) -> Optional[List[str]]:
        """Generate concept images from text."""
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not HAS_DIFFUSERS:
            self.logger.error("diffusers library not installed")
            return None
        
        try:
            self.logger.info(f"Generating {num_images} images: '{prompt}'")
            
            pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if 'cuda' in self.config.device else torch.float32,
                use_safetensors=True
            )
            pipe = pipe.to(self.config.device)
            
            images = pipe(
                prompt=[prompt] * num_images,
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images
            
            image_paths = []
            for i, img in enumerate(images):
                path = os.path.join(output_dir, f"concept_{i}.png")
                img.save(path)
                image_paths.append(path)
            
            self.logger.info(f"Saved {len(image_paths)} concept images")
            
            self._save_metadata(output_dir, {
                "backend": "stable_diffusion",
                "prompt": prompt,
                "images": image_paths,
                "num_inference_steps": 50,
                "guidance_scale": 7.5
            })
            
            return image_paths
        
        except Exception as e:
            self.logger.error(f"Stable Diffusion generation failed: {e}")
            return None


# ============================================================================
# Audio Narration
# ============================================================================

class NarrationEngine:
    """Generates audio narration for designs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.NarrationEngine")
    
    def narrate(self, text: str, output_file: str) -> bool:
        """Generate TTS audio narration."""
        if not HAS_PYTTSX3:
            self.logger.warning("pyttsx3 not installed; skipping narration")
            return False
        
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
            self.logger.info(f"Narration saved: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Narration failed: {e}")
            return False


# ============================================================================
# Main Design Agent Orchestrator
# ============================================================================

class DesignAgent:
    """Main orchestrator for AI 3D design pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DesignAgent")
        
        # Initialize components
        self.mesh_processor = MeshProcessor(config)
        self.slicing_engine = SlicingEngine(config)
        self.narration_engine = NarrationEngine(config)
        
        # Backend factory
        self.backends = {
            Backend.SHAP_E: ShapEBackend(config),
            Backend.TRIPO: TripoSRBackend(config),
            Backend.DREAMFUSION: DreamFusionBackend(config),
            Backend.GAUSSIAN: GaussianSplattingBackend(config),
            Backend.DIFFUSION: StableDiffusionBackend(config)
        }
    
    def design_and_print(
        self,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        output_subdir: Optional[str] = None,
        refine: bool = True
    ) -> Dict[str, Any]:
        """
        Complete 3D design and print workflow.
        
        Args:
            prompt: Text prompt for generation
            image_path: Single reference image path
            image_dir: Directory of multi-view images
            output_subdir: Custom output subdirectory
            refine: Whether to refine mesh
            
        Returns:
            Dictionary with results, paths, and status
        """
        result = {
            "status": "failed",
            "backend": self.config.backend.value,
            "paths": {},
            "metadata": {},
            "timing": {}
        }
        
        start_time = time.time()
        output_dir = os.path.join(self.config.output_dir, output_subdir or self.config.backend.value)
        
        try:
            # 1. Generation
            self.logger.info(f"Starting generation with {self.config.backend.value}...")
            gen_start = time.time()
            
            backend = self.backends[self.config.backend]
            
            if self.config.backend == Backend.SHAP_E and prompt:
                mesh_path = backend.generate(prompt, output_dir)
            elif self.config.backend == Backend.TRIPO and image_path:
                mesh_path = backend.generate(image_path, output_dir)
            elif self.config.backend == Backend.DREAMFUSION:
                mesh_path = backend.generate(prompt, image_path, output_dir)
            elif self.config.backend == Backend.GAUSSIAN and image_dir:
                mesh_path = backend.generate(image_dir, output_dir)
            elif self.config.backend == Backend.DIFFUSION and prompt:
                mesh_path = backend.generate(prompt, output_dir)
            else:
                raise ValueError("Incompatible backend and input combination")
            
            if not mesh_path:
                raise RuntimeError("Generation failed")
            
            result["timing"]["generation"] = time.time() - gen_start
            result["paths"]["original_mesh"] = mesh_path
            
            # 2. Mesh Refinement
            if refine:
                self.logger.info("Refining mesh...")
                refine_start = time.time()
                
                refined_path = os.path.join(output_dir, "model_refined.stl")
                self.mesh_processor.refine_mesh(mesh_path, refined_path)
                
                result["timing"]["refinement"] = time.time() - refine_start
                result["paths"]["refined_mesh"] = refined_path
                mesh_path = refined_path
            
            # 3. Slicing (if requested)
            if self.config.slice:
                self.logger.info("Slicing for 3D printing...")
                slice_start = time.time()
                
                gcode_path = os.path.join(output_dir, "model.gcode")
                if self.slicing_engine.slice_to_gcode(mesh_path, gcode_path):
                    result["paths"]["gcode"] = gcode_path
                    result["timing"]["slicing"] = time.time() - slice_start
                    
                    # Upload to OctoPrint (if configured)
                    if self.config.octopi_url:
                        self.logger.info("Uploading to OctoPrint...")
                        if self.slicing_engine.upload_to_octoprint(gcode_path):
                            result["paths"]["octoprint_upload"] = True
            
            # 4. Narration (if requested)
            if self.config.narrate:
                self.logger.info("Generating narration...")
                narrate_start = time.time()
                
                narration_text = f"Design generated using {self.config.backend.value}"
                narration_path = os.path.join(output_dir, "narration.wav")
                if self.narration_engine.narrate(narration_text, narration_path):
                    result["paths"]["narration"] = narration_path
                    result["timing"]["narration"] = time.time() - narrate_start
            
            result["status"] = "success"
            result["timing"]["total"] = time.time() - start_time
            result["metadata"] = {
                "prompt": prompt,
                "source_image": image_path,
                "image_directory": image_dir,
                "backend": self.config.backend.value,
                "device": self.config.device,
                "layer_height": self.config.layer_height
            }
            
            self.logger.info(f"✓ Workflow completed in {result['timing']['total']:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            result["error"] = str(e)
        
        return result


# ============================================================================
# CLI & GUI
# ============================================================================

def main_cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="AI-Powered 3D Design & Printing Agent")
    
    parser.add_argument('--backend', choices=[b.value for b in Backend],
                       default='shap_e', help='Generation backend')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--image', type=str, help='Single image for image-to-3D')
    parser.add_argument('--image_dir', type=str, help='Image directory for multi-view 3D')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--slice', action='store_true', help='Slice for 3D printing')
    parser.add_argument('--narrate', action='store_true', help='Generate narration')
    parser.add_argument('--layer_height', type=float, default=0.2, help='Layer height (mm)')
    parser.add_argument('--octopi_url', type=str, help='OctoPrint instance URL')
    parser.add_argument('--octopi_key', type=str, help='OctoPrint API key')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--open_gui', action='store_true', help='Launch GUI instead of CLI')
    
    args = parser.parse_args()
    
    if args.open_gui:
        gui_main()
        return
    
    # Validate inputs
    if not args.prompt and not args.image and not args.image_dir:
        parser.error("Provide --prompt, --image, or --image_dir")
    
    # Create config
    config = Config(
        backend=Backend[args.backend.upper()],
        output_dir=args.output,
        slice=args.slice,
        narrate=args.narrate,
        layer_height=args.layer_height,
        octopi_url=args.octopi_url,
        octopi_api_key=args.octopi_key,
        verbose=args.verbose
    )
    
    # Run agent
    agent = DesignAgent(config)
    result = agent.design_and_print(
        prompt=args.prompt,
        image_path=args.image,
        image_dir=args.image_dir
    )
    
    # Print results
    print("\n" + "="*60)
    print("DESIGN GENERATION COMPLETE")
    print("="*60)
    print(f"Status: {result['status'].upper()}")
    print(f"Backend: {result['backend']}")
    print(f"Total Time: {result['timing'].get('total', 0):.2f}s")
    print("\nGenerated Files:")
    for key, path in result['paths'].items():
        print(f"  {key}: {path}")
    print("="*60 + "\n")


def gui_main():
    """Graphical user interface (Tkinter)."""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError:
        print("ERROR: tkinter not available. Install with: apt-get install python3-tk")
        return
    
    class DesignAgentGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("Design Agent - AI 3D Design & Printing")
            self.root.geometry("800x600")
            self.config = Config()
            self.agent = DesignAgent(self.config)
            
            self.setup_ui()
        
        def setup_ui(self):
            # Backend selection
            ttk.Label(self.root, text="Backend:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
            self.backend_var = tk.StringVar(value='shap_e')
            backend_combo = ttk.Combobox(
                self.root,
                textvariable=self.backend_var,
                values=[b.value for b in Backend],
                state='readonly'
            )
            backend_combo.grid(row=0, column=1, sticky='ew', padx=10, pady=5)
            
            # Prompt input
            ttk.Label(self.root, text="Prompt:").grid(row=1, column=0, sticky='nw', padx=10, pady=5)
            self.prompt_text = tk.Text(self.root, height=4, width=50)
            self.prompt_text.grid(row=1, column=1, sticky='ew', padx=10, pady=5)
            
            # Image selection
            ttk.Label(self.root, text="Image:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
            self.image_path_var = tk.StringVar()
            ttk.Entry(self.root, textvariable=self.image_path_var, state='readonly').grid(
                row=2, column=1, sticky='ew', padx=10, pady=5
            )
            ttk.Button(self.root, text="Browse", command=self.select_image).grid(
                row=2, column=2, padx=5, pady=5
            )
            
            # Options
            self.slice_var = tk.BooleanVar()
            ttk.Checkbutton(self.root, text="Slice for 3D Printing", variable=self.slice_var).grid(
                row=3, column=0, columnspan=2, sticky='w', padx=10, pady=5
            )
            
            self.narrate_var = tk.BooleanVar()
            ttk.Checkbutton(self.root, text="Generate Narration", variable=self.narrate_var).grid(
                row=4, column=0, columnspan=2, sticky='w', padx=10, pady=5
            )
            
            # Generate button
            ttk.Button(self.root, text="Generate Design", command=self.generate).grid(
                row=5, column=0, columnspan=3, sticky='ew', padx=10, pady=20
            )
            
            # Output
            ttk.Label(self.root, text="Status:").grid(row=6, column=0, sticky='nw', padx=10, pady=5)
            self.output_text = tk.Text(self.root, height=8, width=80, state='disabled')
            self.output_text.grid(row=6, column=1, columnspan=2, sticky='ew', padx=10, pady=5)
            
            self.root.columnconfigure(1, weight=1)
        
        def select_image(self):
            path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
            if path:
                self.image_path_var.set(path)
        
        def generate(self):
            try:
                prompt = self.prompt_text.get("1.0", "end-1c").strip()
                image_path = self.image_path_var.get() or None
                
                self.config.backend = Backend[self.backend_var.get().upper()]
                self.config.slice = self.slice_var.get()
                self.config.narrate = self.narrate_var.get()
                
                self.agent = DesignAgent(self.config)
                
                self.log("Starting generation...")
                result = self.agent.design_and_print(prompt=prompt, image_path=image_path)
                
                if result['status'] == 'success':
                    self.log(f"✓ Success! Total time: {result['timing']['total']:.2f}s")
                    self.log(f"\nGenerated files:")
                    for key, path in result['paths'].items():
                        self.log(f"  {key}: {path}")
                else:
                    self.log(f"✗ Failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        def log(self, message):
            self.output_text.config(state='normal')
            self.output_text.insert('end', f"{message}\n")
            self.output_text.see('end')
            self.output_text.config(state='disabled')
            self.root.update()
    
    root = tk.Tk()
    app = DesignAgentGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main_cli()
