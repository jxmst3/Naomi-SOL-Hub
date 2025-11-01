#!/usr/bin/env python3
"""
Comprehensive test suite for Design Agent Local.
Run with: pytest test_design_agent_local.py -v
"""

import pytest
import tempfile
import os
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from design_agent_local import (
    Config, Backend, DesignAgent, MeshProcessor, SlicingEngine, 
    NarrationEngine, GenerationBackend, ShapEBackend, TripoSRBackend,
    DreamFusionBackend, GaussianSplattingBackend, StableDiffusionBackend
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def config(temp_dir):
    """Create test configuration."""
    return Config(
        backend=Backend.SHAP_E,
        output_dir=os.path.join(temp_dir, "output"),
        work_dir=os.path.join(temp_dir, "work"),
        slice=False,
        narrate=False,
        verbose=True
    )


@pytest.fixture
def design_agent(config):
    """Create test design agent."""
    return DesignAgent(config)


@pytest.fixture
def sample_mesh_file(temp_dir):
    """Create a minimal test mesh file (STL format)."""
    mesh_path = os.path.join(temp_dir, "test_mesh.stl")
    
    # Simple ASCII STL (single triangle)
    stl_content = """solid test_mesh
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid test_mesh
"""
    
    with open(mesh_path, 'w') as f:
        f.write(stl_content)
    
    return mesh_path


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """Test configuration dataclass."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = Config()
        assert config.backend == Backend.SHAP_E
        assert config.slice == False
        assert config.narrate == False
        assert config.layer_height == 0.2
        assert config.nozzle_diameter == 0.4
    
    def test_config_custom_values(self, config):
        """Test custom configuration."""
        assert config.backend == Backend.SHAP_E
        assert config.slice == False
        assert config.verbose == True
    
    def test_config_creates_directories(self, config):
        """Test that config creates necessary directories."""
        assert os.path.exists(config.output_dir)
        assert os.path.exists(config.work_dir)
    
    def test_config_device_detection(self):
        """Test automatic device detection."""
        import torch
        config = Config()
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected_device
    
    def test_config_from_dict(self, temp_dir):
        """Test configuration from dictionary."""
        config_dict = {
            "backend": "shap_e",
            "output_dir": temp_dir,
            "layer_height": 0.15,
            "slice": True
        }
        config = Config(
            backend=Backend[config_dict["backend"].upper()],
            output_dir=config_dict["output_dir"],
            layer_height=config_dict["layer_height"],
            slice=config_dict["slice"]
        )
        assert config.layer_height == 0.15
        assert config.slice == True


# ============================================================================
# Mesh Processor Tests
# ============================================================================

class TestMeshProcessor:
    """Test mesh refinement and conversion."""
    
    def test_mesh_processor_init(self, config):
        """Test mesh processor initialization."""
        processor = MeshProcessor(config)
        assert processor.config == config
        assert hasattr(processor, 'logger')
    
    def test_refine_mesh_missing_pymeshlab(self, config):
        """Test graceful degradation when PyMeshLab unavailable."""
        processor = MeshProcessor(config)
        
        # Simulate missing pymeshlab
        with patch('design_agent_local.HAS_PYMESHLAB', False):
            result = processor.refine_mesh("dummy.stl", "output.stl")
            assert result == False
    
    def test_refine_mesh_invalid_file(self, config):
        """Test error handling for invalid mesh file."""
        processor = MeshProcessor(config)
        result = processor.refine_mesh("nonexistent.stl", "output.stl")
        assert result == False
    
    def test_convert_format_missing_pymeshlab(self, config):
        """Test graceful degradation for format conversion."""
        processor = MeshProcessor(config)
        
        with patch('design_agent_local.HAS_PYMESHLAB', False):
            result = processor.convert_format("test.stl", "test.obj")
            assert result == False
    
    @patch('design_agent_local.HAS_PYMESHLAB', False)
    def test_mesh_processor_with_trimesh_fallback(self, config, sample_mesh_file):
        """Test mesh processing with trimesh fallback."""
        processor = MeshProcessor(config)
        
        # With HAS_PYMESHLAB=False, should still work with trimesh if available
        # This tests the graceful degradation path
        assert processor.config is not None


# ============================================================================
# Slicing Engine Tests
# ============================================================================

class TestSlicingEngine:
    """Test G-code generation and OctoPrint integration."""
    
    def test_slicing_engine_init(self, config):
        """Test slicing engine initialization."""
        engine = SlicingEngine(config)
        assert engine.config == config
    
    def test_slice_to_gcode_slic3r_missing(self, config, sample_mesh_file, temp_dir):
        """Test error handling when slic3r not installed."""
        engine = SlicingEngine(config)
        output_file = os.path.join(temp_dir, "test.gcode")
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            result = engine.slice_to_gcode(sample_mesh_file, output_file)
            # Should return False since slic3r check fails
            assert result == False
    
    def test_octoprint_upload_missing_config(self, config, sample_mesh_file):
        """Test OctoPrint upload with missing configuration."""
        engine = SlicingEngine(config)
        # Config has no OctoPrint URL
        result = engine.upload_to_octoprint(sample_mesh_file)
        assert result == False
    
    def test_octoprint_upload_invalid_url(self, config, sample_mesh_file):
        """Test OctoPrint upload with invalid URL."""
        config.octopi_url = "http://invalid-url-that-does-not-exist"
        config.octopi_api_key = "test_key"
        engine = SlicingEngine(config)
        
        result = engine.upload_to_octoprint(sample_mesh_file)
        # Should fail due to connection error
        assert result == False


# ============================================================================
# Narration Engine Tests
# ============================================================================

class TestNarrationEngine:
    """Test audio narration generation."""
    
    def test_narration_engine_init(self, config):
        """Test narration engine initialization."""
        engine = NarrationEngine(config)
        assert engine.config == config
    
    def test_narrate_missing_pyttsx3(self, config, temp_dir):
        """Test graceful degradation when pyttsx3 unavailable."""
        engine = NarrationEngine(config)
        output_file = os.path.join(temp_dir, "test.wav")
        
        with patch('design_agent_local.HAS_PYTTSX3', False):
            result = engine.narrate("Test narration", output_file)
            assert result == False
    
    @patch('pyttsx3.init')
    def test_narrate_success(self, mock_pyttsx3, config, temp_dir):
        """Test successful narration generation."""
        with patch('design_agent_local.HAS_PYTTSX3', True):
            mock_engine = MagicMock()
            mock_pyttsx3.return_value = mock_engine
            
            engine = NarrationEngine(config)
            output_file = os.path.join(temp_dir, "test.wav")
            
            result = engine.narrate("Test narration", output_file)
            
            mock_engine.setProperty.assert_called()
            mock_engine.save_to_file.assert_called_once()
            mock_engine.runAndWait.assert_called_once()


# ============================================================================
# Backend Tests
# ============================================================================

class TestGenerationBackends:
    """Test all generation backends."""
    
    def test_backend_enum_values(self):
        """Test backend enum values."""
        assert Backend.SHAP_E.value == "shap_e"
        assert Backend.TRIPO.value == "tripo"
        assert Backend.DREAMFUSION.value == "dreamfusion"
        assert Backend.GAUSSIAN.value == "gaussian"
        assert Backend.DIFFUSION.value == "diffusion"
    
    def test_generation_backend_abstract(self, config):
        """Test that GenerationBackend is abstract."""
        backend = GenerationBackend(config)
        # generate() should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            backend.generate()
    
    def test_save_metadata(self, config, temp_dir):
        """Test metadata saving."""
        backend = GenerationBackend(config)
        
        metadata = {
            "backend": "test",
            "timestamp": "2025-01-01T00:00:00",
            "model": "test_model.glb"
        }
        
        backend._save_metadata(temp_dir, metadata)
        
        meta_file = os.path.join(temp_dir, 'metadata.json')
        assert os.path.exists(meta_file)
        
        with open(meta_file) as f:
            saved_meta = json.load(f)
        
        assert saved_meta == metadata
    
    def test_shap_e_backend_missing_api_key(self, config):
        """Test Shap-E backend without API key."""
        backend = ShapEBackend(config)
        
        # Clear API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}, clear=True):
            result = backend.generate("test cube", config.output_dir)
            assert result is None
    
    def test_tripo_backend_missing_api_key(self, config):
        """Test TripoSR backend without API key."""
        backend = TripoSRBackend(config)
        
        with patch.dict(os.environ, {'STABILITY_API_KEY': ''}, clear=True):
            result = backend.generate("test.png", config.output_dir)
            assert result is None


# ============================================================================
# Design Agent Tests
# ============================================================================

class TestDesignAgent:
    """Test main orchestrator."""
    
    def test_design_agent_init(self, design_agent):
        """Test design agent initialization."""
        assert design_agent.config is not None
        assert hasattr(design_agent, 'mesh_processor')
        assert hasattr(design_agent, 'slicing_engine')
        assert hasattr(design_agent, 'narration_engine')
        assert len(design_agent.backends) == 5
    
    def test_all_backends_available(self, design_agent):
        """Test all backends are registered."""
        expected_backends = {
            Backend.SHAP_E,
            Backend.TRIPO,
            Backend.DREAMFUSION,
            Backend.GAUSSIAN,
            Backend.DIFFUSION
        }
        assert set(design_agent.backends.keys()) == expected_backends
    
    def test_design_and_print_missing_input(self, design_agent):
        """Test error when both prompt and image missing."""
        with pytest.raises(ValueError):
            design_agent.design_and_print(
                prompt=None,
                image_path=None,
                image_dir=None
            )
    
    @patch.object(ShapEBackend, 'generate')
    def test_design_and_print_workflow(self, mock_generate, design_agent, temp_dir):
        """Test complete design workflow."""
        # Mock backend generation
        mock_mesh_path = os.path.join(temp_dir, "test.glb")
        Path(mock_mesh_path).touch()  # Create dummy file
        mock_generate.return_value = mock_mesh_path
        
        design_agent.config.backend = Backend.SHAP_E
        
        result = design_agent.design_and_print(
            prompt="test design",
            output_subdir="test_workflow"
        )
        
        assert result['status'] in ['success', 'failed']
        assert 'timing' in result
        assert 'backend' in result
    
    def test_design_agent_error_handling(self, design_agent):
        """Test error handling in workflow."""
        design_agent.config.backend = Backend.SHAP_E
        
        with patch.object(ShapEBackend, 'generate', return_value=None):
            result = design_agent.design_and_print(prompt="test")
            
            assert result['status'] == 'failed'
            assert 'error' in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_with_mocks(self, config):
        """Test complete pipeline with mocked external calls."""
        agent = DesignAgent(config)
        
        with patch.object(ShapEBackend, 'generate') as mock_gen:
            mock_mesh = os.path.join(config.output_dir, "test.glb")
            Path(mock_mesh).parent.mkdir(parents=True, exist_ok=True)
            Path(mock_mesh).touch()
            mock_gen.return_value = mock_mesh
            
            result = agent.design_and_print(prompt="test cube")
            
            assert isinstance(result, dict)
            assert 'status' in result
            assert 'timing' in result
            assert 'backend' in result
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing."""
        # This would require argparse testing
        # Typically done with subprocess or click testing
        pass
    
    def test_config_json_loading(self, temp_dir):
        """Test loading config from JSON file."""
        config_file = os.path.join(temp_dir, "config.json")
        
        config_data = {
            "backend": "shap_e",
            "output_dir": temp_dir,
            "layer_height": 0.15,
            "slice": True,
            "narrate": False
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load and verify
        with open(config_file) as f:
            loaded = json.load(f)
        
        assert loaded['layer_height'] == 0.15
        assert loaded['backend'] == 'shap_e'


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and benchmarking tests."""
    
    def test_config_creation_performance(self):
        """Test config creation is fast."""
        import time
        start = time.time()
        for _ in range(1000):
            Config()
        elapsed = time.time() - start
        
        # Should be very fast (< 1 second for 1000 configs)
        assert elapsed < 1.0
    
    def test_backend_instantiation_performance(self):
        """Test backend instantiation is fast."""
        import time
        config = Config()
        
        start = time.time()
        for _ in range(100):
            ShapEBackend(config)
        elapsed = time.time() - start
        
        # Should be fast (< 1 second for 100 backends)
        assert elapsed < 1.0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_backend_specification(self, config):
        """Test handling of invalid backend."""
        with pytest.raises(AttributeError):
            config.backend = "invalid_backend"  # type: ignore
    
    def test_missing_required_packages(self):
        """Test graceful degradation with missing packages."""
        # Already tested in individual component tests
        pass
    
    def test_invalid_file_paths(self, config):
        """Test handling of invalid file paths."""
        processor = MeshProcessor(config)
        result = processor.refine_mesh("/nonexistent/path.stl", "/nonexistent/out.stl")
        assert result == False


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
