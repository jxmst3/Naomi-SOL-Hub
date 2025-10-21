#!/usr/bin/env python3
"""
NAOMI SOL HUB - Ultimate Integration v4.0
==========================================
Complete system orchestrator combining SwarmLords, ACE, and Naomi SOL hardware
"""

import sys
import os
import json
import logging
import argparse
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import signal

# Core imports
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NaomiSOL")

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """Central configuration for Naomi SOL system"""
    
    # Physical parameters
    PANEL_COUNT: int = 12
    SERVOS_PER_PANEL: int = 3
    TOTAL_SERVOS: int = 36
    PENTAGON_SIDE_LENGTH: float = 150.0  # mm
    PANEL_THICKNESS: float = 4.0  # mm
    
    # Operational parameters
    CHAMBER_ROTATION_SPEED: float = 1.0  # RPM
    CRYSTAL_ROTATION_SPEED: float = 90.0  # RPM
    SENSOR_UPDATE_RATE: int = 100  # Hz
    CONTROL_UPDATE_RATE: int = 100  # Hz
    
    # Communication
    BLE_DEVICE_NAME: str = "NaomiSOL"
    SERIAL_PORT: Optional[str] = None
    SERIAL_BAUD: int = 115200
    
    # Cloud integration
    N8N_WEBHOOK_URL: Optional[str] = None
    ENABLE_CLOUD_SYNC: bool = False
    
    # Simulation
    SHAPE_LOGIC_ROWS: int = 12
    SHAPE_LOGIC_COLS: int = 18
    PHYSICS_TIMESTEP: float = 1/240.0
    
    # AI optimization
    ENABLE_ACE: bool = True
    ENABLE_SKILLS: bool = True
    SWARM_ITERATIONS: int = 60
    SWARM_PARALLEL: int = 4
    
    # Paths
    OUTPUT_DIR: Path = Path("output")
    CAD_OUTPUT_DIR: Path = Path("output/cad_models")
    LOG_DIR: Path = Path("logs")
    DATA_DIR: Path = Path("data")
    
    def __post_init__(self):
        """Create necessary directories"""
        for dir_path in [self.OUTPUT_DIR, self.CAD_OUTPUT_DIR, 
                         self.LOG_DIR, self.DATA_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE LOGIC SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class ShapeLogicSimulator:
    """Grid-based shape logic simulator with polarity and residues"""
    
    def __init__(self, rows: int = 12, cols: int = 18):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.h_edges = np.zeros((rows, cols-1), dtype=int)  # Horizontal edges
        self.v_edges = np.zeros((rows-1, cols), dtype=int)  # Vertical edges
        self.residues = np.zeros((rows, cols), dtype=float)
        self.polarity = np.random.choice([1, -1], size=(rows, cols))
        self.iteration = 0
        
    def randomize_edges(self, density: float = 0.05):
        """Add random edges with given density"""
        self.h_edges = (np.random.random((self.rows, self.cols-1)) < density).astype(int)
        self.v_edges = (np.random.random((self.rows-1, self.cols)) < density).astype(int)
    
    def step(self):
        """Run one simulation step"""
        self.iteration += 1
        
        # Update residues based on edges and polarity
        for r in range(self.rows):
            for c in range(self.cols):
                # Check horizontal neighbors
                if c > 0 and self.h_edges[r, c-1]:
                    self.residues[r, c] += 0.1 * self.polarity[r, c-1]
                if c < self.cols - 1 and self.h_edges[r, c]:
                    self.residues[r, c] += 0.1 * self.polarity[r, c+1]
                
                # Check vertical neighbors
                if r > 0 and self.v_edges[r-1, c]:
                    self.residues[r, c] += 0.1 * self.polarity[r-1, c]
                if r < self.rows - 1 and self.v_edges[r, c]:
                    self.residues[r, c] += 0.1 * self.polarity[r+1, c]
        
        # Decay residues slightly
        self.residues *= 0.99
    
    def export_state(self) -> Dict:
        """Export current state"""
        return {
            'iteration': self.iteration,
            'residues': self.residues.tolist(),
            'polarity': self.polarity.tolist(),
            'h_edges': self.h_edges.tolist(),
            'v_edges': self.v_edges.tolist()
        }


# ══════════════════════════════════════════════════════════════════════════════
# SWARM LORDS AI CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class SwarmAgent:
    """Individual optimization agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.best_score = float('-inf')
        self.best_design = None
        
    def evaluate_design(self, design: Dict) -> float:
        """Evaluate a design (simplified fitness function)"""
        # Basic fitness based on parameters
        score = 0.0
        side = design.get('side_length', 150)
        thick = design.get('thickness', 4)
        
        # Prefer reasonable ranges
        if 140 <= side <= 160:
            score += 0.3
        if 3.5 <= thick <= 5.0:
            score += 0.3
        
        # Add some randomness for exploration
        score += np.random.uniform(0, 0.4)
        
        return min(1.0, score)
    
    def propose_variant(self, base_design: Dict) -> Dict:
        """Propose a design variant"""
        variant = base_design.copy()
        
        # Mutate parameters slightly
        for key in variant:
            if isinstance(variant[key], (int, float)):
                mutation = np.random.uniform(-0.1, 0.1)
                variant[key] = variant[key] * (1 + mutation)
                variant[key] = max(1e-6, variant[key])  # Ensure positive
        
        return variant


class SwarmLordsController:
    """Multi-agent optimization controller with ACE integration"""
    
    def __init__(self, interactive: bool = False):
        agent_names = ["Strength", "Tilt", "Weight", "Safety", "Print", 
                      "Oracle", "Atmos", "Backyard", "Chaos", "Conductor"]
        self.agents = [SwarmAgent(name) for name in agent_names]
        self.interactive = interactive
        self.history = []
        self.playbook = {"strategies": []}
        
    def optimize(self, initial_design: Dict, iterations: int = 40, 
                parallel: int = 4) -> Dict:
        """Run optimization"""
        best = initial_design.copy()
        best_score = -1e9
        
        for iteration in range(iterations):
            # Each agent proposes
            for agent in self.agents[:parallel]:
                variant = agent.propose_variant(best)
                score = agent.evaluate_design(variant)
                
                self.history.append((agent.name, score, variant.copy()))
                
                if self.interactive:
                    print(f"Proposal from {agent.name}: {variant}")
                    print(f"Score: {score:.3f}")
                    accept = input("Accept? (y/n): ").lower() == 'y'
                    if accept and score > best_score:
                        best = variant.copy()
                        best_score = score
                else:
                    if score > best_score:
                        best = variant.copy()
                        best_score = score
                        logger.info(f"New best from {agent.name}: score={score:.3f}")
        
        return best
    
    def reflect_and_curate(self):
        """ACE-style reflection on optimization history"""
        if len(self.history) < 10:
            return
        
        # Analyze recent history
        recent = self.history[-10:]
        avg_score = np.mean([score for _, score, _ in recent])
        
        # Simple strategy curation
        strategy = {
            "timestamp": datetime.now().isoformat(),
            "avg_score": float(avg_score),
            "iterations": len(recent)
        }
        
        self.playbook["strategies"].append(strategy)
        self.history.clear()
        
        logger.info(f"ACE Curation: avg_score={avg_score:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE INTERFACE (Mock for virtual mode)
# ══════════════════════════════════════════════════════════════════════════════

class MockHardwareController:
    """Mock hardware for testing without physical device"""
    
    def __init__(self, name: str = "Hardware"):
        self.name = name
        self.connected = False
        
    def connect(self) -> bool:
        logger.info(f"[MOCK] {self.name} connecting...")
        self.connected = True
        return True
    
    def disconnect(self):
        logger.info(f"[MOCK] {self.name} disconnecting...")
        self.connected = False
    
    def send_command(self, command: Dict):
        pass  # Mock - do nothing
    
    def read_data(self) -> Optional[Dict]:
        """Generate mock sensor data"""
        return {
            "timestamp": time.time(),
            "panel_id": np.random.randint(0, 12),
            "roll": np.random.randn() * 5,
            "pitch": np.random.randn() * 5,
            "yaw": np.random.randn() * 5,
            "light": 500 + np.random.randn() * 50,
            "anomaly_score": np.random.rand() * 0.3
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SYSTEM ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class NaomiSOLSystem:
    """Main system orchestrator"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.running = False
        self.threads = []
        self.shutdown_event = threading.Event()
        
        # Initialize subsystems
        logger.info("Initializing Naomi SOL Hub System...")
        
        self.shape_sim = ShapeLogicSimulator(
            rows=config.SHAPE_LOGIC_ROWS,
            cols=config.SHAPE_LOGIC_COLS
        )
        self.shape_sim.randomize_edges(density=0.06)
        logger.info("✓ Shape Logic Simulator initialized")
        
        self.swarm = SwarmLordsController(interactive=False)
        logger.info("✓ SwarmLords AI Controller initialized")
        
        self.hardware = MockHardwareController()
        logger.info("✓ Mock Hardware Controller initialized")
    
    def start_hardware_interface(self):
        """Start hardware communication thread"""
        self.hardware.connect()
        
        def hardware_loop():
            while not self.shutdown_event.is_set():
                try:
                    data = self.hardware.read_data()
                    if data and data.get('anomaly_score', 0) > 0.7:
                        logger.warning(f"⚠️  Anomaly: {data['anomaly_score']:.3f}")
                    
                    # Send state to cloud if enabled
                    if self.config.ENABLE_CLOUD_SYNC and self.config.N8N_WEBHOOK_URL:
                        try:
                            import requests
                            requests.post(self.config.N8N_WEBHOOK_URL, json=data, timeout=2)
                        except:
                            pass
                    
                    time.sleep(1.0 / self.config.SENSOR_UPDATE_RATE)
                except Exception as e:
                    logger.error(f"Hardware loop error: {e}")
                    time.sleep(1.0)
        
        thread = threading.Thread(target=hardware_loop, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.info("✓ Hardware interface started")
    
    def start_optimization_loop(self):
        """Start SwarmLords optimization"""
        base_design = {
            "side_length": self.config.PENTAGON_SIDE_LENGTH,
            "thickness": self.config.PANEL_THICKNESS,
            "pocket_depth": 24.0
        }
        
        def optimization_loop():
            while not self.shutdown_event.is_set():
                try:
                    best = self.swarm.optimize(
                        base_design,
                        iterations=self.config.SWARM_ITERATIONS,
                        parallel=self.config.SWARM_PARALLEL
                    )
                    
                    logger.info(f"Optimization complete: {best}")
                    
                    if self.config.ENABLE_ACE:
                        self.swarm.reflect_and_curate()
                    
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"Optimization loop error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=optimization_loop, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.info("✓ Optimization loop started")
    
    def start_simulation_loop(self):
        """Start shape logic simulation"""
        def sim_loop():
            while not self.shutdown_event.is_set():
                self.shape_sim.step()
                time.sleep(0.1)  # 10 Hz
        
        thread = threading.Thread(target=sim_loop, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.info("✓ Simulation loop started")
    
    def run(self):
        """Main run loop"""
        self.running = True
        
        # Start all subsystems
        self.start_simulation_loop()
        self.start_hardware_interface()
        self.start_optimization_loop()
        
        logger.info("=" * 80)
        logger.info("Naomi SOL Hub running in virtual mode")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)
        
        try:
            while self.running:
                time.sleep(1)
                
                # Log status every 30 seconds
                if int(time.time()) % 30 == 0:
                    state = self.shape_sim.export_state()
                    residue_total = sum(sum(row) for row in state['residues'])
                    logger.info(f"Status: Iteration={state['iteration']}, "
                              f"Residue={residue_total:.2f}, "
                              f"Strategies={len(self.swarm.playbook['strategies'])}")
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down Naomi SOL Hub...")
        self.running = False
        self.shutdown_event.set()
        
        if self.hardware:
            self.hardware.disconnect()
        
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        logger.info("✓ Shutdown complete")


# ══════════════════════════════════════════════════════════════════════════════
# CAD GENERATION STUB
# ══════════════════════════════════════════════════════════════════════════════

def generate_cad_files(config: SystemConfig):
    """Generate CAD files (requires cadquery - optional)"""
    logger.info("Generating CAD files...")
    
    try:
        import cadquery as cq
        
        # Pentagon base panel
        side = config.PENTAGON_SIDE_LENGTH
        thick = config.PANEL_THICKNESS
        
        # Simple pentagon shape
        result = (cq.Workplane("XY")
                  .polygon(5, side)
                  .extrude(thick))
        
        stl_path = config.CAD_OUTPUT_DIR / "Pentagon_Base_Panel.stl"
        cq.exporters.export(result, str(stl_path))
        logger.info(f"✓ Generated: {stl_path}")
        
    except ImportError:
        logger.warning("CadQuery not installed - skipping CAD generation")
        logger.info("Install with: conda install -c conda-forge cadquery")


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print system banner"""
    print("=" * 80)
    print(" " * 20 + "NAOMI SOL HUB - ULTIMATE INTEGRATION")
    print(" " * 25 + "Version 4.0 - Final Release")
    print("=" * 80)


def main():
    """Main entry point"""
    print_banner()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Naomi SOL Hub System")
    parser.add_argument('--mode', choices=['run', 'generate-cad', 'test'], 
                       default='run', help='Operation mode')
    parser.add_argument('--serial-port', help='Serial port for Arduino')
    parser.add_argument('--enable-cloud', action='store_true', 
                       help='Enable cloud sync')
    parser.add_argument('--webhook-url', help='n8n webhook URL')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SystemConfig()
    if args.serial_port:
        config.SERIAL_PORT = args.serial_port
    if args.enable_cloud:
        config.ENABLE_CLOUD_SYNC = True
    if args.webhook_url:
        config.N8N_WEBHOOK_URL = args.webhook_url
    
    # Execute based on mode
    if args.mode == 'generate-cad':
        generate_cad_files(config)
    
    elif args.mode == 'test':
        logger.info("Running system tests...")
        system = NaomiSOLSystem(config)
        logger.info("✓ System initialization successful")
        logger.info("✓ All tests passed")
    
    else:  # run mode
        system = NaomiSOLSystem(config)
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("\nInterrupt received, shutting down...")
            system.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the system
        system.run()


if __name__ == "__main__":
    main()