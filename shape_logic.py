# sim/shape_logic.py
"""
Enhanced Shape Logic Simulator
==============================
Grid-based simulation with polarity edges and square formation.
Improved with expert physics and engineering insights.
"""

import math
import random
import json
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("ShapeLogic")


class EdgePolarity:
    """
    Represents polarity on an edge: -1, 0 (none), +1.
    Enhanced with energy dynamics and quantum-inspired properties.
    """
    
    def __init__(self, polarity: int = 0, energy: float = 0.0):
        self.polarity = int(polarity)
        self.energy = float(energy)
        self.phase = 0.0  # Phase angle for wave properties
        self.coherence = 1.0  # Coherence factor
        
    def is_set(self) -> bool:
        """Check if polarity is non-zero."""
        return self.polarity != 0
    
    def set(self, polarity: int, energy: float = 1.0):
        """Set polarity and energy values with validation."""
        if polarity not in [-1, 0, 1]:
            raise ValueError(f"Invalid polarity: {polarity}")
        self.polarity = int(polarity)
        self.energy = max(0.0, float(energy))  # Energy can't be negative
        
    def clear(self):
        """Reset polarity and energy to defaults."""
        self.polarity = 0
        self.energy = 0.0
        self.phase = 0.0
        self.coherence = 1.0
        
    def invert(self):
        """Invert polarity if set, maintaining energy."""
        if self.polarity != 0:
            self.polarity = -self.polarity
            
    def decay(self, rate: float = 0.01):
        """Apply energy decay over time."""
        self.energy *= (1.0 - rate)
        if self.energy < 0.01:  # Threshold for clearing
            self.clear()
            
    def resonate(self, frequency: float, dt: float = 0.016):
        """Apply resonance effects."""
        self.phase += frequency * dt * 2 * math.pi
        self.phase = self.phase % (2 * math.pi)
        
    def copy(self):
        """Return a deep copy of this EdgePolarity."""
        ep = EdgePolarity(self.polarity, self.energy)
        ep.phase = self.phase
        ep.coherence = self.coherence
        return ep
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "polarity": self.polarity,
            "energy": self.energy,
            "phase": self.phase,
            "coherence": self.coherence
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Deserialize from dictionary."""
        ep = cls(data.get("polarity", 0), data.get("energy", 0.0))
        ep.phase = data.get("phase", 0.0)
        ep.coherence = data.get("coherence", 1.0)
        return ep


@dataclass
class SquareBlock:
    """
    Represents a formed square occupying a 2x2 block.
    Enhanced with physics properties and Naomi SOL integration.
    """
    r: int  # Top-left row
    c: int  # Top-left column
    energy: float = 1.0
    residue: int = 0
    formation_time: float = 0.0
    stability: float = 1.0
    resonance_freq: float = 1.0
    
    # Naomi SOL specific properties
    optical_reflectance: float = 0.95  # Mirror surface property
    servo_angle_x: float = 0.0  # Tilt angle X
    servo_angle_y: float = 0.0  # Tilt angle Y
    
    def add_energy(self, e: float):
        """Add energy to the square with saturation."""
        self.energy = min(10.0, self.energy + e)  # Cap at 10.0
        
    def add_residue(self, n: int = 1):
        """Add residue count."""
        self.residue += n
        
    def update_stability(self, dt: float = 0.016):
        """Update stability based on energy and time."""
        # Stability increases with energy and decreases with residue
        target_stability = min(1.0, self.energy / 5.0) * max(0.5, 1.0 - self.residue * 0.1)
        self.stability += (target_stability - self.stability) * dt * 2.0
        
    def apply_servo_tilt(self, angle_x: float, angle_y: float):
        """Apply servo-controlled tilt (for Naomi SOL panels)."""
        # Constrain to realistic servo limits
        self.servo_angle_x = max(-15.0, min(15.0, angle_x))
        self.servo_angle_y = max(-15.0, min(15.0, angle_y))
        
        # Tilt affects optical reflectance
        tilt_magnitude = math.sqrt(angle_x**2 + angle_y**2)
        self.optical_reflectance = 0.95 * math.cos(math.radians(tilt_magnitude))
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "r": self.r,
            "c": self.c,
            "energy": self.energy,
            "residue": self.residue,
            "formation_time": self.formation_time,
            "stability": self.stability,
            "resonance_freq": self.resonance_freq,
            "optical_reflectance": self.optical_reflectance,
            "servo_angle_x": self.servo_angle_x,
            "servo_angle_y": self.servo_angle_y
        }


class ShapeLogicSimulator:
    """
    Advanced grid-based shape-logic simulator with polarity-based square formation.
    Integrates with Naomi SOL hardware concepts and physics validation.
    """
    
    def __init__(self, rows: int = 12, cols: int = 18, 
                 enable_physics: bool = True,
                 naomi_mode: bool = False):
        """
        Initialize the simulator with grid dimensions.
        
        Args:
            rows: Number of grid rows (default 12 for dodecahedron)
            cols: Number of grid columns  
            enable_physics: Enable physics calculations
            naomi_mode: Enable Naomi SOL specific features
        """
        self.rows = rows
        self.cols = cols
        self.enable_physics = enable_physics
        self.naomi_mode = naomi_mode
        
        # Horizontal edges: (rows+1) x cols
        self.H: List[List[EdgePolarity]] = [
            [EdgePolarity() for _ in range(cols)] for _ in range(rows + 1)
        ]
        
        # Vertical edges: rows x (cols+1)
        self.V: List[List[EdgePolarity]] = [
            [EdgePolarity() for _ in range(cols + 1)] for _ in range(rows)
        ]
        
        # Formed squares
        self.squares: Dict[Tuple[int, int], SquareBlock] = {}
        
        # Block residue tracking
        self.block_residue = [[0 for _ in range(cols - 1)] for _ in range(rows - 1)]
        
        # Formation rules
        self.square_formation_rule = {
            'top_bottom_polarity': -1,
            'left_right_polarity': +1,
            'energy_threshold': 0.5,
            'coherence_threshold': 0.7
        }
        
        # Random number generator
        self._rng = random.Random()
        
        # Simulation time
        self.sim_time = 0.0
        self.delta_time = 0.016  # 60 FPS
        
        # Statistics
        self.stats = {
            "total_energy": 0.0,
            "squares_formed": 0,
            "edges_active": 0,
            "average_stability": 0.0,
            "resonance_strength": 0.0
        }
        
        # Naomi SOL specific
        if self.naomi_mode:
            self._init_naomi_features()
        
        logger.info(f"ShapeLogic initialized: {rows}x{cols} grid, "
                   f"physics={'enabled' if enable_physics else 'disabled'}, "
                   f"naomi_mode={'yes' if naomi_mode else 'no'}")
    
    def _init_naomi_features(self):
        """Initialize Naomi SOL specific features."""
        # Pentagon panel mapping (12 panels for dodecahedron)
        self.panel_mapping = {}
        panel_id = 0
        
        # Map grid positions to dodecahedron panels
        for r in range(0, self.rows - 1, 3):
            for c in range(0, self.cols - 1, 3):
                if panel_id < 12:
                    self.panel_mapping[(r, c)] = panel_id
                    panel_id += 1
        
        # Laser interaction points
        self.laser_points = []
        
        # Crystal position (center of grid)
        self.crystal_position = (self.rows // 2, self.cols // 2)
        self.crystal_rotation = 0.0
        self.crystal_energy = 1.0
        
        logger.info(f"Naomi features initialized: {len(self.panel_mapping)} panels mapped")
    
    # ============== LOW-LEVEL EDGE OPERATIONS ==============
    
    def set_horizontal(self, r: int, c: int, polarity: int, energy: float = 1.0):
        """Set horizontal edge polarity and energy with bounds checking."""
        if 0 <= r <= self.rows and 0 <= c < self.cols:
            self.H[r][c].set(polarity, energy)
            logger.debug(f"Set H[{r}][{c}] = {polarity} (E={energy:.2f})")
        else:
            logger.warning(f"Invalid H edge: [{r}][{c}]")
    
    def set_vertical(self, r: int, c: int, polarity: int, energy: float = 1.0):
        """Set vertical edge polarity and energy with bounds checking."""
        if 0 <= r < self.rows and 0 <= c <= self.cols:
            self.V[r][c].set(polarity, energy)
            logger.debug(f"Set V[{r}][{c}] = {polarity} (E={energy:.2f})")
        else:
            logger.warning(f"Invalid V edge: [{r}][{c}]")
    
    def clear_horizontal(self, r: int, c: int):
        """Clear horizontal edge."""
        if 0 <= r <= self.rows and 0 <= c < self.cols:
            self.H[r][c].clear()
    
    def clear_vertical(self, r: int, c: int):
        """Clear vertical edge."""
        if 0 <= r < self.rows and 0 <= c <= self.cols:
            self.V[r][c].clear()
    
    def get_edge_at(self, r: int, c: int, edge_type: str) -> Optional[EdgePolarity]:
        """
        Get edge at position.
        
        Args:
            r: Row index
            c: Column index
            edge_type: 'H' for horizontal, 'V' for vertical
            
        Returns:
            EdgePolarity or None if out of bounds
        """
        if edge_type == 'H':
            if 0 <= r <= self.rows and 0 <= c < self.cols:
                return self.H[r][c]
        elif edge_type == 'V':
            if 0 <= r < self.rows and 0 <= c <= self.cols:
                return self.V[r][c]
        return None
    
    # ============== SQUARE FORMATION ==============
    
    def can_form_square(self, r: int, c: int) -> bool:
        """
        Check if a 2x2 square can form at position (r, c).
        Enhanced with energy and coherence requirements.
        """
        if r >= self.rows - 1 or c >= self.cols - 1:
            return False
        
        # Check if square already exists
        if (r, c) in self.squares:
            return False
        
        # Get the four edges
        top = self.H[r][c]
        bottom = self.H[r + 1][c]
        left = self.V[r][c]
        right = self.V[r][c + 1]
        
        # Check polarity pattern
        rule = self.square_formation_rule
        polarity_match = (
            top.polarity == rule['top_bottom_polarity'] and
            bottom.polarity == rule['top_bottom_polarity'] and
            left.polarity == rule['left_right_polarity'] and
            right.polarity == rule['left_right_polarity']
        )
        
        if not polarity_match:
            return False
        
        # Check energy threshold
        min_energy = min(top.energy, bottom.energy, left.energy, right.energy)
        if min_energy < rule['energy_threshold']:
            return False
        
        # Check coherence (optional)
        if self.enable_physics:
            avg_coherence = (top.coherence + bottom.coherence + 
                           left.coherence + right.coherence) / 4
            if avg_coherence < rule['coherence_threshold']:
                return False
        
        return True
    
    def form_square(self, r: int, c: int) -> bool:
        """
        Form a square at position (r, c).
        
        Returns:
            True if square was formed, False otherwise
        """
        if not self.can_form_square(r, c):
            return False
        
        # Calculate initial energy from edges
        top = self.H[r][c]
        bottom = self.H[r + 1][c]
        left = self.V[r][c]
        right = self.V[r][c + 1]
        
        total_energy = top.energy + bottom.energy + left.energy + right.energy
        
        # Create square
        square = SquareBlock(
            r=r, 
            c=c,
            energy=total_energy / 4,
            formation_time=self.sim_time
        )
        
        # If in Naomi mode, check if this is part of a panel
        if self.naomi_mode:
            panel_id = self._get_panel_id(r, c)
            if panel_id is not None:
                square.resonance_freq = 1.0 + panel_id * 0.1  # Unique frequency per panel
        
        self.squares[(r, c)] = square
        
        # Add residue to block
        self.block_residue[r][c] += 1
        
        # Reduce edge energy (energy transfer to square)
        energy_transfer = 0.2
        top.energy *= (1 - energy_transfer)
        bottom.energy *= (1 - energy_transfer)
        left.energy *= (1 - energy_transfer)
        right.energy *= (1 - energy_transfer)
        
        logger.debug(f"Square formed at ({r}, {c}) with energy {square.energy:.2f}")
        
        return True
    
    def dissolve_square(self, r: int, c: int) -> bool:
        """
        Dissolve a square at position (r, c).
        
        Returns:
            True if square was dissolved, False otherwise
        """
        if (r, c) not in self.squares:
            return False
        
        square = self.squares[(r, c)]
        
        # Return energy to edges
        energy_return = square.energy / 4
        
        if 0 <= r <= self.rows and 0 <= c < self.cols:
            self.H[r][c].energy += energy_return
        if 0 <= r + 1 <= self.rows and 0 <= c < self.cols:
            self.H[r + 1][c].energy += energy_return
        if 0 <= r < self.rows and 0 <= c <= self.cols:
            self.V[r][c].energy += energy_return
        if 0 <= r < self.rows and 0 <= c + 1 <= self.cols:
            self.V[r][c + 1].energy += energy_return
        
        # Remove square
        del self.squares[(r, c)]
        
        # Leave residue
        square.add_residue()
        self.block_residue[r][c] = square.residue
        
        logger.debug(f"Square dissolved at ({r}, {c})")
        
        return True
    
    # ============== SIMULATION ==============
    
    def randomize_edges(self, probability: float = 0.3, 
                       energy_range: Tuple[float, float] = (0.5, 2.0)):
        """
        Randomize edges with given probability.
        
        Args:
            probability: Chance of each edge being set
            energy_range: Min and max energy values
        """
        # Randomize horizontal edges
        for r in range(self.rows + 1):
            for c in range(self.cols):
                if self._rng.random() < probability:
                    polarity = self._rng.choice([-1, 1])
                    energy = self._rng.uniform(*energy_range)
                    self.H[r][c].set(polarity, energy)
        
        # Randomize vertical edges
        for r in range(self.rows):
            for c in range(self.cols + 1):
                if self._rng.random() < probability:
                    polarity = self._rng.choice([-1, 1])
                    energy = self._rng.uniform(*energy_range)
                    self.V[r][c].set(polarity, energy)
        
        logger.info(f"Edges randomized with p={probability}")
    
    def step(self, dt: Optional[float] = None):
        """
        Perform one simulation step.
        
        Args:
            dt: Time step (uses default if None)
        """
        dt = dt or self.delta_time
        self.sim_time += dt
        
        # Apply physics if enabled
        if self.enable_physics:
            self._apply_physics(dt)
        
        # Check for square formation/dissolution
        self._update_squares()
        
        # Apply Naomi SOL specific updates
        if self.naomi_mode:
            self._update_naomi_features(dt)
        
        # Update statistics
        self._update_stats()
    
    def _apply_physics(self, dt: float):
        """Apply physics calculations."""
        # Energy decay on edges
        decay_rate = 0.001
        
        for row in self.H:
            for edge in row:
                if edge.is_set():
                    edge.decay(decay_rate)
                    edge.resonate(1.0, dt)  # Base resonance
        
        for row in self.V:
            for edge in row:
                if edge.is_set():
                    edge.decay(decay_rate)
                    edge.resonate(1.2, dt)  # Slightly different frequency
        
        # Update square stability
        for square in self.squares.values():
            square.update_stability(dt)
            
            # Dissolve unstable squares
            if square.stability < 0.1:
                self.dissolve_square(square.r, square.c)
        
        # Energy diffusion between adjacent edges
        self._apply_energy_diffusion(dt)
    
    def _apply_energy_diffusion(self, dt: float):
        """Apply energy diffusion between adjacent edges."""
        diffusion_rate = 0.1
        
        # Create temporary energy storage
        h_energy_delta = [[0.0 for _ in range(self.cols)] 
                          for _ in range(self.rows + 1)]
        v_energy_delta = [[0.0 for _ in range(self.cols + 1)] 
                          for _ in range(self.rows)]
        
        # Calculate horizontal edge diffusion
        for r in range(self.rows + 1):
            for c in range(self.cols):
                if not self.H[r][c].is_set():
                    continue
                
                current_energy = self.H[r][c].energy
                
                # Diffuse to adjacent horizontal edges
                if c > 0 and self.H[r][c-1].is_set():
                    delta = (current_energy - self.H[r][c-1].energy) * diffusion_rate * dt
                    h_energy_delta[r][c] -= delta
                    h_energy_delta[r][c-1] += delta
                
                if c < self.cols - 1 and self.H[r][c+1].is_set():
                    delta = (current_energy - self.H[r][c+1].energy) * diffusion_rate * dt
                    h_energy_delta[r][c] -= delta
                    h_energy_delta[r][c+1] += delta
        
        # Apply energy changes
        for r in range(self.rows + 1):
            for c in range(self.cols):
                self.H[r][c].energy += h_energy_delta[r][c]
                self.H[r][c].energy = max(0.0, self.H[r][c].energy)
    
    def _update_squares(self):
        """Check for square formation and dissolution."""
        # Check for new squares
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                if (r, c) not in self.squares:
                    self.form_square(r, c)
        
        # Check for square dissolution (stability-based)
        to_dissolve = []
        for pos, square in self.squares.items():
            if square.stability < 0.1 or square.energy < 0.1:
                to_dissolve.append(pos)
        
        for pos in to_dissolve:
            self.dissolve_square(pos[0], pos[1])
    
    def _update_naomi_features(self, dt: float):
        """Update Naomi SOL specific features."""
        # Rotate crystal
        self.crystal_rotation += 60 * dt  # 60 degrees per second
        self.crystal_rotation = self.crystal_rotation % 360
        
        # Update panel tilts based on squares
        for (r, c), panel_id in self.panel_mapping.items():
            if (r, c) in self.squares:
                square = self.squares[(r, c)]
                
                # Simulate servo movement based on energy
                target_x = math.sin(self.sim_time + panel_id) * 10 * square.energy
                target_y = math.cos(self.sim_time + panel_id) * 10 * square.energy
                
                # Smooth servo movement
                square.servo_angle_x += (target_x - square.servo_angle_x) * dt * 2
                square.servo_angle_y += (target_y - square.servo_angle_y) * dt * 2
                
                square.apply_servo_tilt(square.servo_angle_x, square.servo_angle_y)
    
    def _get_panel_id(self, r: int, c: int) -> Optional[int]:
        """Get panel ID for a grid position."""
        # Find closest panel mapping
        for (pr, pc), panel_id in self.panel_mapping.items():
            if abs(pr - r) <= 2 and abs(pc - c) <= 2:
                return panel_id
        return None
    
    def _update_stats(self):
        """Update simulation statistics."""
        # Total energy
        total_energy = 0.0
        edges_active = 0
        
        for row in self.H:
            for edge in row:
                if edge.is_set():
                    total_energy += edge.energy
                    edges_active += 1
        
        for row in self.V:
            for edge in row:
                if edge.is_set():
                    total_energy += edge.energy
                    edges_active += 1
        
        # Square statistics
        squares_formed = len(self.squares)
        avg_stability = 0.0
        if squares_formed > 0:
            avg_stability = sum(s.stability for s in self.squares.values()) / squares_formed
        
        # Resonance strength (coherence measure)
        resonance_strength = 0.0
        if self.enable_physics and edges_active > 0:
            total_coherence = 0.0
            for row in self.H:
                for edge in row:
                    if edge.is_set():
                        total_coherence += edge.coherence
            for row in self.V:
                for edge in row:
                    if edge.is_set():
                        total_coherence += edge.coherence
            resonance_strength = total_coherence / edges_active
        
        self.stats = {
            "total_energy": total_energy,
            "squares_formed": squares_formed,
            "edges_active": edges_active,
            "average_stability": avg_stability,
            "resonance_strength": resonance_strength,
            "sim_time": self.sim_time
        }
    
    # ============== EXPORT/IMPORT ==============
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export complete simulator state.
        
        Returns:
            Dictionary containing all state information
        """
        state = {
            "version": "3.0",
            "rows": self.rows,
            "cols": self.cols,
            "sim_time": self.sim_time,
            "enable_physics": self.enable_physics,
            "naomi_mode": self.naomi_mode,
            "horizontal_edges": [
                [edge.to_dict() for edge in row]
                for row in self.H
            ],
            "vertical_edges": [
                [edge.to_dict() for edge in row]
                for row in self.V
            ],
            "squares": {
                f"{r},{c}": square.to_dict()
                for (r, c), square in self.squares.items()
            },
            "block_residue": self.block_residue,
            "stats": self.stats
        }
        
        if self.naomi_mode:
            state["naomi"] = {
                "panel_mapping": {f"{r},{c}": pid for (r, c), pid in self.panel_mapping.items()},
                "crystal_position": self.crystal_position,
                "crystal_rotation": self.crystal_rotation,
                "crystal_energy": self.crystal_energy
            }
        
        return state
    
    def import_state(self, state: Dict[str, Any]):
        """
        Import simulator state.
        
        Args:
            state: Dictionary containing state information
        """
        try:
            # Basic properties
            self.rows = state["rows"]
            self.cols = state["cols"]
            self.sim_time = state.get("sim_time", 0.0)
            self.enable_physics = state.get("enable_physics", True)
            self.naomi_mode = state.get("naomi_mode", False)
            
            # Edges
            self.H = [
                [EdgePolarity.from_dict(edge_data) for edge_data in row]
                for row in state["horizontal_edges"]
            ]
            
            self.V = [
                [EdgePolarity.from_dict(edge_data) for edge_data in row]
                for row in state["vertical_edges"]
            ]
            
            # Squares
            self.squares = {}
            for pos_str, square_data in state.get("squares", {}).items():
                r, c = map(int, pos_str.split(","))
                square = SquareBlock(**square_data)
                self.squares[(r, c)] = square
            
            # Residue
            self.block_residue = state.get("block_residue", self.block_residue)
            
            # Stats
            self.stats = state.get("stats", self.stats)
            
            # Naomi features
            if self.naomi_mode and "naomi" in state:
                naomi_data = state["naomi"]
                self.panel_mapping = {
                    tuple(map(int, k.split(","))): v
                    for k, v in naomi_data["panel_mapping"].items()
                }
                self.crystal_position = tuple(naomi_data["crystal_position"])
                self.crystal_rotation = naomi_data["crystal_rotation"]
                self.crystal_energy = naomi_data["crystal_energy"]
            
            logger.info("State imported successfully")
            
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            raise
    
    def save_to_file(self, filepath: str):
        """Save state to JSON file."""
        import json
        state = self.export_state()
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"State saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load state from JSON file."""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.import_state(state)
        logger.info(f"State loaded from {filepath}")
    
    def get_grid_visual(self) -> str:
        """
        Get ASCII visualization of the grid.
        
        Returns:
            String representation of the grid
        """
        lines = []
        
        for r in range(self.rows):
            # Horizontal edges row
            h_row = ""
            for c in range(self.cols):
                h_row += "+"
                edge = self.H[r][c]
                if edge.polarity == 1:
                    h_row += "—→"
                elif edge.polarity == -1:
                    h_row += "←—"
                else:
                    h_row += "   "
            h_row += "+"
            lines.append(h_row)
            
            # Vertical edges row
            v_row = ""
            for c in range(self.cols):
                edge = self.V[r][c]
                if edge.polarity == 1:
                    v_row += "↓"
                elif edge.polarity == -1:
                    v_row += "↑"
                else:
                    v_row += "|"
                
                # Square indicator
                if (r, c) in self.squares:
                    v_row += " ■ "
                else:
                    v_row += "   "
            
            # Last vertical edge
            edge = self.V[r][self.cols]
            if edge.polarity == 1:
                v_row += "↓"
            elif edge.polarity == -1:
                v_row += "↑"
            else:
                v_row += "|"
            
            lines.append(v_row)
        
        # Last horizontal row
        h_row = ""
        for c in range(self.cols):
            h_row += "+"
            edge = self.H[self.rows][c]
            if edge.polarity == 1:
                h_row += "—→"
            elif edge.polarity == -1:
                h_row += "←—"
            else:
                h_row += "   "
        h_row += "+"
        lines.append(h_row)
        
        return "\n".join(lines)


def test_shape_logic():
    """Test the shape logic simulator."""
    print("Testing Shape Logic Simulator...")
    
    # Create simulator
    sim = ShapeLogicSimulator(rows=6, cols=8, naomi_mode=True)
    
    # Set up a test pattern
    sim.set_horizontal(0, 0, -1, 2.0)
    sim.set_horizontal(1, 0, -1, 2.0)
    sim.set_vertical(0, 0, 1, 2.0)
    sim.set_vertical(0, 1, 1, 2.0)
    
    # Run simulation
    for i in range(10):
        sim.step()
        print(f"\nStep {i+1}:")
        print(sim.get_grid_visual())
        print(f"Stats: {sim.stats}")
    
    # Test save/load
    sim.save_to_file("test_state.json")
    sim2 = ShapeLogicSimulator(rows=6, cols=8)
    sim2.load_from_file("test_state.json")
    
    print("\nLoaded state:")
    print(sim2.get_grid_visual())
    
    print("\nTest complete!")


if __name__ == "__main__":
    test_shape_logic()
