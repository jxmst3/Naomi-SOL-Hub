# naomi/cad_interface.py
"""
CAD Interface for Naomi SOL
============================
Generates parametric 3D models for all Naomi SOL components.
Supports STL export for 3D printing and STEP for CAD editing.
"""

import math
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

# Try to import CAD libraries
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    print("WARNING: CadQuery not installed - using basic STL generation")

try:
    from stl import mesh as stl_mesh
    import numpy as np
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    print("WARNING: numpy-stl not installed - STL export limited")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

logger = logging.getLogger("CAD_Interface")


class NaomiSOLPart:
    """
    Parametric part generator for Naomi SOL components.
    Creates 3D models based on design parameters.
    """
    
    def __init__(self, part_name: str, params: Dict):
        """
        Initialize part generator.
        
        Args:
            part_name: Name of the part to generate
            params: Design parameters dictionary
        """
        self.part_name = part_name
        self.params = params
        self.model = None
        self.mesh = None
        
        # Extract common parameters
        self.side_length = params.get('side_length', 150)  # Pentagon side in mm
        self.thickness = params.get('thickness', 4)  # Panel thickness
        self.servo_pocket_depth = params.get('servo_pocket_depth', 24)
        self.mirror_diameter = params.get('mirror_diameter', 70)
        self.infill = params.get('infill_percentage', 30)
        
        logger.info(f"Initialized CAD generator for {part_name}")
    
    def generate_cad(self) -> bool:
        """
        Generate the CAD model.
        
        Returns:
            True if successful
        """
        try:
            if CADQUERY_AVAILABLE:
                return self._generate_cadquery()
            else:
                return self._generate_basic_stl()
        except Exception as e:
            logger.error(f"CAD generation failed: {e}")
            return False
    
    def _generate_cadquery(self) -> bool:
        """Generate using CadQuery (professional CAD)"""
        try:
            if "Pentagon" in self.part_name:
                self.model = self._create_pentagon_panel()
            elif "Central_Hub" in self.part_name:
                self.model = self._create_central_hub()
            elif "Mirror_Platform" in self.part_name:
                self.model = self._create_mirror_platform()
            elif "Servo_Mounting_Bracket" in self.part_name:
                self.model = self._create_servo_bracket()
            elif "Connecting_Rod" in self.part_name:
                self.model = self._create_connecting_rod()
            elif "Ball_Joint" in self.part_name:
                self.model = self._create_ball_joint()
            else:
                logger.warning(f"Unknown part: {self.part_name}")
                return False
            
            logger.info(f"Generated CadQuery model for {self.part_name}")
            return True
            
        except Exception as e:
            logger.error(f"CadQuery generation error: {e}")
            return False
    
    def _create_pentagon_panel(self):
        """Create pentagon panel with servo mounts"""
        # Pentagon vertices
        n = 5
        vertices = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2  # Start from top
            x = self.side_length * math.cos(angle) / (2 * math.sin(math.pi / n))
            y = self.side_length * math.sin(angle) / (2 * math.sin(math.pi / n))
            vertices.append((x, y))
        
        # Create base panel
        panel = (cq.Workplane("XY")
                .polyline(vertices)
                .close()
                .extrude(self.thickness))
        
        # Add bevel for dodecahedron assembly (58.283°)
        bevel_angle = 58.283
        panel = panel.edges(">Z").chamfer(self.thickness * 0.3)
        
        # Add central hub mount
        hub_diameter = 80
        panel = (panel.faces(">Z")
                .workplane()
                .circle(hub_diameter / 2)
                .extrude(5))
        
        # Add servo mounting points (120° apart)
        servo_positions = []
        servo_mount_radius = 45
        for i in range(3):
            angle = i * 120 * math.pi / 180
            x = servo_mount_radius * math.cos(angle)
            y = servo_mount_radius * math.sin(angle)
            servo_positions.append((x, y))
        
        # Create servo pockets
        for pos in servo_positions:
            panel = (panel.faces(">Z")
                    .workplane()
                    .center(pos[0], pos[1])
                    .rect(23, 12.5)  # MG90S dimensions
                    .cutBlind(-self.servo_pocket_depth))
            
            # Add screw holes
            panel = (panel.faces(">Z")
                    .workplane()
                    .center(pos[0] - 13.75, pos[1])
                    .circle(1.5)
                    .cutThruAll())
            
            panel = (panel.faces(">Z")
                    .workplane()
                    .center(pos[0] + 13.75, pos[1])
                    .circle(1.5)
                    .cutThruAll())
        
        # Add sensor mounting points
        sensor_positions = [
            (self.side_length * 0.3, 0),
            (-self.side_length * 0.3, 0),
            (0, self.side_length * 0.3),
            (0, -self.side_length * 0.3)
        ]
        
        for pos in sensor_positions:
            panel = (panel.faces(">Z")
                    .workplane()
                    .center(pos[0], pos[1])
                    .circle(2)
                    .cutBlind(-3))
        
        # Add wire channels
        for i in range(3):
            angle = (i * 120 + 60) * math.pi / 180
            x1 = 20 * math.cos(angle)
            y1 = 20 * math.sin(angle)
            x2 = 60 * math.cos(angle)
            y2 = 60 * math.sin(angle)
            
            panel = (panel.faces(">Z")
                    .workplane()
                    .moveTo(x1, y1)
                    .lineTo(x2, y2)
                    .rect(3, 2)
                    .cutBlind(-2))
        
        # Special handling for laser panel
        if "Laser" in self.part_name:
            # Add laser entry ports
            laser_positions = [(25, 0), (-25, 0)]
            for pos in laser_positions:
                panel = (panel.faces(">Z")
                        .workplane()
                        .center(pos[0], pos[1])
                        .circle(6)  # 12mm diameter for laser module
                        .cutThruAll())
        
        return panel
    
    def _create_central_hub(self):
        """Create central hub for servo mechanism"""
        # Base cylinder
        hub = (cq.Workplane("XY")
              .circle(40)
              .extrude(10))
        
        # Add center hole for shaft
        hub = (hub.faces(">Z")
              .workplane()
              .circle(3)
              .cutThruAll())
        
        # Add ball joint mounts (3x at 120°)
        for i in range(3):
            angle = i * 120 * math.pi / 180
            x = 25 * math.cos(angle)
            y = 25 * math.sin(angle)
            
            # Mount post
            hub = (hub.faces(">Z")
                  .workplane()
                  .center(x, y)
                  .circle(4)
                  .extrude(8))
            
            # Ball socket
            hub = (hub.faces(">Z")
                  .workplane()
                  .center(x, y)
                  .sphere(4)
                  .intersect(hub))
        
        return hub
    
    def _create_mirror_platform(self):
        """Create tilting mirror platform"""
        # Pentagon mirror shape
        n = 5
        vertices = []
        mirror_radius = self.mirror_diameter / 2
        
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = mirror_radius * math.cos(angle)
            y = mirror_radius * math.sin(angle)
            vertices.append((x, y))
        
        # Base platform
        platform = (cq.Workplane("XY")
                   .polyline(vertices)
                   .close()
                   .extrude(3))
        
        # Add mirror recess
        platform = (platform.faces(">Z")
                   .workplane()
                   .polyline([(v[0]*0.9, v[1]*0.9) for v in vertices])
                   .close()
                   .cutBlind(-1.5))
        
        # Add ball joint connection points
        for i in range(3):
            angle = (i * 120 + 60) * math.pi / 180
            x = 20 * math.cos(angle)
            y = 20 * math.sin(angle)
            
            platform = (platform.faces("<Z")
                       .workplane()
                       .center(x, y)
                       .circle(3)
                       .extrude(-5))
        
        return platform
    
    def _create_servo_bracket(self):
        """Create servo mounting bracket"""
        # L-shaped bracket
        bracket = (cq.Workplane("XY")
                  .rect(30, 20)
                  .extrude(3))
        
        # Vertical part
        bracket = (bracket.faces(">Y")
                  .workplane()
                  .rect(30, 25)
                  .extrude(3))
        
        # Servo mounting holes
        bracket = (bracket.faces(">Z")
                  .workplane()
                  .center(-13.75, 0)
                  .circle(1.5)
                  .center(27.5, 0)  # Move to other hole
                  .circle(1.5)
                  .cutThruAll())
        
        # Panel mounting holes
        bracket = (bracket.faces("<X")
                  .workplane()
                  .center(0, -8)
                  .circle(2)
                  .center(0, 16)
                  .circle(2)
                  .cutThruAll())
        
        return bracket
    
    def _create_connecting_rod(self):
        """Create connecting rod between servo and platform"""
        # Rod body
        rod_length = 35
        rod = (cq.Workplane("XY")
              .circle(2)
              .extrude(rod_length))
        
        # Ball ends
        rod = (rod.faces(">Z")
              .workplane()
              .sphere(4))
        
        rod = (rod.faces("<Z")
              .workplane()
              .sphere(4))
        
        # Hollow center for weight reduction
        rod = (rod.faces(">Z")
              .workplane()
              .circle(1)
              .cutBlind(-rod_length + 8))
        
        return rod
    
    def _create_ball_joint(self):
        """Create ball joint connector"""
        # Socket base
        joint = (cq.Workplane("XY")
                .circle(6)
                .extrude(8))
        
        # Socket cavity
        joint = (joint.faces(">Z")
                .workplane()
                .sphere(4.2)
                .cut(joint))
        
        # Opening for ball insertion
        joint = (joint.faces(">Z")
                .workplane()
                .circle(3.5)
                .cutBlind(-4))
        
        # Mounting hole
        joint = (joint.faces("<Z")
                .workplane()
                .circle(2)
                .cutThruAll())
        
        return joint
    
    def _generate_basic_stl(self) -> bool:
        """Generate basic STL without CadQuery"""
        if not STL_AVAILABLE:
            logger.error("numpy-stl not available")
            return False
        
        try:
            if "Pentagon" in self.part_name:
                self.mesh = self._create_pentagon_mesh()
            else:
                # Create simple cube as placeholder
                self.mesh = self._create_cube_mesh()
            
            logger.info(f"Generated basic STL mesh for {self.part_name}")
            return True
            
        except Exception as e:
            logger.error(f"Basic STL generation error: {e}")
            return False
    
    def _create_pentagon_mesh(self):
        """Create pentagon mesh using numpy-stl"""
        n = 5
        vertices = []
        
        # Pentagon vertices (top face)
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = self.side_length * math.cos(angle) / (2 * math.sin(math.pi / n))
            y = self.side_length * math.sin(angle) / (2 * math.sin(math.pi / n))
            vertices.append([x, y, self.thickness])
        
        # Pentagon vertices (bottom face)
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = self.side_length * math.cos(angle) / (2 * math.sin(math.pi / n))
            y = self.side_length * math.sin(angle) / (2 * math.sin(math.pi / n))
            vertices.append([x, y, 0])
        
        vertices = np.array(vertices)
        
        # Create faces
        faces = []
        
        # Top face
        for i in range(n - 2):
            faces.append([0, i + 1, i + 2])
        
        # Bottom face
        for i in range(n - 2):
            faces.append([n, n + i + 2, n + i + 1])
        
        # Side faces
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])
        
        # Create mesh
        pentagon = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        
        for i, face in enumerate(faces):
            for j in range(3):
                pentagon.vectors[i][j] = vertices[face[j]]
        
        return pentagon
    
    def _create_cube_mesh(self):
        """Create simple cube mesh as placeholder"""
        vertices = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
            [0, 0, 10],
            [10, 0, 10],
            [10, 10, 10],
            [0, 10, 10]
        ])
        
        faces = np.array([
            [0, 3, 1], [1, 3, 2],  # Bottom
            [0, 1, 4], [1, 5, 4],  # Front
            [1, 2, 5], [2, 6, 5],  # Right
            [2, 3, 6], [3, 7, 6],  # Back
            [3, 0, 7], [0, 4, 7],  # Left
            [4, 5, 7], [5, 6, 7]   # Top
        ])
        
        cube = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        
        for i, face in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[face[j]]
        
        return cube
    
    def export(self, filepath: str, format: str = "auto"):
        """
        Export the model to file.
        
        Args:
            filepath: Output file path
            format: File format (stl, step, or auto)
        """
        filepath = Path(filepath)
        
        # Auto-detect format from extension
        if format == "auto":
            if filepath.suffix.lower() == ".stl":
                format = "stl"
            elif filepath.suffix.lower() in [".step", ".stp"]:
                format = "step"
            else:
                format = "stl"
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "stl":
                self._export_stl(filepath)
            elif format == "step":
                self._export_step(filepath)
            else:
                logger.error(f"Unknown format: {format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    def _export_stl(self, filepath: Path):
        """Export as STL"""
        if CADQUERY_AVAILABLE and self.model:
            # Export CadQuery model
            cq.exporters.export(self.model, str(filepath))
            logger.info(f"Exported CadQuery STL to {filepath}")
            
        elif self.mesh:
            # Export numpy-stl mesh
            self.mesh.save(str(filepath))
            logger.info(f"Exported STL mesh to {filepath}")
            
        else:
            # Create basic STL file
            self._write_basic_stl(filepath)
    
    def _export_step(self, filepath: Path):
        """Export as STEP"""
        if CADQUERY_AVAILABLE and self.model:
            cq.exporters.export(self.model, str(filepath))
            logger.info(f"Exported STEP to {filepath}")
        else:
            logger.warning("STEP export requires CadQuery")
    
    def _write_basic_stl(self, filepath: Path):
        """Write basic ASCII STL file"""
        with open(filepath, 'w') as f:
            f.write(f"solid {self.part_name}\n")
            
            # Write a simple triangle as placeholder
            f.write("  facet normal 0 0 1\n")
            f.write("    outer loop\n")
            f.write("      vertex 0 0 0\n")
            f.write("      vertex 10 0 0\n")
            f.write("      vertex 5 10 0\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
            
            f.write(f"endsolid {self.part_name}\n")
        
        logger.info(f"Wrote basic STL to {filepath}")
    
    def get_print_time_estimate(self) -> float:
        """
        Estimate 3D print time in hours.
        
        Returns:
            Estimated print time
        """
        # Simple estimation based on volume and parameters
        if CADQUERY_AVAILABLE and self.model:
            # Get bounding box
            bb = self.model.val().BoundingBox()
            volume = (bb.xmax - bb.xmin) * (bb.ymax - bb.ymin) * (bb.zmax - bb.zmin)
        else:
            # Rough estimate for pentagon
            volume = self.side_length * self.side_length * self.thickness
        
        # Print time factors
        layer_height = self.params.get('layer_height', 0.2)
        infill = self.infill / 100.0
        print_speed = 50  # mm/s typical
        
        # Approximate layer count
        layers = self.thickness / layer_height
        
        # Approximate print time (very rough)
        # Time = volume * infill / (speed * layer_height * extrusion_width)
        time_hours = (volume * infill) / (print_speed * layer_height * 0.4 * 3600)
        
        # Add overhead for moves, retractions, etc
        time_hours *= 1.3
        
        return time_hours
    
    def get_material_usage(self) -> float:
        """
        Estimate material usage in grams.
        
        Returns:
            Estimated material weight
        """
        # Get volume estimate
        if CADQUERY_AVAILABLE and self.model:
            bb = self.model.val().BoundingBox()
            volume = (bb.xmax - bb.xmin) * (bb.ymax - bb.ymin) * (bb.zmax - bb.zmin)
        else:
            volume = self.side_length * self.side_length * self.thickness
        
        # Material density (PETG)
        density = 1.27  # g/cm³
        
        # Convert mm³ to cm³
        volume_cm3 = volume / 1000
        
        # Weight with infill
        weight = volume_cm3 * density * (self.infill / 100.0)
        
        # Add support material estimate
        weight *= 1.1
        
        return weight


def generate_all_parts(params: Dict, output_dir: str = "output/cad_models"):
    """
    Generate all Naomi SOL parts.
    
    Args:
        params: Design parameters
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parts to generate
    parts = [
        ("Pentagon_Base_Panel", 11),  # 11 standard panels
        ("Pentagon_Laser_Panel", 1),  # 1 panel with laser ports
        ("Central_Hub", 12),
        ("Mirror_Platform", 12),
        ("Servo_Mounting_Bracket", 36),
        ("Connecting_Rod", 36),
        ("Ball_Joint_Connector", 72)
    ]
    
    total_time = 0
    total_material = 0
    
    print("\n" + "="*60)
    print("GENERATING NAOMI SOL CAD MODELS")
    print("="*60)
    
    for part_name, quantity in parts:
        print(f"\n{part_name} (×{quantity})...")
        
        # Generate part
        part = NaomiSOLPart(part_name, params)
        
        if part.generate_cad():
            # Export STL
            stl_file = output_path / f"{part_name}.stl"
            part.export(str(stl_file))
            
            # Export STEP if available
            if CADQUERY_AVAILABLE:
                step_file = output_path / f"{part_name}.step"
                part.export(str(step_file), format="step")
            
            # Calculate estimates
            time_est = part.get_print_time_estimate()
            material_est = part.get_material_usage()
            
            print(f"  ✓ Generated: {stl_file.name}")
            print(f"    Print time (each): {time_est:.1f} hours")
            print(f"    Material (each): {material_est:.1f}g")
            print(f"    Total for {quantity}: {time_est*quantity:.1f}h, {material_est*quantity:.0f}g")
            
            total_time += time_est * quantity
            total_material += material_est * quantity
        else:
            print(f"  ✗ Failed to generate {part_name}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total print time: {total_time:.1f} hours ({total_time/24:.1f} days)")
    print(f"Total material: {total_material:.0f}g ({total_material/1000:.1f}kg)")
    print(f"Files saved to: {output_path.absolute()}")


def test_cad():
    """Test CAD generation"""
    print("Testing CAD Interface...")
    
    # Test parameters
    params = {
        'side_length': 150,
        'thickness': 4,
        'servo_pocket_depth': 24,
        'mirror_diameter': 70,
        'infill_percentage': 30,
        'layer_height': 0.2
    }
    
    # Test single part
    print("\nGenerating test pentagon panel...")
    part = NaomiSOLPart("Pentagon_Base_Panel", params)
    
    if part.generate_cad():
        part.export("test_panel.stl")
        print(f"✓ Generated test_panel.stl")
        print(f"  Estimated print time: {part.get_print_time_estimate():.1f} hours")
        print(f"  Estimated material: {part.get_material_usage():.1f}g")
    else:
        print("✗ Generation failed")
    
    print("\nCAD test complete!")


if __name__ == "__main__":
    # Test single part
    test_cad()
    
    # Generate all parts
    print("\n" + "="*80)
    response = input("Generate all Naomi SOL parts? (y/n): ")
    if response.lower() == 'y':
        params = {
            'side_length': 150,
            'thickness': 4,
            'servo_pocket_depth': 24,
            'mirror_diameter': 70,
            'infill_percentage': 30,
            'layer_height': 0.2
        }
        generate_all_parts(params)
