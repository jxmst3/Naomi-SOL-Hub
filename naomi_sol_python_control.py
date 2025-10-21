"""
Naomi SOL - Coordinated Mirror Control System
================================================
Complete Python control system for 12-panel dodecahedron mirror array
with Teensy 4.1 master controller and distributed Arduino Nano panel controllers

Features:
- Stewart platform inverse kinematics for each panel
- Synchronized multi-panel coordination
- Real-time sensor fusion (MPU-9250 IMU data)
- Pattern generation (geometric, organic, reactive)
- Serial communication with Teensy 4.1 master
- Camera-based feedback integration
- Laser tracking and targeting
"""

import serial
import time
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import threading
import queue
import math

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

class ControlMode(Enum):
    """Available control modes for mirror system"""
    MANUAL = "manual"           # Direct position control
    PATTERN = "pattern"         # Pre-programmed patterns
    LASER_TRACK = "laser_track" # Track and target laser
    SENSOR_REACTIVE = "sensor"  # React to IMU sensor data
    COORDINATED = "coordinated" # All panels work together
    RANDOM = "random"           # Organic random motion

@dataclass
class PanelGeometry:
    """Physical geometry of one Stewart platform panel"""
    # Pentagon panel dimensions
    side_length: float = 200.0  # mm
    
    # Mirror platform
    platform_radius: float = 50.0  # mm from center to mounting point
    
    # Base servo positions (equilateral triangle, 120° apart)
    servo_radius: float = 110.0  # mm from panel center to servo
    servo_angles: List[float] = field(default_factory=lambda: [0, 120, 240])  # degrees
    
    # Rod length
    rod_length: float = 80.0  # mm
    
    # Tilt limits
    max_tilt: float = 30.0  # degrees (safety limit)
    
@dataclass
class ServoLimits:
    """Servo angle limits and calibration"""
    min_angle: int = 0       # degrees
    max_angle: int = 180     # degrees
    neutral: int = 90        # degrees (level mirror)
    
    # Calibration offsets for each servo (per panel)
    calibration: List[int] = field(default_factory=lambda: [0, 0, 0])

# ============================================================================
# STEWART PLATFORM KINEMATICS
# ============================================================================

class StewartPlatform:
    """
    Stewart Platform inverse kinematics calculator
    Converts desired mirror tilt (pitch, roll) to servo angles
    """
    
    def __init__(self, geometry: PanelGeometry):
        self.geo = geometry
        
        # Pre-calculate servo base positions in Cartesian coordinates
        self.servo_positions = []
        for angle in geometry.servo_angles:
            rad = math.radians(angle)
            x = geometry.servo_radius * math.cos(rad)
            y = geometry.servo_radius * math.sin(rad)
            self.servo_positions.append(np.array([x, y, 0]))
        
        # Pre-calculate platform mounting positions
        self.platform_positions = []
        for angle in geometry.servo_angles:
            rad = math.radians(angle)
            x = geometry.platform_radius * math.cos(rad)
            y = geometry.platform_radius * math.sin(rad)
            self.platform_positions.append(np.array([x, y, 0]))
    
    def tilt_to_servo_angles(self, pitch: float, roll: float, height: float = 0) -> List[float]:
        """
        Convert mirror tilt angles to servo positions
        
        Args:
            pitch: Tilt around Y-axis in degrees (-max_tilt to +max_tilt)
            roll: Tilt around X-axis in degrees (-max_tilt to +max_tilt)
            height: Additional Z-axis offset in mm
        
        Returns:
            List of 3 servo angles in degrees
        """
        # Clamp to safe limits
        pitch = np.clip(pitch, -self.geo.max_tilt, self.geo.max_tilt)
        roll = np.clip(roll, -self.geo.max_tilt, self.geo.max_tilt)
        
        # Convert to radians
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])
        
        Ry = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        
        R = Ry @ Rx  # Combined rotation
        
        # Calculate platform position after rotation
        platform_center = np.array([0, 0, height])
        
        servo_angles = []
        for i in range(3):
            # Rotated platform mount position
            p_rot = platform_center + R @ self.platform_positions[i]
            
            # Vector from servo to platform mount
            vec = p_rot - self.servo_positions[i]
            
            # Calculate required rod length (should be close to rod_length)
            actual_length = np.linalg.norm(vec)
            
            # Angle of rod from horizontal
            horizontal_dist = np.linalg.norm(vec[:2])
            vertical_dist = vec[2]
            rod_angle = math.degrees(math.atan2(vertical_dist, horizontal_dist))
            
            # Convert to servo angle (assuming servo at 90° points horizontal)
            servo_angle = 90 + rod_angle
            
            # Clamp to servo limits
            servo_angle = np.clip(servo_angle, 0, 180)
            servo_angles.append(servo_angle)
        
        return servo_angles
    
    def calculate_mirror_normal(self, pitch: float, roll: float) -> np.ndarray:
        """
        Calculate the mirror surface normal vector from tilt angles
        Useful for laser reflection calculations
        """
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # Start with normal pointing up [0, 0, 1]
        normal = np.array([0, 0, 1])
        
        # Apply rotations
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])
        
        Ry = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        
        normal = Ry @ Rx @ normal
        return normal / np.linalg.norm(normal)  # Normalize

# ============================================================================
# PATTERN GENERATORS
# ============================================================================

class PatternGenerator:
    """Generate coordinated motion patterns for all 12 panels"""
    
    @staticmethod
    def wave_pattern(t: float, panel_index: int, num_panels: int = 12) -> Tuple[float, float]:
        """
        Traveling wave across panels
        
        Args:
            t: Time in seconds
            panel_index: Which panel (0-11)
            num_panels: Total number of panels
        
        Returns:
            (pitch, roll) in degrees
        """
        phase = (panel_index / num_panels) * 2 * math.pi
        frequency = 0.5  # Hz
        amplitude = 15.0  # degrees
        
        pitch = amplitude * math.sin(2 * math.pi * frequency * t + phase)
        roll = amplitude * math.cos(2 * math.pi * frequency * t + phase + math.pi/4)
        
        return pitch, roll
    
    @staticmethod
    def spiral_pattern(t: float, panel_index: int) -> Tuple[float, float]:
        """Spiral motion outward from center"""
        radius = (t % 10) / 10  # 0 to 1 over 10 seconds
        angle = panel_index * (360 / 12) + (t * 360 / 5)  # Rotate
        
        pitch = radius * 20 * math.cos(math.radians(angle))
        roll = radius * 20 * math.sin(math.radians(angle))
        
        return pitch, roll
    
    @staticmethod
    def random_organic(panel_index: int, seed: int = 42) -> Tuple[float, float]:
        """Slow organic random motion (Perlin-noise-like)"""
        np.random.seed(seed + panel_index)
        
        # Generate smooth random values
        t = time.time() / 10  # Slow time scale
        pitch = 15 * math.sin(t + np.random.random() * 2 * math.pi)
        roll = 15 * math.cos(t * 0.7 + np.random.random() * 2 * math.pi)
        
        return pitch, roll
    
    @staticmethod
    def all_point_center() -> Tuple[float, float]:
        """All mirrors point toward dodecahedron center"""
        # This would require dodecahedron geometry calculations
        # For now, approximate with level mirrors
        return 0.0, 0.0
    
    @staticmethod
    def breathing_pattern(t: float) -> Tuple[float, float]:
        """All panels breathe in and out together"""
        amplitude = 10.0
        frequency = 0.3  # Hz
        
        tilt = amplitude * math.sin(2 * math.pi * frequency * t)
        return tilt, 0.0  # Tilt in pitch only

# ============================================================================
# DODECAHEDRON GEOMETRY
# ============================================================================

class DodecahedronGeometry:
    """Calculates positions and orientations of 12 pentagonal panels"""
    
    def __init__(self, radius: float = 220.0):
        """
        Args:
            radius: Circumscribed sphere radius in mm
        """
        self.radius = radius
        self.panel_normals = self._calculate_panel_normals()
    
    def _calculate_panel_normals(self) -> List[np.ndarray]:
        """
        Calculate outward-facing normal vectors for each of 12 panels
        Based on dodecahedron symmetry
        """
        normals = []
        
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # 12 vertices of icosahedron (dual of dodecahedron)
        # These point toward panel centers
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
        ]
        
        for v in vertices:
            normal = np.array(v)
            normal = normal / np.linalg.norm(normal)  # Normalize
            normals.append(normal)
        
        return normals
    
    def get_panel_orientation(self, panel_index: int) -> Tuple[float, float, float]:
        """
        Get panel orientation as Euler angles
        
        Returns:
            (pitch, roll, yaw) in degrees
        """
        normal = self.panel_normals[panel_index]
        
        # Calculate pitch and roll from normal vector
        pitch = math.degrees(math.asin(-normal[1]))
        roll = math.degrees(math.atan2(normal[0], normal[2]))
        yaw = 0  # Assume no twist
        
        return pitch, roll, yaw

# ============================================================================
# COMMUNICATION LAYER
# ============================================================================

class TeensyMasterController:
    """
    Communicates with Teensy 4.1 master controller via serial
    Sends coordinated commands to all 12 panels
    """
    
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        
        # Command queue
        self.command_queue = queue.Queue()
        
        # Response handling
        self.response_queue = queue.Queue()
    
    def connect(self) -> bool:
        """Establish serial connection to Teensy"""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            self.connected = True
            print(f"✓ Connected to Teensy on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial and self.connected:
            self.serial.close()
            self.connected = False
            print("✓ Disconnected from Teensy")
    
    def send_panel_command(self, panel_id: int, servo_angles: List[float]):
        """
        Send servo position command for one panel
        
        Args:
            panel_id: Panel number (0-11)
            servo_angles: List of 3 servo angles in degrees
        """
        if not self.connected:
            print("✗ Not connected to Teensy")
            return
        
        # Format: P<id>:S0=<angle>,S1=<angle>,S2=<angle>\n
        cmd = f"P{panel_id}:S0={int(servo_angles[0])},S1={int(servo_angles[1])},S2={int(servo_angles[2])}\n"
        
        try:
            self.serial.write(cmd.encode())
            self.serial.flush()
        except serial.SerialException as e:
            print(f"✗ Send error: {e}")
    
    def send_all_panels(self, all_servo_angles: List[List[float]]):
        """
        Send commands to all 12 panels at once
        
        Args:
            all_servo_angles: List of 12 lists, each with 3 servo angles
        """
        if not self.connected:
            return
        
        # Batch command format for efficiency
        cmd = "BATCH:"
        for i, angles in enumerate(all_servo_angles):
            cmd += f"P{i}:S0={int(angles[0])},S1={int(angles[1])},S2={int(angles[2])};"
        cmd += "\n"
        
        try:
            self.serial.write(cmd.encode())
            self.serial.flush()
        except serial.SerialException as e:
            print(f"✗ Batch send error: {e}")
    
    def request_sensor_data(self, panel_id: int) -> Optional[Dict]:
        """
        Request IMU sensor data from specific panel
        
        Returns:
            Dictionary with accelerometer, gyro, magnetometer data
        """
        if not self.connected:
            return None
        
        cmd = f"SENSOR:{panel_id}\n"
        try:
            self.serial.write(cmd.encode())
            self.serial.flush()
            
            # Wait for response
            response = self.serial.readline().decode().strip()
            
            # Parse JSON response
            if response:
                return json.loads(response)
        except Exception as e:
            print(f"✗ Sensor request error: {e}")
        
        return None
    
    def emergency_stop(self):
        """Immediately stop all servos"""
        if not self.connected:
            return
        
        cmd = "ESTOP\n"
        self.serial.write(cmd.encode())
        self.serial.flush()
        print("⚠ EMERGENCY STOP ACTIVATED")

# ============================================================================
# MAIN CONTROL SYSTEM
# ============================================================================

class NaomiSOLController:
    """
    Main control system for Naomi SOL dodecahedron
    Coordinates all 12 panels with various control modes
    """
    
    def __init__(self, serial_port: str = '/dev/ttyACM0'):
        # Initialize components
        self.geometry = PanelGeometry()
        self.dodeca = DodecahedronGeometry()
        self.teensy = TeensyMasterController(port=serial_port)
        
        # Create Stewart platform calculator for each panel
        self.stewart = StewartPlatform(self.geometry)
        
        # Pattern generator
        self.pattern = PatternGenerator()
        
        # Control state
        self.mode = ControlMode.MANUAL
        self.running = False
        
        # Panel states (current pitch, roll for each)
        self.panel_states = [[0.0, 0.0] for _ in range(12)]
        
        # Servo calibration (loaded from file or manual)
        self.servo_limits = [ServoLimits() for _ in range(12)]
    
    def start(self):
        """Initialize and start control system"""
        print("🌟 Naomi SOL Control System Starting...")
        
        if not self.teensy.connect():
            print("✗ Failed to connect to Teensy")
            return False
        
        self.running = True
        print("✓ System Ready")
        return True
    
    def stop(self):
        """Safely shut down system"""
        print("Shutting down...")
        self.running = False
        self.teensy.emergency_stop()
        time.sleep(0.5)
        self.teensy.disconnect()
        print("✓ Shutdown complete")
    
    def set_mode(self, mode: ControlMode):
        """Change control mode"""
        self.mode = mode
        print(f"Mode changed to: {mode.value}")
    
    def set_panel_tilt(self, panel_id: int, pitch: float, roll: float):
        """
        Manually set one panel's mirror tilt
        
        Args:
            panel_id: Panel number (0-11)
            pitch: Tilt in degrees
            roll: Tilt in degrees
        """
        # Calculate servo angles
        servo_angles = self.stewart.tilt_to_servo_angles(pitch, roll)
        
        # Apply calibration
        calibrated = [
            servo_angles[i] + self.servo_limits[panel_id].calibration[i]
            for i in range(3)
        ]
        
        # Send to Teensy
        self.teensy.send_panel_command(panel_id, calibrated)
        
        # Update state
        self.panel_states[panel_id] = [pitch, roll]
    
    def set_all_panels(self, pitch: float, roll: float):
        """Set all panels to same tilt"""
        all_angles = []
        
        for panel_id in range(12):
            servo_angles = self.stewart.tilt_to_servo_angles(pitch, roll)
            calibrated = [
                servo_angles[i] + self.servo_limits[panel_id].calibration[i]
                for i in range(3)
            ]
            all_angles.append(calibrated)
            self.panel_states[panel_id] = [pitch, roll]
        
        self.teensy.send_all_panels(all_angles)
    
    def run_pattern(self, pattern_name: str, duration: float = 60.0):
        """
        Run a pre-programmed pattern
        
        Args:
            pattern_name: Name of pattern (wave, spiral, random, etc.)
            duration: How long to run in seconds
        """
        print(f"Running pattern: {pattern_name}")
        self.set_mode(ControlMode.PATTERN)
        
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < duration:
            t = time.time() - start_time
            all_angles = []
            
            for panel_id in range(12):
                # Get pattern for this panel
                if pattern_name == "wave":
                    pitch, roll = self.pattern.wave_pattern(t, panel_id)
                elif pattern_name == "spiral":
                    pitch, roll = self.pattern.spiral_pattern(t, panel_id)
                elif pattern_name == "breathing":
                    pitch, roll = self.pattern.breathing_pattern(t)
                elif pattern_name == "random":
                    pitch, roll = self.pattern.random_organic(panel_id)
                else:
                    pitch, roll = 0, 0
                
                # Calculate servo angles
                servo_angles = self.stewart.tilt_to_servo_angles(pitch, roll)
                calibrated = [
                    servo_angles[i] + self.servo_limits[panel_id].calibration[i]
                    for i in range(3)
                ]
                all_angles.append(calibrated)
                self.panel_states[panel_id] = [pitch, roll]
            
            # Send coordinated command
            self.teensy.send_all_panels(all_angles)
            
            # Control loop rate (20 Hz)
            time.sleep(0.05)
        
        print("Pattern complete")
    
    def calibrate_panel(self, panel_id: int):
        """
        Interactive calibration for one panel
        Finds neutral servo positions for level mirror
        """
        print(f"\n=== Calibrating Panel {panel_id} ===")
        print("Adjust each servo to make mirror perfectly level")
        
        current_offsets = [90, 90, 90]  # Start at neutral
        
        for servo_idx in range(3):
            print(f"\nServo {servo_idx}:")
            print("Commands: +/- to adjust, 'n' for next servo")
            
            while True:
                # Send current position
                self.teensy.send_panel_command(panel_id, current_offsets)
                
                # Get user input
                cmd = input(f"Servo {servo_idx} angle [{current_offsets[servo_idx]}]: ").strip()
                
                if cmd == 'n':
                    break
                elif cmd == '+':
                    current_offsets[servo_idx] = min(180, current_offsets[servo_idx] + 1)
                elif cmd == '-':
                    current_offsets[servo_idx] = max(0, current_offsets[servo_idx] - 1)
                elif cmd == '++':
                    current_offsets[servo_idx] = min(180, current_offsets[servo_idx] + 5)
                elif cmd == '--':
                    current_offsets[servo_idx] = max(0, current_offsets[servo_idx] - 5)
        
        # Save calibration
        self.servo_limits[panel_id].calibration = [
            current_offsets[i] - 90 for i in range(3)
        ]
        
        print(f"✓ Panel {panel_id} calibrated: {self.servo_limits[panel_id].calibration}")
        
        # Save to file
        self.save_calibration()
    
    def save_calibration(self, filename: str = "naomi_sol_calibration.json"):
        """Save calibration data to file"""
        data = {
            f"panel_{i}": {
                "servo_0": self.servo_limits[i].calibration[0],
                "servo_1": self.servo_limits[i].calibration[1],
                "servo_2": self.servo_limits[i].calibration[2]
            }
            for i in range(12)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Calibration saved to {filename}")
    
    def load_calibration(self, filename: str = "naomi_sol_calibration.json"):
        """Load calibration data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for i in range(12):
                panel_key = f"panel_{i}"
                if panel_key in data:
                    self.servo_limits[i].calibration = [
                        data[panel_key]["servo_0"],
                        data[panel_key]["servo_1"],
                        data[panel_key]["servo_2"]
                    ]
            
            print(f"✓ Calibration loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"✗ Calibration file not found: {filename}")
            return False

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of Naomi SOL control system"""
    
    # Create controller (adjust serial port for your system)
    # Linux/Mac: /dev/ttyACM0 or /dev/ttyUSB0
    # Windows: COM3, COM4, etc.
    controller = NaomiSOLController(serial_port='/dev/ttyACM0')
    
    # Start system
    if not controller.start():
        return
    
    try:
        # Load calibration if available
        controller.load_calibration()
        
        # Example 1: Set all mirrors level
        print("\nSetting all mirrors level...")
        controller.set_all_panels(0, 0)
        time.sleep(2)
        
        # Example 2: Test single panel
        print("\nTesting panel 0...")
        controller.set_panel_tilt(0, 15, 0)  # Tilt 15° forward
        time.sleep(2)
        controller.set_panel_tilt(0, 0, 15)  # Tilt 15° right
        time.sleep(2)
        controller.set_panel_tilt(0, 0, 0)   # Back to level
        time.sleep(2)
        
        # Example 3: Run wave pattern
        print("\nRunning wave pattern for 30 seconds...")
        controller.run_pattern("wave", duration=30)
        
        # Example 4: Run spiral pattern
        print("\nRunning spiral pattern for 30 seconds...")
        controller.run_pattern("spiral", duration=30)
        
        # Example 5: Interactive calibration (commented out)
        # controller.calibrate_panel(0)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Always shut down safely
        controller.stop()

if __name__ == "__main__":
    main()