# config/configuration.py
"""
Configuration Manager for Naomi SOL
====================================
Centralized configuration management with validation and hot-reload.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

logger = logging.getLogger("ConfigManager")


@dataclass
class HardwareConfig:
    """Hardware configuration settings"""
    controller: str = "Arduino_Portenta_H7"
    connection: str = "BLE"
    ble_name: str = "NaomiSOL"
    baud_rate: int = 115200
    i2c_speed: int = 400000
    servo_count: int = 36
    panel_count: int = 12
    pwm_frequency: int = 50
    
    # Sensor configuration
    imu_type: str = "MPU9250"
    imu_sample_rate: int = 100
    madgwick_beta: float = 0.041
    
    # Optical sensors
    tsl2561_gain: int = 16
    tsl2591_gain: int = 25
    opt101_offset: float = 0.0


@dataclass
class SimulationConfig:
    """Simulation configuration settings"""
    fps: int = 60
    grid_rows: int = 12
    grid_cols: int = 18
    enable_shaders: bool = True
    enable_physics: bool = True
    physics_timestep: float = 1.0/240.0
    
    # Shape logic parameters
    edge_probability: float = 0.3
    energy_decay_rate: float = 0.001
    diffusion_rate: float = 0.1
    formation_threshold: float = 0.5
    coherence_threshold: float = 0.7


@dataclass
class OptimizationConfig:
    """Optimization configuration settings"""
    agent_count: int = 10
    iterations: int = 100
    interactive: bool = True
    learning_rate: float = 0.01
    
    # PSO parameters
    inertia_weight: float = 0.7
    cognitive_param: float = 1.5
    social_param: float = 1.5
    
    # Neural network
    use_pretrained: bool = True
    model_type: str = "torch"
    training_epochs: int = 50


@dataclass
class DesignConfig:
    """Design parameters configuration"""
    # Pentagon panel dimensions
    side_length: float = 150.0  # mm
    thickness: float = 4.0       # mm
    bevel_angle: float = 58.283  # degrees
    
    # Servo mechanism (BaBot)
    servo_pocket_depth: float = 24.0
    servo_mount_radius: float = 45.0
    mirror_diameter: float = 70.0
    ball_joint_diameter: float = 8.0
    connecting_rod_length: float = 35.0
    
    # 3D printing
    material: str = "PETG"
    infill_percentage: int = 30
    layer_height: float = 0.2
    nozzle_temp: int = 240
    bed_temp: int = 70
    print_speed: int = 50
    
    # Assembly
    screw_size: str = "M2"
    screw_length: float = 8.0


@dataclass
class ControlConfig:
    """Control system configuration"""
    # PID parameters
    pid_p: float = 2.0
    pid_i: float = 0.5
    pid_d: float = 0.1
    
    # Control limits
    max_tilt_angle: float = 15.0  # degrees
    servo_min_angle: float = 60.0
    servo_max_angle: float = 120.0
    servo_speed: float = 0.5  # 0-1
    
    # Update rates (Hz)
    sensor_update_rate: int = 100
    control_update_rate: int = 100
    ble_update_rate: int = 10
    
    # Safety
    enable_emergency_stop: bool = True
    max_acceleration: float = 50.0  # deg/s²
    watchdog_timeout: float = 5.0   # seconds


@dataclass
class NetworkConfig:
    """Network and communication configuration"""
    enable_wifi: bool = False
    wifi_ssid: str = ""
    wifi_password: str = ""
    
    enable_mqtt: bool = False
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "naomi_sol"
    
    enable_web_server: bool = False
    web_port: int = 8080
    
    enable_data_logging: bool = True
    log_directory: str = "data/logs"
    log_rotation: str = "daily"


@dataclass
class SafetyConfig:
    """Safety and limits configuration"""
    max_temperature: float = 60.0  # °C
    min_battery_voltage: float = 10.8  # V (3S LiPo)
    max_current_draw: float = 10.0  # A
    
    enable_watchdog: bool = True
    enable_overcurrent_protection: bool = True
    enable_thermal_protection: bool = True
    
    emergency_stop_acceleration: float = 100.0  # m/s²
    collision_threshold: float = 20.0  # N


@dataclass
class SystemConfig:
    """Complete system configuration"""
    version: str = "3.0"
    name: str = "Naomi SOL Hub"
    mode: str = "full_integration"
    
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Profiles
    profile: str = "default"
    available_profiles: List[str] = field(default_factory=lambda: ["default", "simulation", "hardware", "competition"])


class ConfigManager:
    """
    Configuration manager with validation and hot-reload.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/system_config.json")
        self.config = SystemConfig()
        self.watchers = []
        self.last_modified = 0
        
        # Create config directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create configuration
        self.load()
        
        logger.info(f"ConfigManager initialized with profile: {self.config.profile}")
    
    def load(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if successful
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                # Update configuration
                self._update_from_dict(data)
                
                # Update last modified time
                self.last_modified = os.path.getmtime(self.config_path)
                
                logger.info(f"Configuration loaded from {self.config_path}")
                return True
            else:
                # Create default configuration
                self.save()
                logger.info("Created default configuration")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful
        """
        try:
            # Convert to dictionary
            data = asdict(self.config)
            
            # Update metadata
            data['last_modified'] = datetime.now().isoformat()
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update last modified time
            self.last_modified = os.path.getmtime(self.config_path)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary"""
        # Update top-level fields
        for key in ['version', 'name', 'mode', 'profile']:
            if key in data:
                setattr(self.config, key, data[key])
        
        # Update sub-configurations
        if 'hardware' in data:
            self.config.hardware = HardwareConfig(**data['hardware'])
        if 'simulation' in data:
            self.config.simulation = SimulationConfig(**data['simulation'])
        if 'optimization' in data:
            self.config.optimization = OptimizationConfig(**data['optimization'])
        if 'design' in data:
            self.config.design = DesignConfig(**data['design'])
        if 'control' in data:
            self.config.control = ControlConfig(**data['control'])
        if 'network' in data:
            self.config.network = NetworkConfig(**data['network'])
        if 'safety' in data:
            self.config.safety = SafetyConfig(**data['safety'])
    
    def reload_if_changed(self) -> bool:
        """
        Reload configuration if file has changed.
        
        Returns:
            True if reloaded
        """
        if not self.config_path.exists():
            return False
        
        current_mtime = os.path.getmtime(self.config_path)
        if current_mtime > self.last_modified:
            logger.info("Configuration file changed, reloading...")
            return self.load()
        
        return False
    
    def load_profile(self, profile_name: str) -> bool:
        """
        Load a configuration profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            True if successful
        """
        profile_path = self.config_path.parent / f"profile_{profile_name}.json"
        
        if not profile_path.exists():
            logger.error(f"Profile not found: {profile_name}")
            return False
        
        try:
            with open(profile_path, 'r') as f:
                data = json.load(f)
            
            self._update_from_dict(data)
            self.config.profile = profile_name
            
            logger.info(f"Loaded profile: {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return False
    
    def save_profile(self, profile_name: str) -> bool:
        """
        Save current configuration as a profile.
        
        Args:
            profile_name: Name for the profile
            
        Returns:
            True if successful
        """
        profile_path = self.config_path.parent / f"profile_{profile_name}.json"
        
        try:
            data = asdict(self.config)
            data['profile'] = profile_name
            
            with open(profile_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Add to available profiles
            if profile_name not in self.config.available_profiles:
                self.config.available_profiles.append(profile_name)
                self.save()
            
            logger.info(f"Saved profile: {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Hardware validation
        if self.config.hardware.servo_count != 36:
            errors.append("Servo count must be 36 for Naomi SOL")
        if self.config.hardware.panel_count != 12:
            errors.append("Panel count must be 12 for dodecahedron")
        
        # Design validation
        if self.config.design.side_length < 100:
            errors.append("Pentagon side length too small (<100mm)")
        if self.config.design.thickness < 2:
            errors.append("Panel thickness too thin (<2mm)")
        if self.config.design.infill_percentage < 20:
            errors.append("Infill too low (<20%)")
        
        # Control validation
        if self.config.control.max_tilt_angle > 30:
            errors.append("Max tilt angle too large (>30°)")
        if self.config.control.sensor_update_rate > 1000:
            errors.append("Sensor update rate too high (>1000Hz)")
        
        # Safety validation
        if not self.config.safety.enable_watchdog:
            errors.append("Warning: Watchdog disabled")
        if self.config.safety.max_temperature > 80:
            errors.append("Max temperature too high (>80°C)")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Configuration validation failed: {errors}")
        
        return is_valid, errors
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path.
        
        Args:
            path: Dot-separated path (e.g., "hardware.servo_count")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        try:
            parts = path.split('.')
            value = self.config
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, path: str, value: Any) -> bool:
        """
        Set configuration value by path.
        
        Args:
            path: Dot-separated path
            value: Value to set
            
        Returns:
            True if successful
        """
        try:
            parts = path.split('.')
            obj = self.config
            
            # Navigate to parent
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return False
            
            # Set value
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)
                self.save()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to set config value: {e}")
            return False
    
    def export_arduino_header(self, output_path: str = "firmware/config.h"):
        """
        Export configuration as Arduino header file.
        
        Args:
            output_path: Path for header file
        """
        header = f"""// Naomi SOL Configuration
// Auto-generated from system_config.json
// Generated: {datetime.now().isoformat()}

#ifndef NAOMI_CONFIG_H
#define NAOMI_CONFIG_H

// System
#define FIRMWARE_VERSION "{self.config.version}"
#define PANEL_COUNT {self.config.hardware.panel_count}
#define SERVO_COUNT {self.config.hardware.servo_count}

// Hardware
#define I2C_SPEED {self.config.hardware.i2c_speed}L
#define PWM_FREQUENCY {self.config.hardware.pwm_frequency}
#define BAUD_RATE {self.config.hardware.baud_rate}L

// Sensors
#define IMU_SAMPLE_RATE {self.config.hardware.imu_sample_rate}
#define MADGWICK_BETA {self.config.hardware.madgwick_beta}f

// Control
#define PID_P {self.config.control.pid_p}f
#define PID_I {self.config.control.pid_i}f
#define PID_D {self.config.control.pid_d}f
#define MAX_TILT_ANGLE {self.config.control.max_tilt_angle}f
#define SERVO_MIN_ANGLE {self.config.control.servo_min_angle}f
#define SERVO_MAX_ANGLE {self.config.control.servo_max_angle}f

// Update rates
#define SENSOR_UPDATE_RATE {self.config.control.sensor_update_rate}
#define CONTROL_UPDATE_RATE {self.config.control.control_update_rate}
#define BLE_UPDATE_RATE {self.config.control.ble_update_rate}

// Safety
#define MAX_TEMPERATURE {self.config.safety.max_temperature}f
#define MIN_BATTERY_VOLTAGE {self.config.safety.min_battery_voltage}f
#define WATCHDOG_TIMEOUT {int(self.config.control.watchdog_timeout * 1000)}

// Design parameters
#define PANEL_SIDE_LENGTH {self.config.design.side_length}f
#define PANEL_THICKNESS {self.config.design.thickness}f
#define MIRROR_DIAMETER {self.config.design.mirror_diameter}f

#endif // NAOMI_CONFIG_H
"""
        
        with open(output_path, 'w') as f:
            f.write(header)
        
        logger.info(f"Exported Arduino header to {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self.config)
    
    def __str__(self) -> str:
        """String representation"""
        return json.dumps(self.to_dict(), indent=2)


# Global configuration instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def test_config():
    """Test configuration manager"""
    print("Testing Configuration Manager...")
    
    # Create manager
    config = ConfigManager(Path("test_config.json"))
    
    # Test get/set
    print(f"Servo count: {config.get('hardware.servo_count')}")
    config.set('design.side_length', 160)
    print(f"Side length: {config.get('design.side_length')}")
    
    # Validate
    valid, errors = config.validate()
    print(f"Configuration valid: {valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Save profile
    config.save_profile("test_profile")
    
    # Export Arduino header
    config.export_arduino_header("test_config.h")
    
    print("\nConfiguration test complete!")


if __name__ == "__main__":
    test_config()
