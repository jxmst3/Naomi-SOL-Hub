# naomi/hardware_control.py
"""
Naomi SOL Hardware Control Interface
====================================
Interfaces with Arduino Portenta H7/Teensy 4.1 and all sensors.
Supports BLE, Serial, and Mock connections.

Hardware Components:
- 12 Pentagon panels (11 standard + 1 with laser ports)
- 36 MG90S servo motors (3 per panel - BaBot mechanism)
- 12 MPU-9250 IMUs (1 per panel)
- Optical sensors (TSL2561, TSL2591, OPT101, photodiodes)
- Arduino Portenta H7 or Teensy 4.1 controller
- PCA9685 PWM drivers for servo control
"""

import json
import time
import threading
import queue
import struct
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger("NaomiHardware")

# Try to import hardware libraries
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial not installed - Serial connection unavailable")

try:
    import asyncio
    import bleak
    from bleak import BleakClient, BleakScanner
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False
    logger.warning("bleak not installed - BLE connection unavailable")


class ConnectionType(Enum):
    """Hardware connection types"""
    SERIAL = "Serial"
    BLE = "BLE"
    MOCK = "Mock"


@dataclass
class SensorData:
    """Complete sensor data from one panel"""
    panel_id: int
    timestamp: float
    
    # IMU data (MPU-9250)
    roll: float  # degrees
    pitch: float  # degrees
    yaw: float  # degrees
    accel_x: float  # m/s²
    accel_y: float  # m/s²
    accel_z: float  # m/s²
    gyro_x: float  # rad/s
    gyro_y: float  # rad/s
    gyro_z: float  # rad/s
    mag_x: float  # µT
    mag_y: float  # µT
    mag_z: float  # µT
    
    # Optical sensors
    light_intensity: float  # lux (TSL2561/TSL2591)
    ir_intensity: float  # arbitrary units (OPT101)
    photodiode_voltage: float  # volts
    
    # Environmental
    temperature: float  # °C
    pressure: float  # Pa
    
    # Computed values
    anomaly_score: float
    stability: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "panel_id": self.panel_id,
            "timestamp": self.timestamp,
            "imu": {
                "roll": self.roll,
                "pitch": self.pitch,
                "yaw": self.yaw,
                "accel": [self.accel_x, self.accel_y, self.accel_z],
                "gyro": [self.gyro_x, self.gyro_y, self.gyro_z],
                "mag": [self.mag_x, self.mag_y, self.mag_z]
            },
            "optical": {
                "light": self.light_intensity,
                "ir": self.ir_intensity,
                "photodiode": self.photodiode_voltage
            },
            "environmental": {
                "temperature": self.temperature,
                "pressure": self.pressure
            },
            "computed": {
                "anomaly_score": self.anomaly_score,
                "stability": self.stability
            }
        }


@dataclass
class PanelControl:
    """Control parameters for one panel"""
    panel_id: int
    servo1_angle: float  # 0-180 degrees
    servo2_angle: float  # 0-180 degrees
    servo3_angle: float  # 0-180 degrees
    tilt_x: float  # -15 to +15 degrees
    tilt_y: float  # -15 to +15 degrees
    led_state: bool
    
    def to_command_dict(self) -> Dict:
        """Convert to command dictionary"""
        return {
            "cmd": "SET_PANEL",
            "id": self.panel_id,
            "s1": self.servo1_angle,
            "s2": self.servo2_angle,
            "s3": self.servo3_angle,
            "tx": self.tilt_x,
            "ty": self.tilt_y,
            "led": 1 if self.led_state else 0
        }


class NaomiHardwareController:
    """
    Main hardware controller for Naomi SOL system.
    Manages communication with Arduino and all sensors.
    """
    
    # BLE Service and Characteristic UUIDs
    BLE_SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
    BLE_CHAR_WRITE_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
    BLE_CHAR_READ_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"
    
    def __init__(self, connection: str = "BLE", 
                 port: Optional[str] = None,
                 baud_rate: int = 115200,
                 design_params: Optional[Any] = None):
        """
        Initialize hardware controller.
        
        Args:
            connection: Connection type (BLE, Serial, Mock)
            port: Serial port (auto-detect if None)
            baud_rate: Serial baud rate
            design_params: Design parameters from main system
        """
        self.connection_type = ConnectionType(connection)
        self.port = port
        self.baud_rate = baud_rate
        self.design_params = design_params
        
        # Connection objects
        self.serial_conn = None
        self.ble_client = None
        self.is_connected = False
        
        # Data queues
        self.command_queue = queue.Queue()
        self.sensor_queue = queue.Queue()
        
        # Sensor data storage
        self.latest_sensor_data = {}  # panel_id -> SensorData
        
        # Panel states
        self.panel_states = {}  # panel_id -> PanelControl
        for i in range(12):
            self.panel_states[i] = PanelControl(
                panel_id=i,
                servo1_angle=90,
                servo2_angle=90,
                servo3_angle=90,
                tilt_x=0,
                tilt_y=0,
                led_state=False
            )
        
        # Threading
        self.running = False
        self.read_thread = None
        self.write_thread = None
        
        # Mock mode data generation
        self.mock_time = 0.0
        
        # Calibration data
        self.calibration = self._load_calibration()
        
        logger.info(f"Hardware controller initialized: {connection} mode")
    
    def _load_calibration(self) -> Dict:
        """Load calibration data for sensors"""
        # Default calibration values
        return {
            "imu": {
                "accel_offset": [0, 0, 0],
                "accel_scale": [1, 1, 1],
                "gyro_offset": [0, 0, 0],
                "mag_offset": [0, 0, 0],
                "mag_scale": [1, 1, 1]
            },
            "optical": {
                "tsl2561_gain": 16,
                "tsl2591_gain": 25,
                "opt101_offset": 0
            },
            "servo": {
                "min_pulse": [150] * 36,  # Per servo calibration
                "max_pulse": [600] * 36,
                "center": [375] * 36
            }
        }
    
    def connect(self) -> bool:
        """
        Connect to hardware.
        
        Returns:
            True if connection successful
        """
        try:
            if self.connection_type == ConnectionType.SERIAL:
                return self._connect_serial()
            elif self.connection_type == ConnectionType.BLE:
                return self._connect_ble()
            elif self.connection_type == ConnectionType.MOCK:
                return self._connect_mock()
            else:
                logger.error(f"Unknown connection type: {self.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _connect_serial(self) -> bool:
        """Connect via Serial/USB"""
        if not SERIAL_AVAILABLE:
            logger.error("Serial not available - install pyserial")
            return False
        
        try:
            # Auto-detect port if not specified
            if not self.port:
                ports = serial.tools.list_ports.comports()
                for port in ports:
                    # Look for Arduino/Teensy
                    if "Arduino" in port.description or "Teensy" in port.description:
                        self.port = port.device
                        logger.info(f"Auto-detected port: {self.port}")
                        break
                else:
                    logger.error("No Arduino/Teensy found")
                    return False
            
            # Open serial connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.1
            )
            
            # Wait for Arduino to reset
            time.sleep(2)
            
            # Send handshake
            self._send_command({"cmd": "HELLO", "version": "3.0"})
            
            # Wait for response
            response = self._read_response(timeout=5)
            if response and response.get("status") == "OK":
                logger.info(f"Connected to {self.port} at {self.baud_rate} baud")
                self.is_connected = True
                self._start_threads()
                return True
            else:
                logger.error("Handshake failed")
                return False
                
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            return False
    
    def _connect_ble(self) -> bool:
        """Connect via Bluetooth Low Energy"""
        if not BLE_AVAILABLE:
            logger.error("BLE not available - install bleak")
            return False
        
        async def connect_async():
            try:
                # Scan for devices
                logger.info("Scanning for BLE devices...")
                devices = await BleakScanner.discover(timeout=5)
                
                target_device = None
                for device in devices:
                    if "NaomiSOL" in (device.name or ""):
                        target_device = device
                        logger.info(f"Found Naomi SOL: {device.address}")
                        break
                
                if not target_device:
                    logger.error("Naomi SOL BLE device not found")
                    return False
                
                # Connect
                self.ble_client = BleakClient(target_device.address)
                await self.ble_client.connect()
                
                if self.ble_client.is_connected:
                    logger.info(f"Connected to {target_device.name}")
                    self.is_connected = True
                    
                    # Start notification handler
                    await self.ble_client.start_notify(
                        self.BLE_CHAR_READ_UUID,
                        self._ble_notification_handler
                    )
                    
                    return True
                else:
                    return False
                    
            except Exception as e:
                logger.error(f"BLE connection failed: {e}")
                return False
        
        # Run async connection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(connect_async())
        
        if result:
            self._start_threads()
        
        return result
    
    def _connect_mock(self) -> bool:
        """Connect in mock mode (for testing)"""
        logger.info("Connected in MOCK mode")
        self.is_connected = True
        self._start_threads()
        return True
    
    def _start_threads(self):
        """Start read/write threads"""
        self.running = True
        
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        
        self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.write_thread.start()
    
    def _read_loop(self):
        """Continuous read loop for sensor data"""
        while self.running:
            try:
                if self.connection_type == ConnectionType.MOCK:
                    # Generate mock sensor data
                    self._generate_mock_data()
                    time.sleep(0.01)  # 100Hz
                    
                elif self.connection_type == ConnectionType.SERIAL:
                    if self.serial_conn and self.serial_conn.in_waiting:
                        line = self.serial_conn.readline()
                        if line:
                            self._process_sensor_data(line)
                            
                elif self.connection_type == ConnectionType.BLE:
                    # BLE uses notification handler
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Read error: {e}")
                time.sleep(0.1)
    
    def _write_loop(self):
        """Continuous write loop for commands"""
        while self.running:
            try:
                if not self.command_queue.empty():
                    command = self.command_queue.get(timeout=0.1)
                    self._send_command(command)
                else:
                    time.sleep(0.01)
                    
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Write error: {e}")
    
    def _send_command(self, command: Dict):
        """Send command to hardware"""
        try:
            if self.connection_type == ConnectionType.SERIAL:
                if self.serial_conn:
                    json_str = json.dumps(command) + "\n"
                    self.serial_conn.write(json_str.encode())
                    logger.debug(f"Sent: {command}")
                    
            elif self.connection_type == ConnectionType.BLE:
                if self.ble_client and self.ble_client.is_connected:
                    json_str = json.dumps(command)
                    
                    async def write_async():
                        await self.ble_client.write_gatt_char(
                            self.BLE_CHAR_WRITE_UUID,
                            json_str.encode()
                        )
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(write_async())
                    logger.debug(f"Sent: {command}")
                    
            elif self.connection_type == ConnectionType.MOCK:
                logger.debug(f"Mock sent: {command}")
                
        except Exception as e:
            logger.error(f"Send command error: {e}")
    
    def _read_response(self, timeout: float = 1.0) -> Optional[Dict]:
        """Read response from hardware"""
        if self.connection_type == ConnectionType.SERIAL:
            if self.serial_conn:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.serial_conn.in_waiting:
                        line = self.serial_conn.readline()
                        try:
                            return json.loads(line.decode().strip())
                        except:
                            pass
        return None
    
    def _process_sensor_data(self, data: bytes):
        """Process incoming sensor data"""
        try:
            # Try to parse as JSON
            json_data = json.loads(data.decode().strip())
            
            # Create SensorData object
            sensor_data = SensorData(
                panel_id=json_data.get("id", 0),
                timestamp=json_data.get("t", time.time()),
                roll=json_data.get("r", 0),
                pitch=json_data.get("p", 0),
                yaw=json_data.get("y", 0),
                accel_x=json_data.get("ax", 0),
                accel_y=json_data.get("ay", 0),
                accel_z=json_data.get("az", 9.81),
                gyro_x=json_data.get("gx", 0),
                gyro_y=json_data.get("gy", 0),
                gyro_z=json_data.get("gz", 0),
                mag_x=json_data.get("mx", 0),
                mag_y=json_data.get("my", 0),
                mag_z=json_data.get("mz", 0),
                light_intensity=json_data.get("l", 0),
                ir_intensity=json_data.get("ir", 0),
                photodiode_voltage=json_data.get("pd", 0),
                temperature=json_data.get("temp", 25),
                pressure=json_data.get("pr", 101325),
                anomaly_score=json_data.get("as", 0),
                stability=json_data.get("st", 1)
            )
            
            # Store latest data
            self.latest_sensor_data[sensor_data.panel_id] = sensor_data
            
            # Put in queue
            self.sensor_queue.put(sensor_data)
            
        except Exception as e:
            logger.debug(f"Parse error: {e} for data: {data}")
    
    def _ble_notification_handler(self, sender, data):
        """Handle BLE notifications"""
        self._process_sensor_data(data)
    
    def _generate_mock_data(self):
        """Generate mock sensor data for testing"""
        self.mock_time += 0.01  # 100Hz
        
        for panel_id in range(12):
            # Generate realistic sensor data with some variation
            sensor_data = SensorData(
                panel_id=panel_id,
                timestamp=time.time(),
                roll=5 * np.sin(self.mock_time + panel_id * 0.5),
                pitch=5 * np.cos(self.mock_time + panel_id * 0.5),
                yaw=panel_id * 30 + 10 * np.sin(self.mock_time * 0.1),
                accel_x=0.1 * np.random.randn(),
                accel_y=0.1 * np.random.randn(),
                accel_z=9.81 + 0.1 * np.random.randn(),
                gyro_x=0.01 * np.random.randn(),
                gyro_y=0.01 * np.random.randn(),
                gyro_z=0.01 * np.random.randn(),
                mag_x=48 + np.random.randn(),
                mag_y=5 + np.random.randn(),
                mag_z=15 + np.random.randn(),
                light_intensity=100 + 50 * np.sin(self.mock_time + panel_id),
                ir_intensity=50 + 20 * np.cos(self.mock_time + panel_id),
                photodiode_voltage=2.5 + 0.5 * np.sin(self.mock_time * 2),
                temperature=25 + 2 * np.sin(self.mock_time * 0.01),
                pressure=101325 + 100 * np.sin(self.mock_time * 0.02),
                anomaly_score=max(0, min(1, 0.5 + 0.3 * np.sin(self.mock_time * 0.5))),
                stability=0.9 + 0.1 * np.random.random()
            )
            
            self.latest_sensor_data[panel_id] = sensor_data
            
            # Only put every 10th reading in queue (simulate 10Hz output)
            if int(self.mock_time * 100) % 10 == 0:
                self.sensor_queue.put(sensor_data)
    
    # ============== PUBLIC INTERFACE ==============
    
    def disconnect(self):
        """Disconnect from hardware"""
        self.running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=1)
        if self.write_thread:
            self.write_thread.join(timeout=1)
        
        if self.connection_type == ConnectionType.SERIAL:
            if self.serial_conn:
                self.serial_conn.close()
                
        elif self.connection_type == ConnectionType.BLE:
            if self.ble_client:
                async def disconnect_async():
                    await self.ble_client.disconnect()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(disconnect_async())
        
        self.is_connected = False
        logger.info("Disconnected from hardware")
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to hardware"""
        logger.info("Attempting reconnection...")
        self.disconnect()
        time.sleep(1)
        return self.connect()
    
    def read_all_sensors(self) -> Dict:
        """
        Read all sensor data.
        
        Returns:
            Dictionary of panel_id -> sensor data
        """
        result = {}
        for panel_id, data in self.latest_sensor_data.items():
            result[panel_id] = data.to_dict()
        return result
    
    def read_panel_sensors(self, panel_id: int) -> Optional[Dict]:
        """
        Read sensors from specific panel.
        
        Args:
            panel_id: Panel ID (0-11)
            
        Returns:
            Sensor data dictionary or None
        """
        if panel_id in self.latest_sensor_data:
            return self.latest_sensor_data[panel_id].to_dict()
        return None
    
    def set_panel_tilt(self, panel_id: int, tilt_x: float, tilt_y: float):
        """
        Set panel tilt angles.
        
        Args:
            panel_id: Panel ID (0-11)
            tilt_x: X-axis tilt (-15 to +15 degrees)
            tilt_y: Y-axis tilt (-15 to +15 degrees)
        """
        # Constrain angles
        tilt_x = max(-15, min(15, tilt_x))
        tilt_y = max(-15, min(15, tilt_y))
        
        # Calculate servo angles using inverse kinematics
        angles = self._inverse_kinematics(tilt_x, tilt_y)
        
        # Update panel state
        if panel_id in self.panel_states:
            state = self.panel_states[panel_id]
            state.servo1_angle = angles[0]
            state.servo2_angle = angles[1]
            state.servo3_angle = angles[2]
            state.tilt_x = tilt_x
            state.tilt_y = tilt_y
            
            # Send command
            self.command_queue.put(state.to_command_dict())
    
    def _inverse_kinematics(self, tilt_x: float, tilt_y: float) -> Tuple[float, float, float]:
        """
        Calculate servo angles for desired tilt.
        Uses 3-DOF parallel platform kinematics.
        
        Args:
            tilt_x: X-axis tilt in degrees
            tilt_y: Y-axis tilt in degrees
            
        Returns:
            Tuple of (servo1, servo2, servo3) angles in degrees
        """
        # Convert to radians
        tx_rad = np.radians(tilt_x)
        ty_rad = np.radians(tilt_y)
        
        # Platform geometry (120 degrees apart)
        angles = [0, 120, 240]
        servos = []
        
        for angle in angles:
            angle_rad = np.radians(angle)
            
            # Calculate servo angle based on platform tilt
            # This is simplified - real implementation would use full kinematic model
            servo_angle = 90 + (
                tilt_x * np.cos(angle_rad) - 
                tilt_y * np.sin(angle_rad)
            )
            
            # Constrain to servo limits
            servo_angle = max(60, min(120, servo_angle))
            servos.append(servo_angle)
        
        return tuple(servos)
    
    def home_all_panels(self):
        """Move all panels to home position"""
        logger.info("Homing all panels...")
        
        for panel_id in range(12):
            self.set_panel_tilt(panel_id, 0, 0)
        
        # Send home command
        self.command_queue.put({"cmd": "HOME_ALL"})
    
    def execute_command(self, command: Dict):
        """
        Execute arbitrary command.
        
        Args:
            command: Command dictionary
        """
        self.command_queue.put(command)
    
    def run_calibration(self):
        """Run sensor calibration routine"""
        logger.info("Starting calibration...")
        
        # Send calibration command
        self.command_queue.put({"cmd": "CALIBRATE"})
        
        # Wait for completion
        time.sleep(10)
        
        # Read calibration data
        response = self._read_response(timeout=5)
        if response and "calibration" in response:
            self.calibration = response["calibration"]
            
            # Save calibration
            with open("calibration.json", "w") as f:
                json.dump(self.calibration, f, indent=2)
            
            logger.info("Calibration complete")
            return True
        else:
            logger.error("Calibration failed")
            return False
    
    def get_status(self) -> Dict:
        """Get hardware status"""
        return {
            "connected": self.is_connected,
            "connection_type": self.connection_type.value,
            "panels_online": len(self.latest_sensor_data),
            "command_queue_size": self.command_queue.qsize(),
            "sensor_queue_size": self.sensor_queue.qsize()
        }
    
    def test_panel(self, panel_id: int):
        """Test individual panel movement"""
        logger.info(f"Testing panel {panel_id}")
        
        # Move in a circle
        steps = 36
        for i in range(steps):
            angle = i * 360 / steps
            tilt_x = 10 * np.sin(np.radians(angle))
            tilt_y = 10 * np.cos(np.radians(angle))
            self.set_panel_tilt(panel_id, tilt_x, tilt_y)
            time.sleep(0.1)
        
        # Return to home
        self.set_panel_tilt(panel_id, 0, 0)
        logger.info(f"Panel {panel_id} test complete")


def test_hardware():
    """Test hardware interface"""
    print("Testing Naomi SOL Hardware Interface...")
    
    # Try different connection modes
    for connection_type in ["Mock", "Serial", "BLE"]:
        print(f"\n--- Testing {connection_type} mode ---")
        
        controller = NaomiHardwareController(connection=connection_type)
        
        if controller.connect():
            print(f"✓ Connected via {connection_type}")
            
            # Read sensors
            time.sleep(1)
            sensors = controller.read_all_sensors()
            print(f"Reading {len(sensors)} panels")
            
            if sensors:
                # Show first panel data
                first_panel = list(sensors.values())[0]
                print(f"Panel 0 IMU: Roll={first_panel['imu']['roll']:.1f}°")
                print(f"Panel 0 Light: {first_panel['optical']['light']:.1f} lux")
            
            # Test panel movement
            if connection_type == "Mock":
                controller.test_panel(0)
            
            # Get status
            status = controller.get_status()
            print(f"Status: {status}")
            
            controller.disconnect()
            print(f"✓ Disconnected")
        else:
            print(f"✗ Could not connect via {connection_type}")
    
    print("\nHardware test complete!")


if __name__ == "__main__":
    test_hardware()
