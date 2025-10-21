# tests/test_naomi_sol.py
"""
Comprehensive Test Suite for Naomi SOL Hub System
==================================================
Tests all components with unit and integration tests.
"""

import unittest
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from sim.shape_logic import ShapeLogicSimulator, EdgePolarity, SquareBlock
from ai.swarm import SwarmLordsController, Agent
from naomi.hardware_control import NaomiHardwareController, SensorData, PanelControl
from naomi.cad_interface import NaomiSOLPart
from config.configuration import ConfigManager, SystemConfig


class TestShapeLogic(unittest.TestCase):
    """Test the Shape Logic Simulator"""
    
    def setUp(self):
        """Set up test simulator"""
        self.sim = ShapeLogicSimulator(rows=6, cols=8, naomi_mode=True)
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertEqual(self.sim.rows, 6)
        self.assertEqual(self.sim.cols, 8)
        self.assertTrue(self.sim.naomi_mode)
        self.assertEqual(len(self.sim.H), 7)  # rows + 1
        self.assertEqual(len(self.sim.V), 6)  # rows
        self.assertEqual(len(self.sim.H[0]), 8)  # cols
        self.assertEqual(len(self.sim.V[0]), 9)  # cols + 1
    
    def test_edge_polarity(self):
        """Test edge polarity operations"""
        edge = EdgePolarity()
        self.assertFalse(edge.is_set())
        
        edge.set(1, 2.0)
        self.assertTrue(edge.is_set())
        self.assertEqual(edge.polarity, 1)
        self.assertEqual(edge.energy, 2.0)
        
        edge.invert()
        self.assertEqual(edge.polarity, -1)
        
        edge.decay(0.5)
        self.assertEqual(edge.energy, 1.0)
        
        edge.clear()
        self.assertFalse(edge.is_set())
    
    def test_set_edges(self):
        """Test setting horizontal and vertical edges"""
        self.sim.set_horizontal(0, 0, 1, 2.0)
        self.assertEqual(self.sim.H[0][0].polarity, 1)
        self.assertEqual(self.sim.H[0][0].energy, 2.0)
        
        self.sim.set_vertical(0, 0, -1, 1.5)
        self.assertEqual(self.sim.V[0][0].polarity, -1)
        self.assertEqual(self.sim.V[0][0].energy, 1.5)
    
    def test_square_formation(self):
        """Test square formation logic"""
        # Set up edges for square formation
        # Top and bottom: -1, Left and right: +1
        self.sim.set_horizontal(0, 0, -1, 1.0)
        self.sim.set_horizontal(1, 0, -1, 1.0)
        self.sim.set_vertical(0, 0, 1, 1.0)
        self.sim.set_vertical(0, 1, 1, 1.0)
        
        # Check if square can form
        self.assertTrue(self.sim.can_form_square(0, 0))
        
        # Form square
        self.assertTrue(self.sim.form_square(0, 0))
        self.assertIn((0, 0), self.sim.squares)
        
        # Check square properties
        square = self.sim.squares[(0, 0)]
        self.assertEqual(square.r, 0)
        self.assertEqual(square.c, 0)
        self.assertGreater(square.energy, 0)
    
    def test_simulation_step(self):
        """Test simulation stepping"""
        self.sim.randomize_edges(probability=0.3)
        initial_time = self.sim.sim_time
        
        self.sim.step()
        
        self.assertGreater(self.sim.sim_time, initial_time)
        self.assertIsNotNone(self.sim.stats)
        self.assertIn('total_energy', self.sim.stats)
    
    def test_export_import_state(self):
        """Test state export and import"""
        # Set up some state
        self.sim.set_horizontal(0, 0, 1, 2.0)
        self.sim.form_square(0, 0)
        
        # Export state
        state = self.sim.export_state()
        self.assertEqual(state['rows'], 6)
        self.assertEqual(state['cols'], 8)
        self.assertIn('horizontal_edges', state)
        self.assertIn('squares', state)
        
        # Create new simulator and import
        sim2 = ShapeLogicSimulator(rows=3, cols=3)
        sim2.import_state(state)
        
        self.assertEqual(sim2.rows, 6)
        self.assertEqual(sim2.cols, 8)
        self.assertEqual(sim2.H[0][0].polarity, 1)
        self.assertIn((0, 0), sim2.squares)


class TestSwarmLords(unittest.TestCase):
    """Test the SwarmLords Optimizer"""
    
    def setUp(self):
        """Set up test optimizer"""
        self.swarm = SwarmLordsController(
            agent_count=3,
            interactive=False,
            use_pretrained=False
        )
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.swarm.agent_count, 3)
        self.assertEqual(len(self.swarm.agents), 3)
        self.assertFalse(self.swarm.interactive)
        self.assertEqual(self.swarm.global_best_fitness, -np.inf)
    
    def test_agent_creation(self):
        """Test agent creation"""
        agent = self.swarm.agents[0]
        self.assertIsInstance(agent, Agent)
        self.assertEqual(len(agent.position), self.swarm.n_params)
        self.assertEqual(len(agent.velocity), self.swarm.n_params)
        
        # Check position is within bounds
        for i, (param, bounds) in enumerate(self.swarm.PARAM_BOUNDS.items()):
            self.assertGreaterEqual(agent.position[i], bounds[0])
            self.assertLessEqual(agent.position[i], bounds[1])
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation"""
        position = np.array([150, 4, 24, 70, 30, 0.2, 0.5, 10, 2.0, 0.5, 0.1])
        fitness = self.swarm.evaluate_fitness(position)
        
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, -100)  # Should have reasonable fitness
    
    def test_position_conversion(self):
        """Test position to params conversion"""
        position = np.array([150, 4, 24, 70, 30, 0.2, 0.5, 10, 2.0, 0.5, 0.1])
        params = self.swarm._position_to_params(position)
        
        self.assertEqual(params['side_length'], 150)
        self.assertEqual(params['thickness'], 4)
        self.assertEqual(params['servo_pocket_depth'], 24)
    
    def test_optimization_step(self):
        """Test optimization step"""
        initial_fitness = self.swarm.global_best_fitness
        
        self.swarm.step()
        
        # Global best should be updated
        self.assertGreater(self.swarm.global_best_fitness, initial_fitness)
        
        # Agents should have moved
        for agent in self.swarm.agents:
            self.assertIsNotNone(agent.fitness)
    
    def test_optimize(self):
        """Test full optimization"""
        initial_params = {
            'side_length': 150,
            'thickness': 4,
            'servo_pocket_depth': 24,
            'mirror_diameter': 70,
            'infill_percentage': 30,
            'layer_height': 0.2,
            'servo_speed': 0.5,
            'sensor_gain': 10,
            'control_p': 2.0,
            'control_i': 0.5,
            'control_d': 0.1
        }
        
        result = self.swarm.optimize(
            initial_params=initial_params,
            iterations=5  # Small number for testing
        )
        
        self.assertIn('params', result)
        self.assertIn('fitness', result)
        self.assertIn('improvement', result)
        self.assertGreater(result['fitness'], -np.inf)


class TestHardwareControl(unittest.TestCase):
    """Test the Hardware Control Interface"""
    
    def setUp(self):
        """Set up test hardware controller"""
        self.hw = NaomiHardwareController(connection="Mock")
    
    def test_initialization(self):
        """Test hardware initialization"""
        self.assertEqual(self.hw.connection_type.value, "Mock")
        self.assertEqual(len(self.hw.panel_states), 12)
        self.assertFalse(self.hw.is_connected)
    
    def test_mock_connection(self):
        """Test mock connection"""
        self.assertTrue(self.hw.connect())
        self.assertTrue(self.hw.is_connected)
        
        # Wait for mock data generation
        time.sleep(0.1)
        
        # Should have sensor data
        self.assertGreater(len(self.hw.latest_sensor_data), 0)
    
    def test_sensor_data(self):
        """Test sensor data structure"""
        sensor_data = SensorData(
            panel_id=0,
            timestamp=time.time(),
            roll=5.0,
            pitch=-3.0,
            yaw=45.0,
            accel_x=0.1,
            accel_y=0.2,
            accel_z=9.81,
            gyro_x=0.01,
            gyro_y=0.02,
            gyro_z=0.03,
            mag_x=48.0,
            mag_y=5.0,
            mag_z=15.0,
            light_intensity=100.0,
            ir_intensity=50.0,
            photodiode_voltage=2.5,
            temperature=25.0,
            pressure=101325.0,
            anomaly_score=0.1,
            stability=0.9
        )
        
        data_dict = sensor_data.to_dict()
        self.assertEqual(data_dict['panel_id'], 0)
        self.assertEqual(data_dict['imu']['roll'], 5.0)
        self.assertEqual(data_dict['optical']['light'], 100.0)
    
    def test_panel_control(self):
        """Test panel control"""
        panel = PanelControl(
            panel_id=0,
            servo1_angle=90,
            servo2_angle=90,
            servo3_angle=90,
            tilt_x=0,
            tilt_y=0,
            led_state=False
        )
        
        cmd = panel.to_command_dict()
        self.assertEqual(cmd['cmd'], 'SET_PANEL')
        self.assertEqual(cmd['id'], 0)
        self.assertEqual(cmd['s1'], 90)
    
    def test_inverse_kinematics(self):
        """Test inverse kinematics calculation"""
        angles = self.hw._inverse_kinematics(10, 5)
        
        self.assertEqual(len(angles), 3)
        for angle in angles:
            self.assertGreaterEqual(angle, 60)
            self.assertLessEqual(angle, 120)
    
    def test_disconnect(self):
        """Test disconnection"""
        self.hw.connect()
        self.assertTrue(self.hw.is_connected)
        
        self.hw.disconnect()
        self.assertFalse(self.hw.is_connected)


class TestCADInterface(unittest.TestCase):
    """Test the CAD Interface"""
    
    def setUp(self):
        """Set up test CAD generator"""
        self.params = {
            'side_length': 150,
            'thickness': 4,
            'servo_pocket_depth': 24,
            'mirror_diameter': 70,
            'infill_percentage': 30,
            'layer_height': 0.2
        }
        self.part = NaomiSOLPart("Pentagon_Base_Panel", self.params)
    
    def test_initialization(self):
        """Test CAD part initialization"""
        self.assertEqual(self.part.part_name, "Pentagon_Base_Panel")
        self.assertEqual(self.part.side_length, 150)
        self.assertEqual(self.part.thickness, 4)
    
    def test_print_time_estimate(self):
        """Test print time estimation"""
        time_est = self.part.get_print_time_estimate()
        
        self.assertIsInstance(time_est, float)
        self.assertGreater(time_est, 0)
        self.assertLess(time_est, 100)  # Should be reasonable
    
    def test_material_usage(self):
        """Test material usage estimation"""
        material = self.part.get_material_usage()
        
        self.assertIsInstance(material, float)
        self.assertGreater(material, 0)
        self.assertLess(material, 1000)  # Should be reasonable in grams


class TestConfiguration(unittest.TestCase):
    """Test the Configuration Manager"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = ConfigManager(Path("test_config.json"))
    
    def tearDown(self):
        """Clean up test files"""
        if Path("test_config.json").exists():
            Path("test_config.json").unlink()
    
    def test_initialization(self):
        """Test configuration initialization"""
        self.assertIsInstance(self.config.config, SystemConfig)
        self.assertEqual(self.config.config.version, "3.0")
        self.assertEqual(self.config.config.hardware.panel_count, 12)
    
    def test_get_set(self):
        """Test get and set operations"""
        # Test get
        servo_count = self.config.get('hardware.servo_count')
        self.assertEqual(servo_count, 36)
        
        # Test set
        self.assertTrue(self.config.set('design.side_length', 160))
        self.assertEqual(self.config.get('design.side_length'), 160)
        
        # Test invalid path
        self.assertIsNone(self.config.get('invalid.path'))
        self.assertFalse(self.config.set('invalid.path', 123))
    
    def test_validation(self):
        """Test configuration validation"""
        valid, errors = self.config.validate()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
        # Set invalid value
        self.config.config.design.side_length = 50  # Too small
        valid, errors = self.config.validate()
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
    
    def test_save_load(self):
        """Test save and load operations"""
        # Modify config
        self.config.set('design.thickness', 5)
        
        # Save
        self.assertTrue(self.config.save())
        
        # Create new config and load
        config2 = ConfigManager(Path("test_config.json"))
        self.assertEqual(config2.get('design.thickness'), 5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_simulation_to_optimization(self):
        """Test simulation feeding into optimization"""
        # Create simulator
        sim = ShapeLogicSimulator(rows=6, cols=8)
        sim.randomize_edges(probability=0.3)
        
        # Run simulation
        for _ in range(10):
            sim.step()
        
        # Get state
        state = sim.export_state()
        
        # Create optimizer
        swarm = SwarmLordsController(agent_count=3, interactive=False)
        
        # Use simulation metrics as fitness inputs
        initial_params = {
            'side_length': 150,
            'thickness': 4,
            'servo_pocket_depth': 24,
            'mirror_diameter': 70,
            'infill_percentage': 30,
            'layer_height': 0.2,
            'servo_speed': 0.5,
            'sensor_gain': 10,
            'control_p': 2.0,
            'control_i': 0.5,
            'control_d': 0.1
        }
        
        # Add simulation metrics to fitness
        if state['stats']['average_stability'] > 0.8:
            initial_params['servo_speed'] *= 1.2
        
        # Optimize
        result = swarm.optimize(initial_params, iterations=3)
        
        self.assertIsNotNone(result)
        self.assertIn('params', result)
    
    def test_hardware_to_simulation(self):
        """Test hardware data feeding into simulation"""
        # Create hardware controller
        hw = NaomiHardwareController(connection="Mock")
        hw.connect()
        
        # Wait for data
        time.sleep(0.2)
        
        # Get sensor data
        sensors = hw.read_all_sensors()
        
        # Create simulator
        sim = ShapeLogicSimulator(rows=6, cols=8, naomi_mode=True)
        
        # Use sensor data to affect simulation
        if sensors:
            panel_data = sensors.get(0, {})
            if panel_data:
                # Adjust simulation based on sensor data
                roll = panel_data.get('imu', {}).get('roll', 0)
                if abs(roll) > 5:
                    sim.randomize_edges(probability=0.5)
        
        sim.step()
        
        self.assertIsNotNone(sim.stats)
        
        hw.disconnect()


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestShapeLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmLords))
    suite.addTests(loader.loadTestsFromTestCase(TestHardwareControl))
    suite.addTests(loader.loadTestsFromTestCase(TestCADInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
