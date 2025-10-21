# ai/swarm.py
"""
SwarmLords Multi-Agent Optimizer for Naomi SOL
===============================================
Advanced optimization using swarm intelligence with interactive mode,
pre-trained model support, and training history tracking.
"""

import json
import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed - using basic optimization")

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger("SwarmLords")


@dataclass
class Agent:
    """Individual agent in the swarm"""
    id: int
    position: np.ndarray  # Current design parameters
    velocity: np.ndarray  # Rate of change
    fitness: float = 0.0
    best_position: np.ndarray = None
    best_fitness: float = -np.inf
    exploration_rate: float = 0.1
    
    def __post_init__(self):
        if self.best_position is None:
            self.best_position = self.position.copy()


class SwarmLordsController:
    """
    Advanced multi-agent optimization controller.
    Uses swarm intelligence to optimize Naomi SOL design parameters.
    """
    
    # Parameter bounds for Naomi SOL
    PARAM_BOUNDS = {
        'side_length': (100, 200),      # Pentagon side in mm
        'thickness': (2, 8),             # Panel thickness in mm
        'servo_pocket_depth': (15, 30),  # Servo pocket depth
        'mirror_diameter': (50, 100),    # Mirror size
        'infill_percentage': (20, 50),   # 3D print infill
        'layer_height': (0.1, 0.3),      # Print layer height
        'servo_speed': (0.1, 1.0),       # Servo movement speed
        'sensor_gain': (1, 100),         # Sensor amplification
        'control_p': (0.1, 10.0),        # PID P term
        'control_i': (0.01, 1.0),        # PID I term
        'control_d': (0.01, 1.0),        # PID D term
    }
    
    def __init__(self, agent_count: int = 10,
                 interactive: bool = False,
                 use_pretrained: bool = False,
                 model_path: Optional[str] = None):
        """
        Initialize SwarmLords controller.
        
        Args:
            agent_count: Number of agents in swarm
            interactive: Enable interactive optimization
            use_pretrained: Load pre-trained model
            model_path: Path to pre-trained model
        """
        self.agent_count = agent_count
        self.interactive = interactive
        self.use_pretrained = use_pretrained
        
        # Agents
        self.agents: List[Agent] = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # Neural network model
        self.model = None
        self.model_type = "torch" if TORCH_AVAILABLE else "basic"
        
        # Training history
        self.history = {
            'iterations': [],
            'best_fitness': [],
            'average_fitness': [],
            'designs': [],
            'timestamps': []
        }
        
        # Optimization parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        
        # Interactive mode state
        self.pending_approval = None
        self.user_feedback = []
        
        # Paths
        self.model_dir = Path("ai/training_data")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize swarm and models"""
        # Initialize agents
        self._init_agents()
        
        # Load or create model
        if self.use_pretrained:
            self._load_model()
        else:
            self._create_model()
        
        logger.info(f"SwarmLords initialized: {self.agent_count} agents, "
                   f"model={self.model_type}, interactive={self.interactive}")
    
    def _init_agents(self):
        """Initialize agent population"""
        self.agents = []
        
        # Parameter dimensions
        self.n_params = len(self.PARAM_BOUNDS)
        
        for i in range(self.agent_count):
            # Random initial position
            position = np.zeros(self.n_params)
            for j, (param, bounds) in enumerate(self.PARAM_BOUNDS.items()):
                position[j] = np.random.uniform(bounds[0], bounds[1])
            
            # Random initial velocity
            velocity = np.random.randn(self.n_params) * 0.1
            
            agent = Agent(
                id=i,
                position=position,
                velocity=velocity,
                exploration_rate=0.1 + 0.2 * np.random.random()
            )
            
            self.agents.append(agent)
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def _create_model(self):
        """Create neural network model for fitness prediction"""
        if TORCH_AVAILABLE:
            self._create_torch_model()
        elif TENSORFLOW_AVAILABLE:
            self._create_keras_model()
        else:
            logger.info("No ML framework available - using basic optimization")
            self.model = None
    
    def _create_torch_model(self):
        """Create PyTorch model"""
        class FitnessNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 16)
                self.fc4 = nn.Linear(16, 1)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
                return x
        
        self.model = FitnessNet(self.n_params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model_type = "torch"
        
        logger.info("Created PyTorch fitness model")
    
    def _create_keras_model(self):
        """Create Keras/TensorFlow model"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.n_params,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        self.model_type = "keras"
        
        logger.info("Created Keras fitness model")
    
    def _load_model(self):
        """Load pre-trained model"""
        model_path = self.model_dir / "pretrained_model.pth"
        keras_path = self.model_dir / "pretrained_model.h5"
        
        if TORCH_AVAILABLE and model_path.exists():
            self._create_torch_model()
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded PyTorch model from {model_path}")
            
        elif TENSORFLOW_AVAILABLE and keras_path.exists():
            self.model = keras.models.load_model(keras_path)
            self.model_type = "keras"
            logger.info(f"Loaded Keras model from {keras_path}")
            
        else:
            logger.info("No pre-trained model found - creating new model")
            self._create_model()
    
    def _save_model(self):
        """Save trained model"""
        if self.model is None:
            return
        
        if self.model_type == "torch" and TORCH_AVAILABLE:
            model_path = self.model_dir / "pretrained_model.pth"
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved PyTorch model to {model_path}")
            
        elif self.model_type == "keras" and TENSORFLOW_AVAILABLE:
            model_path = self.model_dir / "pretrained_model.h5"
            self.model.save(model_path)
            logger.info(f"Saved Keras model to {model_path}")
    
    def evaluate_fitness(self, position: np.ndarray) -> float:
        """
        Evaluate fitness of a design.
        
        Args:
            position: Parameter vector
            
        Returns:
            Fitness score
        """
        # Extract parameters
        params = self._position_to_params(position)
        
        # Multi-objective fitness function
        fitness = 0.0
        
        # Structural strength (prefer thicker panels)
        thickness_score = params['thickness'] / 8.0
        fitness += thickness_score * 10
        
        # Weight efficiency (prefer lighter)
        weight_score = 1.0 - (params['infill_percentage'] / 100.0)
        fitness += weight_score * 5
        
        # Print time (prefer faster)
        print_time = params['thickness'] * params['infill_percentage'] / params['layer_height']
        print_score = 100.0 / (print_time + 1)
        fitness += print_score * 3
        
        # Mirror effectiveness (larger is better)
        mirror_score = params['mirror_diameter'] / 100.0
        fitness += mirror_score * 8
        
        # Control responsiveness
        control_score = params['control_p'] * params['servo_speed']
        fitness += control_score * 2
        
        # Sensor sensitivity
        sensor_score = math.log(params['sensor_gain']) / math.log(100)
        fitness += sensor_score * 2
        
        # Penalties
        # Avoid extreme values
        for param, value in params.items():
            bounds = self.PARAM_BOUNDS[param]
            range_size = bounds[1] - bounds[0]
            normalized = (value - bounds[0]) / range_size
            
            # Penalty for being too close to bounds
            if normalized < 0.1 or normalized > 0.9:
                fitness -= 5
        
        # Special constraints for Naomi SOL
        # Pentagon must be large enough for components
        if params['side_length'] < 120:
            fitness -= 20
        
        # Servo pocket must fit servo
        if params['servo_pocket_depth'] < 20:
            fitness -= 15
        
        # Use neural network if available
        if self.model:
            nn_score = self._predict_fitness(position)
            fitness = 0.7 * fitness + 0.3 * nn_score
        
        return fitness
    
    def _predict_fitness(self, position: np.ndarray) -> float:
        """Use neural network to predict fitness"""
        if self.model is None:
            return 0.0
        
        if self.model_type == "torch" and TORCH_AVAILABLE:
            with torch.no_grad():
                x = torch.FloatTensor(position).unsqueeze(0)
                pred = self.model(x).item()
                return pred
                
        elif self.model_type == "keras" and TENSORFLOW_AVAILABLE:
            pred = self.model.predict(position.reshape(1, -1), verbose=0)[0, 0]
            return float(pred)
        
        return 0.0
    
    def _position_to_params(self, position: np.ndarray) -> Dict:
        """Convert position vector to parameter dictionary"""
        params = {}
        for i, param_name in enumerate(self.PARAM_BOUNDS.keys()):
            params[param_name] = position[i]
        return params
    
    def _params_to_position(self, params: Dict) -> np.ndarray:
        """Convert parameter dictionary to position vector"""
        position = np.zeros(self.n_params)
        for i, param_name in enumerate(self.PARAM_BOUNDS.keys()):
            position[i] = params.get(param_name, 0)
        return position
    
    def step(self):
        """Perform one optimization step"""
        # Evaluate fitness for all agents
        for agent in self.agents:
            fitness = self.evaluate_fitness(agent.position)
            agent.fitness = fitness
            
            # Update personal best
            if fitness > agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = agent.position.copy()
            
            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = agent.position.copy()
        
        # Update velocities and positions
        for agent in self.agents:
            # Random coefficients
            r1 = np.random.random(self.n_params)
            r2 = np.random.random(self.n_params)
            
            # Velocity update (PSO equation)
            cognitive = self.c1 * r1 * (agent.best_position - agent.position)
            social = self.c2 * r2 * (self.global_best_position - agent.position)
            
            agent.velocity = (self.w * agent.velocity + 
                             cognitive + social)
            
            # Add exploration noise
            if np.random.random() < agent.exploration_rate:
                agent.velocity += np.random.randn(self.n_params) * 0.1
            
            # Update position
            agent.position += agent.velocity
            
            # Enforce bounds
            for i, (param, bounds) in enumerate(self.PARAM_BOUNDS.items()):
                agent.position[i] = np.clip(agent.position[i], bounds[0], bounds[1])
        
        # Adaptive parameters
        self.w *= 0.99  # Decrease inertia over time
        self.w = max(0.4, self.w)
    
    def optimize(self, initial_params: Optional[Dict] = None,
                iterations: int = 100,
                callback: Optional[Callable] = None) -> Dict:
        """
        Run optimization.
        
        Args:
            initial_params: Starting parameters
            iterations: Number of iterations
            callback: Function to call each iteration
            
        Returns:
            Optimized parameters
        """
        logger.info(f"Starting optimization: {iterations} iterations")
        
        # Set initial position if provided
        if initial_params:
            initial_pos = self._params_to_position(initial_params)
            self.agents[0].position = initial_pos
        
        # Optimization loop
        for iteration in range(iterations):
            # Step optimization
            self.step()
            
            # Record history
            avg_fitness = np.mean([a.fitness for a in self.agents])
            self.history['iterations'].append(iteration)
            self.history['best_fitness'].append(self.global_best_fitness)
            self.history['average_fitness'].append(avg_fitness)
            self.history['timestamps'].append(time.time())
            
            # Interactive mode
            if self.interactive and iteration % 10 == 0:
                self._interactive_check()
            
            # Progress update
            if iteration % 20 == 0:
                logger.info(f"Iteration {iteration}: "
                          f"Best={self.global_best_fitness:.2f}, "
                          f"Avg={avg_fitness:.2f}")
            
            # Callback
            if callback:
                callback(iteration, self.global_best_fitness)
        
        # Final result
        best_params = self._position_to_params(self.global_best_position)
        
        # Save best design
        self.history['designs'].append({
            'params': best_params,
            'fitness': self.global_best_fitness,
            'timestamp': time.time()
        })
        
        # Train model on results
        if self.model:
            self._train_model()
        
        # Save history
        self._save_history()
        
        # Save model
        if self.model:
            self._save_model()
        
        logger.info(f"Optimization complete: Best fitness = {self.global_best_fitness:.2f}")
        
        return {
            'params': best_params,
            'fitness': self.global_best_fitness,
            'improvement': self.global_best_fitness / (initial_params.get('fitness', 1) if initial_params else 1)
        }
    
    def _interactive_check(self):
        """Interactive mode - ask user for feedback"""
        if not self.interactive:
            return
        
        # Get current best design
        best_params = self._position_to_params(self.global_best_position)
        
        print("\n" + "="*50)
        print("SWARM OPTIMIZATION PROPOSAL")
        print("="*50)
        print(f"Fitness Score: {self.global_best_fitness:.2f}")
        print("\nProposed Design Parameters:")
        
        for param, value in best_params.items():
            print(f"  {param:20s}: {value:8.2f}")
        
        print("\n[A]ccept, [R]eject, [S]kip, [C]ontinue without asking?")
        
        try:
            response = input("> ").strip().lower()
            
            if response == 'a':
                logger.info("User accepted proposal")
                self.user_feedback.append({
                    'params': best_params,
                    'fitness': self.global_best_fitness,
                    'feedback': 'accept'
                })
            elif response == 'r':
                logger.info("User rejected proposal")
                # Penalize this design
                self.global_best_fitness *= 0.5
                self.user_feedback.append({
                    'params': best_params,
                    'fitness': self.global_best_fitness,
                    'feedback': 'reject'
                })
            elif response == 'c':
                self.interactive = False
                logger.info("Continuing without interaction")
            else:
                logger.info("Skipping feedback")
                
        except KeyboardInterrupt:
            self.interactive = False
            logger.info("Disabling interactive mode")
    
    def _train_model(self):
        """Train neural network on collected data"""
        if not self.model or len(self.history['designs']) < 10:
            return
        
        logger.info("Training fitness model...")
        
        # Prepare training data
        X = []
        y = []
        
        for agent in self.agents:
            X.append(agent.position)
            y.append(agent.fitness)
        
        X = np.array(X)
        y = np.array(y)
        
        if self.model_type == "torch" and TORCH_AVAILABLE:
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            # Training loop
            self.model.train()
            for epoch in range(50):
                self.optimizer.zero_grad()
                predictions = self.model(X_tensor)
                loss = self.criterion(predictions, y_tensor)
                loss.backward()
                self.optimizer.step()
            
            logger.info(f"Model trained, final loss: {loss.item():.4f}")
            
        elif self.model_type == "keras" and TENSORFLOW_AVAILABLE:
            # Train Keras model
            self.model.fit(X, y, epochs=50, batch_size=4, verbose=0)
            loss = self.model.evaluate(X, y, verbose=0)
            logger.info(f"Model trained, final loss: {loss:.4f}")
    
    def _save_history(self):
        """Save optimization history"""
        history_file = self.model_dir / "optimization_history.json"
        
        with open(history_file, 'w') as f:
            # Convert numpy arrays to lists for JSON
            history_serializable = {
                k: v if not isinstance(v[0] if v else None, np.ndarray) else
                   [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
                for k, v in self.history.items()
            }
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"Saved optimization history to {history_file}")
    
    def load_history(self, filepath: str):
        """Load optimization history"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        logger.info(f"Loaded history from {filepath}")
    
    def get_best_design(self) -> Dict:
        """Get current best design"""
        if self.global_best_position is None:
            return {}
        
        return {
            'params': self._position_to_params(self.global_best_position),
            'fitness': self.global_best_fitness
        }
    
    def show_history(self):
        """Display optimization history"""
        if not self.history['iterations']:
            print("No history available")
            return
        
        print("\n" + "="*60)
        print("OPTIMIZATION HISTORY")
        print("="*60)
        
        print(f"Total iterations: {len(self.history['iterations'])}")
        print(f"Best fitness achieved: {max(self.history['best_fitness']):.2f}")
        print(f"Final average fitness: {self.history['average_fitness'][-1]:.2f}")
        
        # Plot if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Fitness over time
            ax1.plot(self.history['iterations'], 
                    self.history['best_fitness'], 
                    'b-', label='Best Fitness')
            ax1.plot(self.history['iterations'], 
                    self.history['average_fitness'], 
                    'r--', label='Average Fitness')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Fitness')
            ax1.set_title('SwarmLords Optimization Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Agent positions
            if self.agents:
                positions = np.array([a.position for a in self.agents])
                ax2.imshow(positions.T, aspect='auto', cmap='viridis')
                ax2.set_xlabel('Agent')
                ax2.set_ylabel('Parameter')
                ax2.set_title('Agent Parameter Values')
                
                # Add parameter names
                param_names = list(self.PARAM_BOUNDS.keys())
                ax2.set_yticks(range(len(param_names)))
                ax2.set_yticklabels(param_names)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("\n(Install matplotlib to see graphs)")


def test_swarm():
    """Test the SwarmLords optimizer"""
    print("Testing SwarmLords Optimizer...")
    
    # Create controller
    swarm = SwarmLordsController(
        agent_count=5,
        interactive=False,
        use_pretrained=False
    )
    
    # Initial parameters
    initial = {
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
    
    # Run optimization
    print("\nStarting optimization...")
    result = swarm.optimize(
        initial_params=initial,
        iterations=50
    )
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULT")
    print("="*50)
    print(f"Improvement: {result['improvement']:.1%}")
    print("\nOptimized Parameters:")
    for param, value in result['params'].items():
        print(f"  {param:20s}: {value:8.2f}")
    
    # Show history
    swarm.show_history()
    
    print("\nSwarmLords test complete!")


if __name__ == "__main__":
    test_swarm()
