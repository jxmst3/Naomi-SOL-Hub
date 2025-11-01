#!/usr/bin/env python3
"""
SwarmLords Enhanced - Integrated Multi-Agent AI System
=======================================================

Combines:
- SwarmLords distributed optimization
- GitHub code fetching and integration
- Skills management and execution
- Shape logic simulation
- Neural network learning

This creates a self-improving AI system where agents can:
1. Optimize servo configurations using swarm intelligence
2. Fetch and integrate code improvements from GitHub
3. Learn and execute skills dynamically
4. Simulate and validate designs before deployment
"""

import numpy as np
import logging
import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our components
import sys
sys.path.insert(0, '/mnt/project')

logger = logging.getLogger(__name__)

# Import from project root
try:
    from crew import CodeAgentCrew
    from github_code_fetcher import GitHubCodeFetcherTool
    from code_cherry_picker import CodeCherryPickerTool
except ImportError as e:
    logger.warning(f"Could not import crew components: {e}")
    CodeAgentCrew = None
    GitHubCodeFetcherTool = None
    CodeCherryPickerTool = None

# Try to import the skills manager
try:
    from skills_manager import SkillsManager
except ImportError:
    SkillsManager = None

# Try to import shape logic simulator
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "shape_logic", "/mnt/project/shape_logic1.py"
    )
    shape_logic_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shape_logic_module)
    ShapeLogicSimulator = shape_logic_module.ShapeLogicSimulator
except Exception as e:
    logger.warning(f"Could not import shape logic: {e}")
    ShapeLogicSimulator = None

# ==================== ENHANCED AGENT ====================

@dataclass
class EnhancedSwarmAgent:
    """
    Enhanced agent that combines servo control, learning, and code integration
    """
    agent_id: int
    servo_id: int
    panel_id: int
    
    # State
    position: np.ndarray = None  # Multi-dimensional state
    velocity: np.ndarray = None
    fitness: float = -np.inf
    best_position: np.ndarray = None
    best_fitness: float = -np.inf
    
    # Learning
    learning_rate: float = 0.1
    exploration_rate: float = 0.3
    aggression: float = 0.5
    
    # Skills and capabilities
    skills: List[str] = field(default_factory=list)
    code_improvements: List[Dict] = field(default_factory=list)
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Collaboration
    neighbors: List[int] = field(default_factory=list)
    shared_knowledge: Dict = field(default_factory=dict)
    
    # Simulation state
    simulation_score: float = 0.0
    validation_passed: bool = False
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.random.randn(11) * 0.1
        if self.velocity is None:
            self.velocity = np.zeros(11)
        if self.best_position is None:
            self.best_position = self.position.copy()
    
    def update_fitness(self, new_fitness: float):
        """Update fitness and personal best"""
        self.fitness = new_fitness
        self.performance_history.append(new_fitness)
        
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_position = self.position.copy()
            return True
        return False
    
    def get_performance_trend(self, window: int = 20) -> float:
        """Calculate performance trend"""
        if len(self.performance_history) < window:
            return 0.0
        recent = list(self.performance_history)[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / window
    
    def adapt_behavior(self):
        """Adapt agent behavior based on performance"""
        trend = self.get_performance_trend()
        
        if trend > 0:
            # Improving - reduce exploration
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
            self.aggression = min(1.0, self.aggression * 1.05)
        else:
            # Stagnating - increase exploration
            self.exploration_rate = min(0.5, self.exploration_rate * 1.05)
            self.aggression = max(0.1, self.aggression * 0.95)
    
    def to_dict(self) -> Dict:
        """Serialize agent state"""
        return {
            'agent_id': self.agent_id,
            'servo_id': self.servo_id,
            'panel_id': self.panel_id,
            'position': self.position.tolist() if self.position is not None else None,
            'fitness': float(self.fitness),
            'best_fitness': float(self.best_fitness),
            'learning_rate': float(self.learning_rate),
            'exploration_rate': float(self.exploration_rate),
            'skills': self.skills,
            'performance_trend': self.get_performance_trend()
        }


# ==================== ENHANCED SWARM LORDS ====================

class SwarmLordsEnhanced:
    """
    Enhanced SwarmLords system with integrated AI capabilities
    """
    
    def __init__(self,
                 num_agents: int = 36,
                 enable_code_fetching: bool = True,
                 enable_skills: bool = True,
                 enable_simulation: bool = True,
                 github_token: Optional[str] = None,
                 interactive: bool = False):
        
        self.num_agents = num_agents
        self.enable_code_fetching = enable_code_fetching
        self.enable_skills = enable_skills
        self.enable_simulation = enable_simulation
        self.interactive = interactive
        
        # Agents
        self.agents: Dict[int, EnhancedSwarmAgent] = {}
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # Integrated components
        self.code_crew = None
        self.skills_manager = None
        self.simulator = None
        
        # Optimization state
        self.iteration = 0
        self.convergence_history = []
        self.improvement_history = []
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Parameter bounds (from swarm1.py)
        self.param_bounds = {
            'side_length': (100, 200),
            'thickness': (2, 8),
            'servo_pocket_depth': (15, 30),
            'mirror_diameter': (50, 100),
            'infill_percentage': (20, 50),
            'layer_height': (0.1, 0.3),
            'servo_speed': (0.1, 1.0),
            'sensor_gain': (1, 100),
            'control_p': (0.1, 10.0),
            'control_i': (0.01, 1.0),
            'control_d': (0.01, 1.0),
        }
        
        self._initialize(github_token)
        
        logger.info(f"SwarmLordsEnhanced initialized: {num_agents} agents, "
                   f"code_fetching={enable_code_fetching}, "
                   f"skills={enable_skills}, "
                   f"simulation={enable_simulation}")
    
    def _initialize(self, github_token: Optional[str]):
        """Initialize all components"""
        # Initialize agents
        self._init_agents()
        
        # Initialize code crew
        if self.enable_code_fetching and CodeAgentCrew:
            try:
                self.code_crew = CodeAgentCrew(github_token=github_token)
                logger.info("Code fetching crew initialized")
            except Exception as e:
                logger.warning(f"Could not initialize code crew: {e}")
                self.code_crew = None
        else:
            self.code_crew = None
        
        # Initialize skills manager
        if self.enable_skills and SkillsManager:
            try:
                self.skills_manager = SkillsManager()
                logger.info("Skills manager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize skills manager: {e}")
                self.skills_manager = None
        
        # Initialize simulator
        if self.enable_simulation and ShapeLogicSimulator:
            try:
                self.simulator = ShapeLogicSimulator(
                    rows=12, 
                    cols=18, 
                    enable_physics=True,
                    naomi_mode=True
                )
                logger.info("Shape logic simulator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize simulator: {e}")
                self.simulator = None
    
    def _init_agents(self):
        """Initialize agent swarm"""
        servos_per_panel = 3
        num_panels = self.num_agents // servos_per_panel
        
        for i in range(self.num_agents):
            panel_id = i // servos_per_panel
            servo_id = i
            
            agent = EnhancedSwarmAgent(
                agent_id=i,
                servo_id=servo_id,
                panel_id=panel_id
            )
            
            # Assign initial skills
            agent.skills = ['optimize', 'coordinate', 'learn']
            
            self.agents[i] = agent
        
        # Establish communication network
        self._establish_network()
    
    def _establish_network(self):
        """Create agent communication network"""
        for agent_id, agent in self.agents.items():
            # Connect to agents on same panel
            panel_agents = [
                a_id for a_id, a in self.agents.items()
                if a.panel_id == agent.panel_id and a_id != agent_id
            ]
            agent.neighbors.extend(panel_agents)
            
            # Add cross-panel connections
            if agent_id % 3 == 0:  # First servo on each panel
                adjacent = (agent_id + 3) % self.num_agents
                if adjacent not in agent.neighbors:
                    agent.neighbors.append(adjacent)
    
    def evaluate_fitness(self, position: np.ndarray) -> float:
        """
        Multi-objective fitness evaluation
        """
        # Convert position to parameters
        params = self._position_to_params(position)
        
        fitness = 0.0
        
        # Structural fitness
        fitness += self._evaluate_structural(params) * 10
        
        # Efficiency fitness
        fitness += self._evaluate_efficiency(params) * 5
        
        # Control fitness
        fitness += self._evaluate_control(params) * 8
        
        # Simulation fitness (if enabled)
        if self.simulator:
            fitness += self._evaluate_simulation(params) * 7
        
        return fitness
    
    def _evaluate_structural(self, params: Dict) -> float:
        """Evaluate structural properties"""
        score = 0.0
        
        # Thickness (prefer 4-6mm)
        ideal_thickness = 5.0
        thickness_score = 1.0 - abs(params['thickness'] - ideal_thickness) / 3.0
        score += max(0, thickness_score)
        
        # Side length (prefer 140-160mm)
        ideal_length = 150
        length_score = 1.0 - abs(params['side_length'] - ideal_length) / 50
        score += max(0, length_score)
        
        # Servo pocket depth (must fit servo)
        if params['servo_pocket_depth'] >= 20:
            score += 1.0
        else:
            score -= 1.0
        
        return score / 3.0
    
    def _evaluate_efficiency(self, params: Dict) -> float:
        """Evaluate efficiency metrics"""
        score = 0.0
        
        # Weight efficiency
        weight_factor = params['thickness'] * params['infill_percentage'] / 100
        weight_score = 1.0 - (weight_factor / 4.0)
        score += max(0, weight_score)
        
        # Print time
        print_time = (params['thickness'] * params['infill_percentage'] / 
                     params['layer_height'])
        time_score = 1.0 / (1.0 + print_time / 100)
        score += time_score
        
        # Mirror effectiveness
        mirror_score = params['mirror_diameter'] / 100.0
        score += mirror_score
        
        return score / 3.0
    
    def _evaluate_control(self, params: Dict) -> float:
        """Evaluate control system properties"""
        score = 0.0
        
        # PID balance (Ziegler-Nichols-like)
        kp = params['control_p']
        ki = params['control_i']
        kd = params['control_d']
        
        # Ideal ratios
        ideal_ratio_i = kp / (2 * ki) if ki > 0 else 0
        ideal_ratio_d = kp * kd / 8
        
        balance_score = 1.0 / (1.0 + abs(ideal_ratio_i - 10) + abs(ideal_ratio_d - 1))
        score += balance_score
        
        # Servo speed (prefer moderate)
        speed_score = 1.0 - abs(params['servo_speed'] - 0.5) * 2
        score += max(0, speed_score)
        
        # Sensor gain (prefer moderate)
        gain_score = 1.0 - abs(np.log10(params['sensor_gain']) - 1) / 2
        score += max(0, gain_score)
        
        return score / 3.0
    
    def _evaluate_simulation(self, params: Dict) -> float:
        """Evaluate using shape logic simulation"""
        if not self.simulator:
            return 0.5
        
        try:
            # Run short simulation
            self.simulator.randomize_edges(probability=0.3)
            
            for _ in range(10):
                self.simulator.step()
            
            # Evaluate simulation results
            stats = self.simulator.stats
            
            score = 0.0
            score += stats['average_stability']
            score += min(1.0, stats['squares_formed'] / 20.0)
            score += min(1.0, stats['resonance_strength'])
            
            return score / 3.0
            
        except Exception as e:
            logger.debug(f"Simulation evaluation failed: {e}")
            return 0.5
    
    def _position_to_params(self, position: np.ndarray) -> Dict:
        """Convert position vector to parameters"""
        params = {}
        for i, (key, bounds) in enumerate(self.param_bounds.items()):
            # Normalize position to bounds
            normalized = (np.tanh(position[i]) + 1) / 2
            params[key] = bounds[0] + normalized * (bounds[1] - bounds[0])
        return params
    
    def optimize_step(self):
        """Execute one optimization step with swarm intelligence"""
        self.iteration += 1
        
        # Evaluate all agents in parallel
        futures = []
        for agent in self.agents.values():
            future = self.executor.submit(self.evaluate_fitness, agent.position)
            futures.append((agent, future))
        
        # Collect results
        for agent, future in futures:
            try:
                fitness = future.result(timeout=5)
                improved = agent.update_fitness(fitness)
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = agent.position.copy()
                    logger.info(f"New global best: {fitness:.3f} from agent {agent.agent_id}")
                
                # Adapt behavior
                if self.iteration % 10 == 0:
                    agent.adapt_behavior()
                
            except Exception as e:
                logger.warning(f"Agent {agent.agent_id} evaluation failed: {e}")
        
        # Update agent positions using PSO
        self._update_positions()
        
        # Record convergence
        avg_fitness = np.mean([a.fitness for a in self.agents.values()])
        self.convergence_history.append({
            'iteration': self.iteration,
            'global_best': float(self.global_best_fitness),
            'average': float(avg_fitness),
            'std': float(np.std([a.fitness for a in self.agents.values()]))
        })
    
    def _update_positions(self):
        """Update agent positions using particle swarm optimization"""
        w = 0.7  # Inertia
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social
        
        for agent in self.agents.values():
            # PSO velocity update
            r1 = np.random.random(len(agent.position))
            r2 = np.random.random(len(agent.position))
            
            cognitive = c1 * r1 * (agent.best_position - agent.position)
            social = c2 * r2 * (self.global_best_position - agent.position)
            
            agent.velocity = w * agent.velocity + cognitive + social
            
            # Exploration
            if np.random.random() < agent.exploration_rate:
                agent.velocity += np.random.randn(len(agent.position)) * 0.1
            
            # Update position
            agent.position += agent.velocity
            
            # Clip to reasonable bounds
            agent.position = np.clip(agent.position, -3, 3)
    
    def optimize(self, iterations: int = 100, 
                callback: Optional[Callable] = None) -> Dict:
        """
        Run full optimization process
        """
        logger.info(f"Starting optimization: {iterations} iterations")
        
        for i in range(iterations):
            self.optimize_step()
            
            if callback:
                callback(self.iteration, self.global_best_fitness)
            
            if i % 20 == 0:
                self._print_status()
            
            # Interactive mode
            if self.interactive and i % 50 == 0:
                self._interactive_check()
        
        # Get final result
        best_params = self._position_to_params(self.global_best_position)
        
        result = {
            'params': best_params,
            'fitness': float(self.global_best_fitness),
            'iterations': self.iteration,
            'convergence_history': self.convergence_history
        }
        
        logger.info(f"Optimization complete: Best fitness = {result['fitness']:.3f}")
        
        return result
    
    def _print_status(self):
        """Print optimization status"""
        if not self.convergence_history:
            return
        
        latest = self.convergence_history[-1]
        logger.info(
            f"Iteration {latest['iteration']}: "
            f"Best={latest['global_best']:.3f}, "
            f"Avg={latest['average']:.3f}, "
            f"Std={latest['std']:.3f}"
        )
    
    def _interactive_check(self):
        """Interactive mode checkpoint"""
        if not self.interactive:
            return
        
        best_params = self._position_to_params(self.global_best_position)
        
        print("\n" + "="*60)
        print("OPTIMIZATION CHECKPOINT")
        print("="*60)
        print(f"Iteration: {self.iteration}")
        print(f"Best Fitness: {self.global_best_fitness:.3f}")
        print("\nCurrent Best Parameters:")
        for key, value in best_params.items():
            print(f"  {key:20s}: {value:8.2f}")
        print("\nContinue? [Y/n/save/improve]")
        
        try:
            response = input("> ").strip().lower()
            
            if response == 'n':
                logger.info("User stopped optimization")
                raise KeyboardInterrupt()
            elif response == 'save':
                self.save_state()
            elif response == 'improve':
                self.fetch_code_improvements()
        except (EOFError, KeyboardInterrupt):
            pass
    
    def fetch_code_improvements(self):
        """Fetch code improvements from GitHub"""
        if not self.code_crew:
            logger.warning("Code fetching not enabled")
            return
        
        logger.info("Fetching code improvements from GitHub...")
        
        # Search for relevant optimization code
        queries = [
            "particle swarm optimization python",
            "servo control optimization",
            "multi-agent reinforcement learning"
        ]
        
        improvements = []
        for query in queries:
            try:
                snippets = self.code_crew.search_and_fetch(query, max_results=3)
                improvements.extend(snippets)
            except Exception as e:
                logger.warning(f"Failed to fetch for '{query}': {e}")
        
        logger.info(f"Found {len(improvements)} potential improvements")
        
        # Store improvements for review
        for snippet in improvements[:5]:
            self.agents[0].code_improvements.append({
                'source': f"{snippet.get('owner')}/{snippet.get('repo')}",
                'path': snippet.get('path'),
                'score': snippet.get('score_hint', 0)
            })
    
    def execute_skill(self, skill_name: str, params: Dict) -> Dict:
        """Execute a skill through the skills manager"""
        if not self.skills_manager:
            logger.warning("Skills manager not available")
            return {}
        
        try:
            result = self.skills_manager.call_skill(skill_name, params)
            logger.info(f"Executed skill '{skill_name}': {result}")
            return result
        except Exception as e:
            logger.error(f"Skill execution failed: {e}")
            return {}
    
    def get_agent_status(self, agent_id: int) -> Dict:
        """Get detailed agent status"""
        if agent_id not in self.agents:
            return {}
        
        agent = self.agents[agent_id]
        return agent.to_dict()
    
    def save_state(self, filepath: Optional[str] = None):
        """Save complete system state"""
        if filepath is None:
            filepath = f"swarm_state_{int(time.time())}.json"
        
        state = {
            'iteration': self.iteration,
            'num_agents': self.num_agents,
            'global_best_fitness': float(self.global_best_fitness),
            'global_best_position': self.global_best_position.tolist(),
            'agents': {
                agent_id: agent.to_dict()
                for agent_id, agent in self.agents.items()
            },
            'convergence_history': self.convergence_history,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load system state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.iteration = state['iteration']
        self.global_best_fitness = state['global_best_fitness']
        self.global_best_position = np.array(state['global_best_position'])
        self.convergence_history = state['convergence_history']
        
        logger.info(f"State loaded from {filepath}")
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down SwarmLordsEnhanced")
        self.executor.shutdown(wait=True)


# ==================== TESTING & DEMONSTRATION ====================

def test_enhanced_swarm():
    """Test the enhanced swarm system"""
    print("="*60)
    print("SwarmLords Enhanced - Integration Test")
    print("="*60)
    
    # Create enhanced swarm
    swarm = SwarmLordsEnhanced(
        num_agents=12,  # Small test swarm
        enable_code_fetching=False,  # Disable for testing
        enable_skills=False,
        enable_simulation=True,
        interactive=False
    )
    
    # Run optimization
    print("\nRunning optimization...")
    result = swarm.optimize(iterations=50)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Final Fitness: {result['fitness']:.3f}")
    print("\nOptimized Parameters:")
    for key, value in result['params'].items():
        print(f"  {key:20s}: {value:8.2f}")
    
    # Save results
    swarm.save_state("test_swarm_state.json")
    
    # Show convergence
    if result['convergence_history']:
        print("\nConvergence History (last 5):")
        for entry in result['convergence_history'][-5:]:
            print(f"  Iter {entry['iteration']:3d}: "
                  f"Best={entry['global_best']:.3f}, "
                  f"Avg={entry['average']:.3f}")
    
    # Shutdown
    swarm.shutdown()
    
    print("\nTest complete!")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    test_enhanced_swarm()
