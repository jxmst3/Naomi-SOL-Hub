#!/usr/bin/env python3
"""
SwarmLords Enhanced - Standalone Test
======================================

Simplified version for testing without external dependencies.
"""

import numpy as np
import logging
import json
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedSwarmAgent:
    """Enhanced agent for swarm optimization"""
    agent_id: int
    servo_id: int
    panel_id: int
    
    position: np.ndarray = None
    velocity: np.ndarray = None
    fitness: float = -np.inf
    best_position: np.ndarray = None
    best_fitness: float = -np.inf
    
    learning_rate: float = 0.1
    exploration_rate: float = 0.3
    aggression: float = 0.5
    
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    neighbors: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.random.randn(11) * 0.1
        if self.velocity is None:
            self.velocity = np.zeros(11)
        if self.best_position is None:
            self.best_position = self.position.copy()
    
    def update_fitness(self, new_fitness: float):
        self.fitness = new_fitness
        self.performance_history.append(new_fitness)
        
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_position = self.position.copy()
            return True
        return False
    
    def get_performance_trend(self, window: int = 20) -> float:
        if len(self.performance_history) < window:
            return 0.0
        recent = list(self.performance_history)[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / window
    
    def adapt_behavior(self):
        trend = self.get_performance_trend()
        
        if trend > 0:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
            self.aggression = min(1.0, self.aggression * 1.05)
        else:
            self.exploration_rate = min(0.5, self.exploration_rate * 1.05)
            self.aggression = max(0.1, self.aggression * 0.95)


class SwarmLordsStandalone:
    """
    Standalone swarm optimizer (no external dependencies)
    """
    
    def __init__(self, num_agents: int = 12):
        self.num_agents = num_agents
        self.agents: Dict[int, EnhancedSwarmAgent] = {}
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        self.iteration = 0
        self.convergence_history = []
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Parameter bounds
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
        
        self._init_agents()
        logger.info(f"SwarmLordsStandalone initialized: {num_agents} agents")
    
    def _init_agents(self):
        servos_per_panel = 3
        
        for i in range(self.num_agents):
            panel_id = i // servos_per_panel
            servo_id = i
            
            agent = EnhancedSwarmAgent(
                agent_id=i,
                servo_id=servo_id,
                panel_id=panel_id
            )
            
            self.agents[i] = agent
        
        # Establish network
        for agent_id, agent in self.agents.items():
            panel_agents = [
                a_id for a_id, a in self.agents.items()
                if a.panel_id == agent.panel_id and a_id != agent_id
            ]
            agent.neighbors.extend(panel_agents)
    
    def evaluate_fitness(self, position: np.ndarray) -> float:
        """Multi-objective fitness evaluation"""
        params = self._position_to_params(position)
        
        fitness = 0.0
        
        # Structural fitness
        thickness_score = 1.0 - abs(params['thickness'] - 5.0) / 3.0
        fitness += max(0, thickness_score) * 10
        
        # Efficiency fitness
        weight_factor = params['thickness'] * params['infill_percentage'] / 100
        weight_score = 1.0 - (weight_factor / 4.0)
        fitness += max(0, weight_score) * 5
        
        # Control fitness
        kp = params['control_p']
        balance_score = 1.0 / (1.0 + abs(kp - 2.0))
        fitness += balance_score * 8
        
        return fitness
    
    def _position_to_params(self, position: np.ndarray) -> Dict:
        params = {}
        for i, (key, bounds) in enumerate(self.param_bounds.items()):
            normalized = (np.tanh(position[i]) + 1) / 2
            params[key] = bounds[0] + normalized * (bounds[1] - bounds[0])
        return params
    
    def optimize_step(self):
        """Execute one optimization step"""
        self.iteration += 1
        
        # Evaluate agents
        for agent in self.agents.values():
            fitness = self.evaluate_fitness(agent.position)
            improved = agent.update_fitness(fitness)
            
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = agent.position.copy()
                logger.info(f"New global best: {fitness:.3f} from agent {agent.agent_id}")
            
            if self.iteration % 10 == 0:
                agent.adapt_behavior()
        
        # Update positions (PSO)
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
        """PSO update"""
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        for agent in self.agents.values():
            r1 = np.random.random(len(agent.position))
            r2 = np.random.random(len(agent.position))
            
            cognitive = c1 * r1 * (agent.best_position - agent.position)
            social = c2 * r2 * (self.global_best_position - agent.position)
            
            agent.velocity = w * agent.velocity + cognitive + social
            
            if np.random.random() < agent.exploration_rate:
                agent.velocity += np.random.randn(len(agent.position)) * 0.1
            
            agent.position += agent.velocity
            agent.position = np.clip(agent.position, -3, 3)
    
    def optimize(self, iterations: int = 100) -> Dict:
        """Run full optimization"""
        logger.info(f"Starting optimization: {iterations} iterations")
        
        for i in range(iterations):
            self.optimize_step()
            
            if i % 20 == 0:
                latest = self.convergence_history[-1]
                logger.info(
                    f"Iteration {latest['iteration']}: "
                    f"Best={latest['global_best']:.3f}, "
                    f"Avg={latest['average']:.3f}"
                )
        
        best_params = self._position_to_params(self.global_best_position)
        
        result = {
            'params': best_params,
            'fitness': float(self.global_best_fitness),
            'iterations': self.iteration,
            'convergence_history': self.convergence_history
        }
        
        logger.info(f"Optimization complete: Best fitness = {result['fitness']:.3f}")
        return result
    
    def shutdown(self):
        """Shutdown"""
        self.executor.shutdown(wait=True)


def test_standalone():
    """Test the standalone swarm"""
    print("="*60)
    print("SwarmLords Standalone Test")
    print("="*60)
    
    swarm = SwarmLordsStandalone(num_agents=12)
    
    print("\nRunning optimization...")
    result = swarm.optimize(iterations=50)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final Fitness: {result['fitness']:.3f}")
    print("\nOptimized Parameters:")
    for key, value in result['params'].items():
        print(f"  {key:20s}: {value:8.2f}")
    
    # Save results
    import os
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'standalone_test_result.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    swarm.shutdown()
    print("\nTest complete!")


if __name__ == '__main__':
    test_standalone()
