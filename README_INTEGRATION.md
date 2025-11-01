# SwarmLords Enhanced - Integrated AI System

## Overview

This integration combines multiple AI components into a unified system for optimizing the Naomi SOL dodecahedron design:

1. **SwarmLords Multi-Agent Optimization** - Distributed swarm intelligence
2. **GitHub Code Fetching** - Automatic code improvement discovery
3. **Skills Management** - Dynamic AI capability execution
4. **Physics Simulation** - Shape logic validation
5. **Neural Network Learning** - Adaptive fitness prediction

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  SwarmLordsEnhanced                     │
│                                                         │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Swarm     │  │   Code       │  │   Skills      │  │
│  │  Agents    │◄─┤   Fetcher    │◄─┤   Manager     │  │
│  │  (36)      │  │   (GitHub)   │  │   (Dynamic)   │  │
│  └────┬───────┘  └──────────────┘  └───────────────┘  │
│       │                                                 │
│       ▼                                                 │
│  ┌────────────────────────────────────────────┐        │
│  │     Particle Swarm Optimization (PSO)      │        │
│  │  • Cognitive component (personal best)     │        │
│  │  • Social component (global best)          │        │
│  │  • Adaptive exploration/exploitation       │        │
│  └────────────┬───────────────────────────────┘        │
│               │                                         │
│               ▼                                         │
│  ┌────────────────────────────────────────────┐        │
│  │       Multi-Objective Fitness              │        │
│  │  • Structural strength                     │        │
│  │  • Manufacturing efficiency                │        │
│  │  • Control system performance              │        │
│  │  • Physics simulation validation           │        │
│  └────────────┬───────────────────────────────┘        │
│               │                                         │
│               ▼                                         │
│  ┌────────────────────────────────────────────┐        │
│  │      Shape Logic Simulator                 │        │
│  │  • Edge polarity dynamics                  │        │
│  │  • Square formation rules                  │        │
│  │  • Energy diffusion                        │        │
│  │  • Naomi SOL specific features             │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Enhanced Swarm Agents

Each agent maintains:
- **Position** in 11-dimensional parameter space
- **Velocity** for PSO movement
- **Learning parameters** (exploration rate, aggression, learning rate)
- **Skills** and code improvements
- **Performance history** for adaptive behavior

### 2. Multi-Objective Fitness Function

The fitness function evaluates designs across multiple criteria:

- **Structural (30%)**: Thickness, panel dimensions, servo pocket depth
- **Efficiency (20%)**: Weight, print time, mirror effectiveness  
- **Control (30%)**: PID balance, servo speed, sensor gain
- **Simulation (20%)**: Physics validation with shape logic

### 3. Code Integration from GitHub

Agents can autonomously fetch relevant code improvements:
- Search queries: "particle swarm optimization", "servo control", etc.
- Automatic integration with code cherry picker
- Score-based ranking of improvements

### 4. Physics Simulation Validation

Real-time validation using Shape Logic Simulator:
- Edge polarity dynamics
- Square formation with energy constraints
- Stability and resonance metrics
- Naomi SOL specific panel behavior

### 5. Adaptive Learning

Agents adapt their behavior based on performance:
- Increase exploration when stagnating
- Increase exploitation when improving
- Dynamic learning rate adjustment
- Behavioral parameter evolution

## Installation

```bash
# Install dependencies
pip install numpy scipy
pip install torch torchvision  # Optional: for neural networks
pip install matplotlib  # Optional: for visualization

# Clone project
git clone <repository-url>
cd naomi-sol-hub
```

## Quick Start

### Basic Optimization

```python
from swarm_lords_enhanced import SwarmLordsEnhanced

# Create swarm
swarm = SwarmLordsEnhanced(
    num_agents=12,
    enable_code_fetching=False,
    enable_simulation=True,
    interactive=False
)

# Run optimization
result = swarm.optimize(iterations=100)

# Get best design
print(f"Best fitness: {result['fitness']:.3f}")
print(f"Parameters: {result['params']}")

# Shutdown
swarm.shutdown()
```

### Full Pipeline

```bash
# Run complete pipeline with 36 agents
python naomi_sol_pipeline.py --agents 36 --iterations 200

# Interactive mode with GitHub integration
python naomi_sol_pipeline.py --interactive --github --github-token YOUR_TOKEN

# Quick test run
python naomi_sol_pipeline.py --agents 12 --iterations 50
```

## Configuration

### Swarm Parameters

```python
swarm = SwarmLordsEnhanced(
    num_agents=36,              # Number of optimization agents
    enable_code_fetching=True,  # Fetch improvements from GitHub
    enable_skills=True,         # Enable skills manager
    enable_simulation=True,     # Enable physics simulation
    github_token="...",         # GitHub API token
    interactive=False           # Interactive mode
)
```

### Optimization Parameters

The system optimizes 11 design parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| side_length | 100-200mm | Pentagon panel side length |
| thickness | 2-8mm | Panel thickness |
| servo_pocket_depth | 15-30mm | Depth for servo mounting |
| mirror_diameter | 50-100mm | Mirror size |
| infill_percentage | 20-50% | 3D print infill density |
| layer_height | 0.1-0.3mm | 3D print layer height |
| servo_speed | 0.1-1.0 | Servo movement speed |
| sensor_gain | 1-100 | Sensor amplification |
| control_p | 0.1-10.0 | PID proportional term |
| control_i | 0.01-1.0 | PID integral term |
| control_d | 0.01-1.0 | PID derivative term |

## Pipeline Stages

### Stage 1: Initial Optimization

- Run swarm optimization for N/2 iterations
- Establish baseline fitness
- Record initial convergence

### Stage 2: Code Improvements (Optional)

- Fetch relevant code from GitHub
- Analyze and rank improvements
- Integrate top candidates

### Stage 3: Refined Optimization

- Continue optimization with improvements
- Run for N/2 more iterations
- Compare with baseline

### Stage 4: Simulation Validation

- Run physics simulation
- Validate stability and energy
- Check square formation

### Stage 5: Final Report

- Generate comprehensive report
- Save optimization history
- Create manufacturing specifications
- Request approval (if interactive)

## Output Files

The pipeline generates several output files:

```
naomi_sol_outputs/
├── report_<timestamp>.json         # Complete optimization report
├── swarm_state_<timestamp>.json    # Swarm agent states
├── manufacturing_specs.json        # Build specifications
└── optimization_history.json       # Convergence data
```

### Report Structure

```json
{
  "timestamp": 1234567890,
  "pipeline_config": {
    "num_agents": 36,
    "total_iterations": 200
  },
  "final_design": {
    "side_length": 150.5,
    "thickness": 5.2,
    ...
  },
  "final_fitness": 87.3,
  "optimization_history": [...],
  "stage_results": {
    "stage1": {...},
    "stage2": {...},
    ...
  }
}
```

## Advanced Usage

### Custom Fitness Function

```python
def custom_fitness(position):
    """Define custom fitness evaluation"""
    params = swarm._position_to_params(position)
    
    # Your custom scoring logic
    score = 0.0
    score += evaluate_strength(params)
    score += evaluate_aesthetics(params)
    
    return score

# Use custom function
swarm.evaluate_fitness = custom_fitness
```

### Skills Execution

```python
# Execute a skill
result = swarm.execute_skill('optimize_pid', {
    'target_response': 0.5,
    'overshoot_limit': 0.1
})
```

### Agent Inspection

```python
# Get detailed agent status
status = swarm.get_agent_status(agent_id=0)

print(f"Agent {status['agent_id']}")
print(f"  Panel: {status['panel_id']}")
print(f"  Fitness: {status['best_fitness']:.3f}")
print(f"  Trend: {status['fitness_trend']:.3f}")
print(f"  Exploration: {status['exploration_rate']:.2f}")
```

### State Management

```python
# Save current state
swarm.save_state('my_optimization.json')

# Load previous state
swarm.load_state('my_optimization.json')

# Continue optimization
result = swarm.optimize(iterations=50)
```

## Integration with Existing Code

### With Original SwarmLords

```python
# Enhanced version is backward compatible
from swarm_lords_enhanced import SwarmLordsEnhanced as SwarmLords

# Use exactly like the original
swarm = SwarmLords(num_agents=36)
```

### With Code Agent Crew

```python
# Access the code crew directly
if swarm.code_crew:
    # Search for specific improvements
    snippets = swarm.code_crew.search_and_fetch(
        "servo optimization algorithm",
        max_results=5
    )
    
    # Integrate into target file
    for snippet in snippets:
        swarm.code_crew.integrate('my_file.py', snippet)
```

### With Skills Manager

```python
# Access skills manager
if swarm.skills_manager:
    # List available skills
    skills = swarm.skills_manager.skills.keys()
    
    # Execute skill
    result = swarm.skills_manager.call_skill(
        'trajectory_planning',
        {'start': [0, 0], 'end': [10, 10]}
    )
```

## Performance Tuning

### For Faster Optimization

```python
swarm = SwarmLordsEnhanced(
    num_agents=12,              # Fewer agents
    enable_code_fetching=False, # Disable GitHub
    enable_simulation=False,    # Disable physics sim
    interactive=False
)
```

### For Better Quality

```python
swarm = SwarmLordsEnhanced(
    num_agents=50,              # More agents
    enable_code_fetching=True,  # Enable improvements
    enable_simulation=True,     # Enable validation
    interactive=True            # Manual review
)

result = swarm.optimize(iterations=500)  # More iterations
```

### Parallel Execution

The system automatically uses ThreadPoolExecutor for parallel fitness evaluation:

```python
# Fitness evaluations run in parallel across 8 threads
# No configuration needed - automatic
```

## Troubleshooting

### Low Fitness Scores

- Increase number of agents
- Run more iterations
- Check parameter bounds
- Enable simulation validation

### Slow Convergence

- Adjust PSO parameters (w, c1, c2)
- Increase agent exploration rates
- Use adaptive behavior
- Try different initial positions

### GitHub Rate Limiting

- Provide GitHub token for authentication
- Reduce code fetch frequency
- Cache fetched improvements

### Memory Issues

- Reduce number of agents
- Disable performance history tracking
- Limit convergence history storage
- Use smaller simulation grid

## Examples

### Example 1: Quick Test

```python
from swarm_lords_enhanced import SwarmLordsEnhanced

swarm = SwarmLordsEnhanced(num_agents=6, enable_simulation=False)
result = swarm.optimize(iterations=20)
print(f"Test fitness: {result['fitness']:.2f}")
swarm.shutdown()
```

### Example 2: Full Production Run

```bash
python naomi_sol_pipeline.py \
    --agents 36 \
    --iterations 500 \
    --github \
    --github-token $GITHUB_TOKEN \
    --interactive
```

### Example 3: Comparison Study

```python
# Run multiple configurations
configs = [
    {'agents': 12, 'iterations': 100},
    {'agents': 24, 'iterations': 100},
    {'agents': 36, 'iterations': 100},
]

results = []
for config in configs:
    swarm = SwarmLordsEnhanced(num_agents=config['agents'])
    result = swarm.optimize(iterations=config['iterations'])
    results.append(result)
    swarm.shutdown()

# Compare results
for i, result in enumerate(results):
    print(f"Config {i+1}: fitness = {result['fitness']:.3f}")
```

## Contributing

To add new features:

1. **New Fitness Criteria**: Add evaluation function in `_evaluate_*` methods
2. **New Skills**: Add skill .md files to `ai/skills/` directory
3. **New Optimization Algorithms**: Extend `_update_positions` method
4. **New Simulations**: Integrate with `simulator` attribute

## License

See project LICENSE file.

## Credits

Integrated from:
- `swarm.py` - Basic swarm optimization
- `swarm1.py` - Advanced ML integration
- `swarm_lords.py` - Distributed agents
- `crew.py` - Code fetching system
- `shape_logic1.py` - Physics simulation
- `skills_manager.py` - Dynamic capabilities

## Support

For issues or questions:
- Check troubleshooting section
- Review example code
- Check log files in `naomi_sol_outputs/`
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
