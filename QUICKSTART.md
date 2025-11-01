# SwarmLords Enhanced - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Choose Your Version

**Option A: Standalone (No Dependencies)**
```bash
python swarm_standalone_test.py
```
✅ Works immediately  
✅ No setup required  
✅ Perfect for testing

**Option B: Full Integration (With Dependencies)**
```bash
python naomi_sol_pipeline.py --agents 12 --iterations 50
```
⚠️ Requires fixing imports in project files first

### Step 2: Run Your First Optimization

```bash
# Quick 50-iteration test
python swarm_standalone_test.py

# Expected output:
# - Fitness improves from ~14 to ~21
# - Parameters optimized for Naomi SOL
# - Results saved to JSON
```

### Step 3: Review Results

```bash
# View the optimized parameters
cat standalone_test_result.json
```

## 📊 What You'll See

```
SwarmLords Standalone Test
====================================

Running optimization...

Iteration 1: Best=15.138, Avg=14.022
Iteration 21: Best=21.107, Avg=18.560
Iteration 41: Best=21.444, Avg=18.803

RESULTS
====================================
Final Fitness: 21.444

Optimized Parameters:
  side_length         : 146.76
  thickness           : 5.02
  servo_pocket_depth  : 26.46
  mirror_diameter     : 72.83
  infill_percentage   : 22.87
  layer_height        : 0.22
  servo_speed         : 0.61
  sensor_gain         : 52.69
  control_p           : 1.99
  control_i           : 0.30
  control_d           : 0.48
```

## 🎯 Understanding the Results

### Key Metrics

- **Fitness Score**: Higher is better (0-30 range)
  - <15: Poor design
  - 15-20: Acceptable
  - 20-25: Good
  - >25: Excellent

- **Convergence**: Should see steady improvement
  - Fast initial gains (first 10 iterations)
  - Steady refinement (11-30)
  - Fine-tuning (31+)

### Optimized Parameters

| Parameter | Typical Value | Your Result | Status |
|-----------|--------------|-------------|---------|
| thickness | 4-6mm | 5.02mm | ✅ Optimal |
| side_length | 140-160mm | 146.76mm | ✅ Optimal |
| infill | 20-30% | 22.87% | ✅ Efficient |
| control_p | 1.5-2.5 | 1.99 | ✅ Well-tuned |

## 🔧 Next Steps

### 1. Tune for Your Needs

Edit the parameter bounds in the script:
```python
self.param_bounds = {
    'side_length': (140, 160),  # Adjust range
    'thickness': (4, 6),         # Adjust range
    # ... etc
}
```

### 2. Increase Quality

```bash
# More agents = better exploration
python swarm_standalone_test.py  # (edit num_agents to 24)

# More iterations = better convergence  
python swarm_standalone_test.py  # (edit iterations to 200)
```

### 3. Customize Fitness

Add your own scoring logic:
```python
def evaluate_fitness(self, position: np.ndarray) -> float:
    params = self._position_to_params(position)
    
    # Your custom criteria
    my_score = 0.0
    if params['thickness'] > 5:
        my_score += 10
    
    return my_score
```

## 📚 Full Documentation

For complete documentation, see:
- **README_INTEGRATION.md** - Full user guide
- **INTEGRATION_SUMMARY.md** - Implementation details
- **swarm_lords_enhanced.py** - Main system with inline docs

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Use `swarm_standalone_test.py` instead

### Issue: Low fitness scores
**Solution**: Increase iterations or agents

### Issue: No convergence
**Solution**: Check parameter bounds, increase exploration

### Issue: Out of memory
**Solution**: Reduce number of agents

## 💡 Pro Tips

1. **Start Small**: Test with 6-12 agents first
2. **Iterate**: Run multiple times with different seeds
3. **Save States**: Use JSON output to track progress
4. **Compare**: Run with different configurations
5. **Visualize**: Plot fitness over iterations

## 🎓 Learning Path

1. ✅ **Beginner**: Run standalone test
2. ⬜ **Intermediate**: Customize fitness function
3. ⬜ **Advanced**: Add new optimization algorithms
4. ⬜ **Expert**: Integrate with hardware

## 📞 Support

- Check **INTEGRATION_SUMMARY.md** for detailed info
- Review code comments in Python files
- Test changes with standalone version first

## ✨ What's Working Now

✅ Multi-agent swarm optimization  
✅ 11-parameter design space  
✅ Multi-objective fitness function  
✅ Adaptive agent behavior  
✅ Convergence tracking  
✅ Result serialization  
✅ Parallel evaluation  

## 🚧 What Needs Setup

⬜ GitHub code fetching (requires token + import fixes)  
⬜ Skills management (requires ace library)  
⬜ Physics simulation (requires fixed imports)  

## 🎯 Quick Examples

### Example 1: Quick Test
```bash
python swarm_standalone_test.py
# Takes ~1 second
# Produces JSON results
```

### Example 2: Higher Quality
```python
# Edit swarm_standalone_test.py:
swarm = SwarmLordsStandalone(num_agents=24)  # More agents
result = swarm.optimize(iterations=200)      # More iterations
# Takes ~5 seconds
```

### Example 3: Custom Objective
```python
# Add to evaluate_fitness method:
if params['mirror_diameter'] > 80:
    fitness += 5  # Prefer larger mirrors
```

## 📈 Expected Performance

| Agents | Iterations | Time | Quality |
|--------|-----------|------|---------|
| 6 | 50 | <1s | Good |
| 12 | 50 | 1s | Better |
| 24 | 100 | 3s | Excellent |
| 36 | 200 | 10s | Optimal |

## 🎉 Success Criteria

You've succeeded when you see:
- ✅ Fitness improving over iterations
- ✅ Parameters within sensible ranges
- ✅ Convergence stability
- ✅ JSON output generated

Now you're ready to optimize your Naomi SOL design! 🚀

---

**Quick Command Reference:**
```bash
# Run test
python swarm_standalone_test.py

# View results
cat standalone_test_result.json | python -m json.tool

# Check logs
tail -f *.log
```

Happy optimizing! 🎊
