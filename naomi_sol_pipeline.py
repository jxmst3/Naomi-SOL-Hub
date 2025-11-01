#!/usr/bin/env python3
"""
Naomi SOL Optimization Pipeline
================================

Complete demonstration of the integrated SwarmLords system for optimizing
the Naomi SOL dodecahedron design. Shows:

1. Multi-agent swarm optimization
2. Code improvement fetching from GitHub
3. Skills-based enhancement
4. Physics simulation validation
5. Interactive tuning and approval

Usage:
    python naomi_sol_pipeline.py --agents 36 --iterations 200 --interactive
"""

import argparse
import logging
import json
import time
import sys
from pathlib import Path
from typing import Dict, List

# Add project to path
sys.path.insert(0, '/mnt/project')
sys.path.insert(0, '/home/claude')

from swarm_lords_enhanced import SwarmLordsEnhanced, EnhancedSwarmAgent

logger = logging.getLogger(__name__)


class NaomiSOLPipeline:
    """
    Complete optimization pipeline for Naomi SOL
    """
    
    def __init__(self, 
                 num_agents: int = 36,
                 enable_github: bool = False,
                 enable_simulation: bool = True,
                 interactive: bool = False,
                 github_token: str = None):
        
        self.num_agents = num_agents
        self.enable_github = enable_github
        self.enable_simulation = enable_simulation
        self.interactive = interactive
        
        logger.info("Initializing Naomi SOL Optimization Pipeline")
        
        # Create enhanced swarm
        self.swarm = SwarmLordsEnhanced(
            num_agents=num_agents,
            enable_code_fetching=enable_github,
            enable_skills=True,
            enable_simulation=enable_simulation,
            github_token=github_token,
            interactive=interactive
        )
        
        # Pipeline state
        self.pipeline_stage = "initialized"
        self.results_history = []
        self.approved_designs = []
        
        # Output directory
        self.output_dir = Path("naomi_sol_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized: {num_agents} agents")
    
    def run_full_pipeline(self, iterations: int = 100) -> Dict:
        """
        Execute the complete optimization pipeline
        """
        print("\n" + "="*70)
        print(" NAOMI SOL OPTIMIZATION PIPELINE ".center(70, "="))
        print("="*70)
        
        results = {}
        
        # Stage 1: Initial optimization
        print("\n[Stage 1/5] Initial Swarm Optimization")
        print("-" * 70)
        results['stage1'] = self._stage1_initial_optimization(iterations // 2)
        
        # Stage 2: Code improvement (if enabled)
        if self.enable_github:
            print("\n[Stage 2/5] Fetching Code Improvements")
            print("-" * 70)
            results['stage2'] = self._stage2_code_improvements()
        else:
            print("\n[Stage 2/5] Code improvements disabled, skipping")
            results['stage2'] = {'skipped': True}
        
        # Stage 3: Refined optimization
        print("\n[Stage 3/5] Refined Optimization")
        print("-" * 70)
        results['stage3'] = self._stage3_refined_optimization(iterations // 2)
        
        # Stage 4: Simulation validation
        if self.enable_simulation:
            print("\n[Stage 4/5] Physics Simulation Validation")
            print("-" * 70)
            results['stage4'] = self._stage4_simulation_validation()
        else:
            print("\n[Stage 4/5] Simulation disabled, skipping")
            results['stage4'] = {'skipped': True}
        
        # Stage 5: Final report
        print("\n[Stage 5/5] Generating Final Report")
        print("-" * 70)
        results['stage5'] = self._stage5_final_report(results)
        
        self.pipeline_stage = "complete"
        
        return results
    
    def _stage1_initial_optimization(self, iterations: int) -> Dict:
        """
        Stage 1: Run initial swarm optimization
        """
        logger.info(f"Stage 1: Initial optimization ({iterations} iterations)")
        
        start_time = time.time()
        
        # Run optimization
        result = self.swarm.optimize(
            iterations=iterations,
            callback=self._optimization_callback
        )
        
        duration = time.time() - start_time
        
        # Extract best design
        best_design = {
            'params': result['params'],
            'fitness': result['fitness'],
            'stage': 'initial',
            'duration': duration
        }
        
        self.results_history.append(best_design)
        
        print(f"\n✓ Initial optimization complete")
        print(f"  Best fitness: {result['fitness']:.3f}")
        print(f"  Duration: {duration:.1f}s")
        
        return best_design
    
    def _stage2_code_improvements(self) -> Dict:
        """
        Stage 2: Fetch and analyze code improvements from GitHub
        """
        logger.info("Stage 2: Fetching code improvements")
        
        try:
            self.swarm.fetch_code_improvements()
            
            # Analyze improvements
            improvements = []
            for agent in list(self.swarm.agents.values())[:3]:
                improvements.extend(agent.code_improvements)
            
            print(f"\n✓ Found {len(improvements)} code improvements")
            
            if improvements:
                print("\nTop improvements:")
                for i, imp in enumerate(improvements[:5], 1):
                    print(f"  {i}. {imp['source']} (score: {imp.get('score', 0):.2f})")
            
            return {
                'improvements_found': len(improvements),
                'top_improvements': improvements[:5]
            }
            
        except Exception as e:
            logger.error(f"Code improvement fetching failed: {e}")
            return {'error': str(e)}
    
    def _stage3_refined_optimization(self, iterations: int) -> Dict:
        """
        Stage 3: Run refined optimization with learned improvements
        """
        logger.info(f"Stage 3: Refined optimization ({iterations} iterations)")
        
        start_time = time.time()
        
        # Continue optimization from current state
        result = self.swarm.optimize(
            iterations=iterations,
            callback=self._optimization_callback
        )
        
        duration = time.time() - start_time
        
        # Compare with stage 1
        improvement = 0.0
        if self.results_history:
            prev_fitness = self.results_history[-1]['fitness']
            improvement = ((result['fitness'] - prev_fitness) / prev_fitness) * 100
        
        refined_design = {
            'params': result['params'],
            'fitness': result['fitness'],
            'stage': 'refined',
            'duration': duration,
            'improvement_pct': improvement
        }
        
        self.results_history.append(refined_design)
        
        print(f"\n✓ Refined optimization complete")
        print(f"  Best fitness: {result['fitness']:.3f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  Duration: {duration:.1f}s")
        
        return refined_design
    
    def _stage4_simulation_validation(self) -> Dict:
        """
        Stage 4: Validate designs using physics simulation
        """
        logger.info("Stage 4: Simulation validation")
        
        if not self.swarm.simulator:
            return {'error': 'Simulator not available'}
        
        try:
            # Get current best design
            best_params = self.swarm._position_to_params(
                self.swarm.global_best_position
            )
            
            # Run extended simulation
            print("  Running physics simulation...")
            self.swarm.simulator.randomize_edges(probability=0.4)
            
            stability_scores = []
            energy_scores = []
            
            for step in range(50):
                self.swarm.simulator.step()
                
                if step % 10 == 0:
                    stats = self.swarm.simulator.stats
                    stability_scores.append(stats['average_stability'])
                    energy_scores.append(stats['total_energy'])
            
            # Calculate validation metrics
            avg_stability = sum(stability_scores) / len(stability_scores)
            avg_energy = sum(energy_scores) / len(energy_scores)
            
            validation_passed = (
                avg_stability > 0.6 and
                avg_energy > 10.0 and
                self.swarm.simulator.stats['squares_formed'] > 5
            )
            
            print(f"\n✓ Simulation validation complete")
            print(f"  Average stability: {avg_stability:.3f}")
            print(f"  Average energy: {avg_energy:.1f}")
            print(f"  Squares formed: {self.swarm.simulator.stats['squares_formed']}")
            print(f"  Status: {'PASSED' if validation_passed else 'FAILED'}")
            
            return {
                'avg_stability': avg_stability,
                'avg_energy': avg_energy,
                'squares_formed': self.swarm.simulator.stats['squares_formed'],
                'passed': validation_passed
            }
            
        except Exception as e:
            logger.error(f"Simulation validation failed: {e}")
            return {'error': str(e)}
    
    def _stage5_final_report(self, all_results: Dict) -> Dict:
        """
        Stage 5: Generate comprehensive final report
        """
        logger.info("Stage 5: Generating final report")
        
        # Get final best design
        final_design = self.swarm._position_to_params(
            self.swarm.global_best_position
        )
        
        # Compile report
        report = {
            'timestamp': time.time(),
            'pipeline_config': {
                'num_agents': self.num_agents,
                'github_enabled': self.enable_github,
                'simulation_enabled': self.enable_simulation,
                'total_iterations': self.swarm.iteration
            },
            'final_design': final_design,
            'final_fitness': float(self.swarm.global_best_fitness),
            'optimization_history': self.swarm.convergence_history[-20:],
            'stage_results': all_results
        }
        
        # Save report
        report_file = self.output_dir / f"report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save swarm state
        state_file = self.output_dir / f"swarm_state_{int(time.time())}.json"
        self.swarm.save_state(str(state_file))
        
        # Print summary
        print("\n" + "="*70)
        print(" OPTIMIZATION COMPLETE ".center(70, "="))
        print("="*70)
        print(f"\nFinal Fitness: {report['final_fitness']:.3f}")
        print(f"Total Iterations: {report['pipeline_config']['total_iterations']}")
        
        print("\nOptimized Design Parameters:")
        print("-" * 70)
        for key, value in final_design.items():
            print(f"  {key:25s}: {value:8.2f}")
        
        print(f"\n✓ Reports saved to:")
        print(f"  - {report_file}")
        print(f"  - {state_file}")
        
        # Interactive approval
        if self.interactive:
            self._request_approval(report)
        
        return report
    
    def _optimization_callback(self, iteration: int, fitness: float):
        """Callback during optimization"""
        if iteration % 20 == 0:
            print(f"  Iteration {iteration:4d}: fitness = {fitness:.3f}")
    
    def _request_approval(self, report: Dict):
        """Request user approval of design"""
        print("\n" + "="*70)
        print("Design Approval Request")
        print("="*70)
        print("\nWould you like to approve this design?")
        print("[A]pprove, [R]eject, [S]ave only")
        
        try:
            response = input("> ").strip().lower()
            
            if response == 'a':
                self.approved_designs.append(report)
                print("✓ Design approved and saved")
            elif response == 'r':
                print("Design rejected")
            else:
                print("Design saved for review")
        except (EOFError, KeyboardInterrupt):
            print("\nApproval skipped")
    
    def generate_manufacturing_specs(self) -> Dict:
        """
        Generate manufacturing specifications from optimized design
        """
        if not self.results_history:
            logger.error("No optimization results available")
            return {}
        
        best_design = self.results_history[-1]
        params = best_design['params']
        
        specs = {
            '3d_printing': {
                'layer_height': f"{params['layer_height']:.2f}mm",
                'infill': f"{params['infill_percentage']:.0f}%",
                'material': 'PLA or PETG',
                'supports': 'Required for servo pockets'
            },
            'panel_dimensions': {
                'side_length': f"{params['side_length']:.1f}mm",
                'thickness': f"{params['thickness']:.1f}mm",
                'servo_pocket_depth': f"{params['servo_pocket_depth']:.1f}mm"
            },
            'mirrors': {
                'diameter': f"{params['mirror_diameter']:.1f}mm",
                'type': 'First surface mirror',
                'mounting': '3M adhesive backing'
            },
            'servos': {
                'type': 'SG90 micro servo',
                'quantity': 36,
                'speed_setting': f"{params['servo_speed']:.2f}"
            },
            'control_system': {
                'pid_parameters': {
                    'P': f"{params['control_p']:.2f}",
                    'I': f"{params['control_i']:.3f}",
                    'D': f"{params['control_d']:.3f}"
                },
                'sensor_gain': f"{params['sensor_gain']:.0f}"
            }
        }
        
        # Save specs
        specs_file = self.output_dir / "manufacturing_specs.json"
        with open(specs_file, 'w') as f:
            json.dump(specs, f, indent=2)
        
        print("\n" + "="*70)
        print(" MANUFACTURING SPECIFICATIONS ".center(70, "="))
        print("="*70)
        
        for category, items in specs.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for key, value in items.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n✓ Specifications saved to: {specs_file}")
        
        return specs
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down pipeline")
        self.swarm.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Naomi SOL Optimization Pipeline'
    )
    parser.add_argument(
        '--agents', type=int, default=12,
        help='Number of swarm agents (default: 12)'
    )
    parser.add_argument(
        '--iterations', type=int, default=100,
        help='Total optimization iterations (default: 100)'
    )
    parser.add_argument(
        '--interactive', action='store_true',
        help='Enable interactive mode'
    )
    parser.add_argument(
        '--github', action='store_true',
        help='Enable GitHub code fetching'
    )
    parser.add_argument(
        '--no-simulation', action='store_true',
        help='Disable physics simulation'
    )
    parser.add_argument(
        '--github-token', type=str,
        help='GitHub personal access token'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('naomi_sol_pipeline.log')
        ]
    )
    
    try:
        # Create pipeline
        pipeline = NaomiSOLPipeline(
            num_agents=args.agents,
            enable_github=args.github,
            enable_simulation=not args.no_simulation,
            interactive=args.interactive,
            github_token=args.github_token
        )
        
        # Run optimization
        results = pipeline.run_full_pipeline(iterations=args.iterations)
        
        # Generate manufacturing specs
        specs = pipeline.generate_manufacturing_specs()
        
        # Shutdown
        pipeline.shutdown()
        
        print("\n" + "="*70)
        print("Pipeline execution complete!")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
