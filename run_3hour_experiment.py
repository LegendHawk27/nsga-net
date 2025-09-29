#!/usr/bin/env python3
"""
3-Hour NSGA-Net Experiment Configuration
Optimized for maximum efficiency within 3 hours
"""

import sys
import os
import subprocess
import time

# Add the project root to the path
sys.path.insert(0, r'D:\Research\nsga-net')

def run_3hour_experiment():
    """
    Run optimized NSGA-Net experiment designed to complete in ~3 hours
    """
    
    print("=" * 60)
    print("🚀 3-HOUR NSGA-NET EXPERIMENT")
    print("=" * 60)
    
    # Optimized parameters for 3-hour run
    config = {
        'search_space': 'micro',
        'pop_size': 8,           # 8 initial networks
        'n_gens': 5,             # 5 generations
        'n_offspring': 4,        # 4 offspring per generation
        'epochs': 6,             # 6 epochs per network (optimized)
        'init_channels': 24,
        'layers': 11,
        'use_synflow': True,     # Early stopping enabled
    }
    
    # Calculate total networks and estimated time
    total_networks = config['pop_size'] + (config['n_gens'] * config['n_offspring'])
    estimated_time_per_network = 3.5  # minutes (with optimizations)
    estimated_total_time = (total_networks * estimated_time_per_network) / 60  # hours
    
    print(f"📊 EXPERIMENT CONFIGURATION:")
    print(f"   Search Space: {config['search_space']}")
    print(f"   Population Size: {config['pop_size']}")
    print(f"   Generations: {config['n_gens']}")
    print(f"   Offspring per Generation: {config['n_offspring']}")
    print(f"   Epochs per Network: {config['epochs']}")
    print(f"   Total Networks: {total_networks}")
    print(f"   Estimated Time: {estimated_total_time:.1f} hours")
    print(f"   SynFlow Early Stopping: {config['use_synflow']}")
    print("=" * 60)
    
    # Build command
    cmd = [
        'python', 'search/evolution_search.py',
        '--search_space', config['search_space'],
        '--pop_size', str(config['pop_size']),
        '--n_gens', str(config['n_gens']),
        '--n_offspring', str(config['n_offspring']),
        '--epochs', str(config['epochs']),
        '--init_channels', str(config['init_channels']),
        '--layers', str(config['layers']),
        '--use_synflow'
    ]
    
    print(f"🔧 COMMAND: {' '.join(cmd)}")
    print("=" * 60)
    
    # Record start time
    start_time = time.time()
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Starting experiment...")
    print("=" * 60)
    
    try:
        # Run the experiment
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        
        # Calculate total time
        end_time = time.time()
        total_runtime = (end_time - start_time) / 3600  # hours
        
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total Runtime: {total_runtime:.2f} hours")
            print(f"🎯 Target: 3 hours")
            print(f"📈 Efficiency: {3.0/total_runtime:.1f}x faster than target")
            print("=" * 60)
        else:
            print(f"\n❌ Experiment failed with return code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"\n❌ Experiment failed with exception: {e}")

def main():
    """Main function"""
    # Check if we're in the right directory
    if not os.path.exists('search/evolution_search.py'):
        print("❌ Not in the correct directory!")
        print("Please run this script from the NSGA-Net root directory")
        return
    
    # Confirm with user
    print("This will start a 3-hour NSGA-Net experiment.")
    response = input("Continue? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_3hour_experiment()
    else:
        print("Experiment cancelled.")

if __name__ == "__main__":
    main()

