#!/usr/bin/env python3
"""
6-Hour NSGA-Net Experiment Configuration
Optimized for maximum efficiency within 6 hours
Based on current run timing analysis: ~10-12 minutes per network
"""

import sys
import os
import subprocess
import time

# Add the project root to the path
sys.path.insert(0, r'D:\Research\nsga-net')

def run_6hour_experiment():
    """
    Run optimized NSGA-Net experiment designed to complete in ~6 hours
    Based on timing analysis: 10-12 minutes per network
    """
    
    print("=" * 60)
    print("🚀 6-HOUR NSGA-NET EXPERIMENT")
    print("=" * 60)
    
    # Optimized parameters for 6-hour run (360 minutes)
    # Target: ~30 networks total to stay within 6 hours (30 × 12 min = 360 min)
    config = {
        'search_space': 'micro',
        'pop_size': 8,           # 8 initial networks
        'n_gens': 4,             # 4 generations  
        'n_offspring': 6,        # 6 offspring per generation
        'epochs': 3,             # 3 epochs per network (same as current)
        'init_channels': 24,
        'layers': 11,
        'use_synflow': True,     # Early stopping enabled
    }
    
    # Calculate total networks and estimated time
    # SynFlow keeps top 50% of population each generation
    total_networks = config['pop_size'] + (config['n_gens'] * (config['n_offspring'] // 2))
    estimated_time_per_network = 11  # minutes (conservative estimate)
    estimated_total_time = (total_networks * estimated_time_per_network) / 60  # hours
    
    print(f"📊 EXPERIMENT CONFIGURATION:")
    print(f"   Search Space: {config['search_space']}")
    print(f"   Population Size: {config['pop_size']}")
    print(f"   Generations: {config['n_gens']}")
    print(f"   Offspring per Generation: {config['n_offspring']}")
    print(f"   Epochs per Network: {config['epochs']}")
    print(f"   Total Networks (estimated): {total_networks}")
    print(f"   Estimated Time: {estimated_total_time:.1f} hours")
    print(f"   Target: 6.0 hours")
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
    print("🚀 Starting 6-hour experiment...")
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
            print(f"🎯 Target: 6.0 hours")
            print(f"📈 Efficiency: {6.0/total_runtime:.1f}x of target time")
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
    print("This will start a 6-hour NSGA-Net experiment.")
    print("Based on current run analysis:")
    print("- Each network takes ~10-12 minutes")
    print("- SynFlow prefiltering keeps top 50% each generation")
    print("- Estimated completion: ~6 hours")
    print()
    response = input("Continue? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_6hour_experiment()
    else:
        print("Experiment cancelled.")

if __name__ == "__main__":
    main()

