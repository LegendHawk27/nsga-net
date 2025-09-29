#!/usr/bin/env python3
"""
Script to run NSGA-Net evolution search with reasonable parameters for your system
"""

import sys
import os
import subprocess
import argparse

# Add the project root to the path
sys.path.insert(0, r'D:\Research\nsga-net')

def run_evolution_search(search_space='micro', pop_size=8, n_gens=5, epochs=6, 
                        init_channels=24, layers=11, auxiliary=False, cutout=False,
                        use_synflow=True):
    """
    Run NSGA-Net evolution search with specified parameters
    
    Args:
        search_space: 'micro' or 'macro'
        pop_size: Population size (default: 20)
        n_gens: Number of generations (default: 10)
        epochs: Number of training epochs per architecture (default: 20)
        init_channels: Initial number of channels (default: 24)
        layers: Number of layers (default: 11)
        auxiliary: Use auxiliary head (default: False)
        cutout: Use cutout augmentation (default: False)
        use_synflow: Use SynFlow for early stopping (default: True)
    """
    
    print(f"Starting NSGA-Net evolution search...")
    print(f"Search space: {search_space}")
    print(f"Population size: {pop_size}")
    print(f"Generations: {n_gens}")
    print(f"Epochs per architecture: {epochs}")
    print(f"Initial channels: {init_channels}")
    print(f"Layers: {layers}")
    print(f"Auxiliary head: {auxiliary}")
    print(f"Cutout: {cutout}")
    print(f"SynFlow early stopping: {use_synflow}")
    print("=" * 60)
    
    cmd = [
        'python', 'search/evolution_search.py',
        '--search_space', search_space,
        '--pop_size', str(pop_size),
        '--n_gens', str(n_gens),
        '--epochs', str(epochs),
        '--init_channels', str(init_channels),
        '--layers', str(layers)
    ]
    
    if auxiliary:
        cmd.append('--auxiliary')
    if cutout:
        cmd.append('--cutout')
    if use_synflow:
        cmd.append('--use_synflow')
    else:
        cmd.append('--no_synflow')
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the evolution search
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        
        if process.returncode == 0:
            print("\n✓ Evolution search completed successfully!")
        else:
            print(f"\n✗ Evolution search failed with return code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠ Search interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"\n✗ Search failed with exception: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run NSGA-Net evolution search')
    parser.add_argument('--search_space', type=str, default='micro', 
                       choices=['micro', 'macro'], help='Search space type')
    parser.add_argument('--pop_size', type=int, default=8, help='Population size')
    parser.add_argument('--n_gens', type=int, default=5, help='Number of generations')
    parser.add_argument('--epochs', type=int, default=6, help='Epochs per architecture')
    parser.add_argument('--init_channels', type=int, default=24, help='Initial channels')
    parser.add_argument('--layers', type=int, default=11, help='Number of layers')
    parser.add_argument('--auxiliary', action='store_true', help='Use auxiliary head')
    parser.add_argument('--cutout', action='store_true', help='Use cutout augmentation')
    parser.add_argument('--use_synflow', action='store_true', default=True, 
                       help='Use SynFlow for early stopping')
    parser.add_argument('--no_synflow', action='store_true', 
                       help='Disable SynFlow early stopping')
    parser.add_argument('--quick', action='store_true', help='Quick test with minimal parameters')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists('search/evolution_search.py'):
        print("✗ Not in the correct directory!")
        print("Please run this script from the NSGA-Net root directory")
        return
    
    # Handle SynFlow flag
    use_synflow = args.use_synflow and not args.no_synflow
    
    if args.quick:
        print("Running quick test with minimal parameters...")
        run_evolution_search(pop_size=4, n_gens=2, epochs=3, use_synflow=use_synflow)
    else:
        run_evolution_search(
            search_space=args.search_space,
            pop_size=args.pop_size,
            n_gens=args.n_gens,
            epochs=args.epochs,
            init_channels=args.init_channels,
            layers=args.layers,
            auxiliary=args.auxiliary,
            cutout=args.cutout,
            use_synflow=use_synflow
        )

if __name__ == "__main__":
    main()
