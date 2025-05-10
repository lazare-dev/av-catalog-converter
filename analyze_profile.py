#!/usr/bin/env python
"""
Script to analyze profiling results
"""
import os
import sys
import argparse
import pstats
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def analyze_profile(profile_path: Path, output_dir: Path, limit: int = 20) -> None:
    """
    Analyze a profile file and generate reports
    
    Args:
        profile_path: Path to the profile file
        output_dir: Directory to save the reports
        limit: Maximum number of functions to include in the reports
    """
    print(f"Analyzing profile: {profile_path}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load the profile
    stats = pstats.Stats(str(profile_path))
    
    # Generate text report
    text_report_path = output_dir / f"{profile_path.stem}_report.txt"
    with open(text_report_path, 'w') as f:
        stats_stream = stats.stream
        stats.stream = f
        stats.sort_stats('cumulative').print_stats(limit)
        stats.sort_stats('time').print_stats(limit)
        stats.sort_stats('calls').print_stats(limit)
        stats.stream = stats_stream
    
    print(f"Text report saved to: {text_report_path}")
    
    # Extract data for visualization
    function_stats = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        module_name = func[0]
        line_number = func[1]
        function_name = func[2]
        
        # Skip built-in functions
        if module_name == '~' or 'built-in' in module_name:
            continue
        
        function_stats.append({
            'module': module_name,
            'line': line_number,
            'function': function_name,
            'calls': nc,
            'total_time': tt,
            'cumulative_time': ct,
            'time_per_call': tt / nc if nc > 0 else 0
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(function_stats)
    
    if df.empty:
        print("No function statistics found in the profile")
        return
    
    # Sort by cumulative time
    df = df.sort_values('cumulative_time', ascending=False).head(limit)
    
    # Save to CSV
    csv_path = output_dir / f"{profile_path.stem}_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to: {csv_path}")
    
    # Generate visualizations
    try:
        # Time distribution
        plt.figure(figsize=(12, 8))
        plt.barh(df['function'], df['cumulative_time'], label='Cumulative Time')
        plt.barh(df['function'], df['total_time'], label='Total Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Function')
        plt.title(f'Time Distribution - {profile_path.stem}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{profile_path.stem}_time.png")
        
        # Calls distribution
        plt.figure(figsize=(12, 8))
        plt.barh(df['function'], df['calls'])
        plt.xlabel('Number of Calls')
        plt.ylabel('Function')
        plt.title(f'Call Distribution - {profile_path.stem}')
        plt.tight_layout()
        plt.savefig(output_dir / f"{profile_path.stem}_calls.png")
        
        # Time per call
        plt.figure(figsize=(12, 8))
        plt.barh(df['function'], df['time_per_call'])
        plt.xlabel('Time per Call (seconds)')
        plt.ylabel('Function')
        plt.title(f'Time per Call - {profile_path.stem}')
        plt.tight_layout()
        plt.savefig(output_dir / f"{profile_path.stem}_time_per_call.png")
        
        print(f"Visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze profiling results')
    parser.add_argument('--profile', '-p', help='Path to the profile file')
    parser.add_argument('--dir', '-d', help='Directory containing profile files')
    parser.add_argument('--output', '-o', default='profiling/reports', help='Output directory for reports')
    parser.add_argument('--limit', '-l', type=int, default=20, help='Maximum number of functions to include in reports')
    args = parser.parse_args()
    
    if not args.profile and not args.dir:
        parser.error("Either --profile or --dir must be specified")
    
    output_dir = Path(args.output)
    
    if args.profile:
        profile_path = Path(args.profile)
        if not profile_path.exists():
            print(f"Profile file not found: {profile_path}")
            return 1
        
        analyze_profile(profile_path, output_dir, args.limit)
    
    if args.dir:
        profile_dir = Path(args.dir)
        if not profile_dir.exists() or not profile_dir.is_dir():
            print(f"Profile directory not found: {profile_dir}")
            return 1
        
        profile_files = list(profile_dir.glob('*.prof'))
        if not profile_files:
            print(f"No profile files found in: {profile_dir}")
            return 1
        
        for profile_path in profile_files:
            analyze_profile(profile_path, output_dir, args.limit)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
