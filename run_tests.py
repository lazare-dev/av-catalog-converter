#!/usr/bin/env python
"""
Script to run tests for the AV Catalog Converter
"""
import os
import sys
import subprocess
import argparse


def main():
    """Main function to run tests"""
    parser = argparse.ArgumentParser(description='Run tests for AV Catalog Converter')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Default to all tests if no specific test type is specified
    if not (args.unit or args.integration or args.all):
        args.all = True
    
    # Build the pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=.', '--cov-report=term', '--cov-report=html'])
    
    # Add test selection
    if args.all:
        cmd.append('tests/')
    elif args.unit:
        cmd.append('tests/unit/')
    elif args.integration:
        cmd.append('tests/integration/')
    
    # Run the tests
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
