#!/usr/bin/env python
"""
Script to run tests for the AV Catalog Converter
"""
import os
import sys
import subprocess
import argparse
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Main function to run tests"""
    parser = argparse.ArgumentParser(description='Run tests for AV Catalog Converter')

    # Test selection options
    test_group = parser.add_argument_group('Test Selection')
    test_group.add_argument('--unit', action='store_true', help='Run unit tests only')
    test_group.add_argument('--integration', action='store_true', help='Run integration tests only')
    test_group.add_argument('--all', action='store_true', help='Run all tests')

    # Test filtering options
    filter_group = parser.add_argument_group('Test Filtering')
    filter_group.add_argument('--parallel', action='store_true', help='Run parallel processing tests only')
    filter_group.add_argument('--caching', action='store_true', help='Run caching tests only')
    filter_group.add_argument('--rate-limiting', dest='rate_limiting', action='store_true', help='Run rate limiting tests only')

    # Test execution options
    execution_group = parser.add_argument_group('Test Execution')
    execution_group.add_argument('--coverage', action='store_true', help='Generate coverage report')
    execution_group.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    execution_group.add_argument('--no-parallel', dest='no_parallel', action='store_true', help='Disable parallel test execution')
    execution_group.add_argument('--timeout', type=int, default=300, help='Test timeout in seconds (default: 300)')

    args = parser.parse_args()

    # Default to all tests if no specific test type is specified
    if not (args.unit or args.integration or args.all):
        args.all = True

    # Build the pytest command
    cmd = [sys.executable, '-m', 'pytest']

    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    else:
        cmd.append('-v')  # Always use verbose mode for better feedback

    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=.', '--cov-report=term', '--cov-report=html'])

    # Add parallel execution unless disabled
    if not args.no_parallel:
        cmd.extend(['-n', 'auto'])  # Use all available CPU cores

    # Add timeout to prevent hanging tests
    cmd.append(f'--timeout={args.timeout}')

    # Add test selection
    if args.all:
        cmd.append('tests/')
    elif args.unit:
        cmd.append('tests/unit/')
    elif args.integration:
        cmd.append('tests/integration/')

    # Add specific test markers if needed
    if args.parallel:
        cmd.append('-m parallel')
    elif args.caching:
        cmd.append('-m caching')
    elif args.rate_limiting:
        cmd.append('-m rate_limiting')

    # Run the tests
    logging.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            logging.info("All tests passed successfully!")
        else:
            logging.error(f"Tests failed with return code {result.returncode}")
        return result.returncode
    except Exception as e:
        logging.error(f"Error running tests: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
