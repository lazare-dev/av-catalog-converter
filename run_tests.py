#!/usr/bin/env python
"""
Script to run tests for the AV Catalog Converter
"""
import os
import sys
import subprocess
import argparse
import logging
import importlib.util
import platform

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def check_environment():
    """Check if the environment is properly set up for testing"""
    # Check Python version
    python_version = platform.python_version()
    logging.info(f"Python version: {python_version}")

    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    logging.info(f"Running in virtual environment: {in_venv}")

    # Check NumPy version
    try:
        import numpy
        numpy_version = numpy.__version__
        logging.info(f"NumPy version: {numpy_version}")

        # Check if NumPy version is compatible
        if numpy_version.startswith('2.'):
            logging.warning("NumPy 2.x detected. This may cause compatibility issues with pandas 1.5.3.")
            logging.warning("Consider using NumPy 1.24.3 as specified in requirements.txt")

            # Ask user if they want to continue
            if not os.environ.get('CONTINUE_WITH_NUMPY2', ''):
                response = input("Continue with NumPy 2.x? (y/n): ").lower()
                if response != 'y':
                    logging.info("Exiting. Please install NumPy 1.24.3 and try again.")
                    return False
    except ImportError:
        logging.warning("NumPy not found. Some tests may fail.")

    # Check pandas version
    try:
        import pandas
        pandas_version = pandas.__version__
        logging.info(f"pandas version: {pandas_version}")

        # Check if pandas version is compatible
        if not pandas_version.startswith('1.5.'):
            logging.warning(f"pandas {pandas_version} detected. This may cause compatibility issues.")
            logging.warning("Consider using pandas 1.5.3 as specified in requirements.txt")
    except ImportError:
        logging.warning("pandas not found. Some tests may fail.")

    # Check if pytest is installed
    if importlib.util.find_spec("pytest") is None:
        logging.error("pytest is not installed. Please install it with 'pip install pytest'")
        return False

    return True

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
    execution_group.add_argument('--skip-env-check', action='store_true', help='Skip environment compatibility check')
    execution_group.add_argument('--force', '-f', action='store_true', help='Force test execution even if environment check fails')

    args = parser.parse_args()

    # Check environment unless skipped
    if not args.skip_env_check:
        if not check_environment() and not args.force:
            logging.error("Environment check failed. Use --force to run tests anyway.")
            return 1

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
