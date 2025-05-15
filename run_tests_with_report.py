#!/usr/bin/env python3
"""
Run tests and generate a report, regardless of test results.
This script ensures that test reports are always generated, even when tests fail.
"""
import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for test results"""
    os.makedirs('test_results', exist_ok=True)
    logger.info("Ensured test_results directory exists")

def run_tests(args):
    """Run pytest with the specified arguments"""
    # Build the pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test path if specified
    if args.test_path:
        cmd.append(args.test_path)
    else:
        cmd.append('tests/')
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add JUnit XML output
    cmd.append('--junitxml=test_results/backend_results.xml')
    
    # Add any additional arguments
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())
    
    # Run the command
    logger.info(f"Running tests with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Return the exit code
        return result.returncode
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1

def generate_report():
    """Generate the HTML test report"""
    logger.info("Generating test report...")
    try:
        # Run the report generator script
        result = subprocess.run(
            [sys.executable, 'generate_test_report.py'],
            capture_output=True,
            text=True
        )
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Check if report was generated
        if os.path.exists('test_results/test_report.html'):
            logger.info("Test report generated successfully: test_results/test_report.html")
            return True
        else:
            logger.error("Failed to generate test report")
            return False
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run tests and generate a report, regardless of test results.'
    )
    parser.add_argument(
        'test_path',
        nargs='?',
        help='Path to the test file or directory to run (default: tests/)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Run tests with increased verbosity'
    )
    parser.add_argument(
        '--pytest-args',
        help='Additional arguments to pass to pytest (e.g. "-k test_name -xvs")'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating the HTML report'
    )
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    setup_directories()
    
    # Run tests
    logger.info("Starting test execution")
    start_time = datetime.now()
    test_result = run_tests(args)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Log test result
    if test_result == 0:
        logger.info(f"Tests passed in {duration:.2f} seconds")
    else:
        logger.warning(f"Tests failed with exit code {test_result} in {duration:.2f} seconds")
    
    # Generate report unless disabled
    if not args.no_report:
        report_result = generate_report()
        if not report_result:
            logger.warning("Report generation failed")
    
    # Return the test result
    return test_result

if __name__ == '__main__':
    sys.exit(main())
