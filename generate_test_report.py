#!/usr/bin/env python3
"""
Generate an HTML test report from JUnit XML test results
"""
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

def parse_junit_xml(file_path):
    """Parse a JUnit XML file and return test statistics"""
    if not os.path.exists(file_path):
        print(f"Warning: Test result file not found: {file_path}")
        # Return empty results instead of None to ensure report generation
        # Include all required fields with default values
        return {
            'tests': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'success_rate': 0.0,
            'test_cases': [],
            'status': 'missing'
        }

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Get test statistics
        tests = int(root.attrib.get('tests', 0))
        failures = int(root.attrib.get('failures', 0))
        errors = int(root.attrib.get('errors', 0))
        skipped = int(root.attrib.get('skipped', 0))

        # Get test cases
        test_cases = []
        for test_case in root.findall('.//testcase'):
            case = {
                'name': test_case.attrib.get('name', 'Unknown'),
                'classname': test_case.attrib.get('classname', 'Unknown'),
                'time': float(test_case.attrib.get('time', 0)),
                'status': 'passed'
            }

            # Check for failures
            failure = test_case.find('failure')
            if failure is not None:
                case['status'] = 'failed'
                case['message'] = failure.attrib.get('message', '')
                case['type'] = failure.attrib.get('type', '')
                case['text'] = failure.text

            # Check for errors
            error = test_case.find('error')
            if error is not None:
                case['status'] = 'error'
                case['message'] = error.attrib.get('message', '')
                case['type'] = error.attrib.get('type', '')
                case['text'] = error.text

            # Check for skipped
            skipped_tag = test_case.find('skipped')
            if skipped_tag is not None:
                case['status'] = 'skipped'
                case['message'] = skipped_tag.attrib.get('message', '')

            test_cases.append(case)

        # If tests count is 0 but we have test cases, count them
        if tests == 0 and test_cases:
            tests = len(test_cases)
            failures = sum(1 for case in test_cases if case['status'] == 'failed')
            errors = sum(1 for case in test_cases if case['status'] == 'error')
            skipped = sum(1 for case in test_cases if case['status'] == 'skipped')
            print(f"Updated test counts from test cases: {tests} tests, {failures} failures, {errors} errors, {skipped} skipped")

        # Calculate success rate
        success_rate = ((tests - failures - errors) / tests * 100) if tests > 0 else 0

        return {
            'tests': tests,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'success_rate': success_rate,
            'test_cases': test_cases
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def generate_html_report(backend_results, frontend_results, integration_results):
    """Generate an HTML report from test results"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure all results have the required fields with valid values
    for results in [backend_results, frontend_results, integration_results]:
        if results:
            # Ensure tests count is valid
            if 'tests' not in results or results['tests'] == 0:
                if 'test_cases' in results and results['test_cases']:
                    results['tests'] = len(results['test_cases'])

            # Ensure other counts are valid
            if 'failures' not in results:
                results['failures'] = 0
            if 'errors' not in results:
                results['errors'] = 0
            if 'skipped' not in results:
                results['skipped'] = 0

            # Recalculate success rate
            if 'success_rate' not in results or results['success_rate'] == 0:
                if results['tests'] > 0:
                    results['success_rate'] = ((results['tests'] - results['failures'] - results['errors']) / results['tests'] * 100)
                else:
                    results['success_rate'] = 0.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AV Catalog Converter Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .summary {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            width: 30%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .summary-box h3 {{
            margin-top: 0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .success-rate {{
            font-size: 24px;
            font-weight: bold;
        }}
        .high {{
            color: #27ae60;
        }}
        .medium {{
            color: #f39c12;
        }}
        .low {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .passed {{
            color: #27ae60;
        }}
        .failed {{
            color: #e74c3c;
        }}
        .error {{
            color: #c0392b;
        }}
        .skipped {{
            color: #7f8c8d;
        }}
        .details {{
            background-color: #f8f9fa;
            border-left: 4px solid #ddd;
            padding: 10px;
            margin-top: 5px;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .error-details {{
            border-left-color: #e74c3c;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 20px;
        }}
        .no-results {{
            color: #7f8c8d;
            font-style: italic;
            padding: 20px;
            text-align: center;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>AV Catalog Converter Test Report</h1>
    <p class="timestamp">Generated on {now}</p>

    <div class="summary">
"""

    # Backend summary
    html += """
        <div class="summary-box">
            <h3>Backend Tests</h3>
"""

    if backend_results:
        success_rate_class = 'high' if backend_results['success_rate'] >= 90 else ('medium' if backend_results['success_rate'] >= 70 else 'low')
        html += f"""
            <p>Total Tests: {backend_results['tests']}</p>
            <p>Passed: {backend_results['tests'] - backend_results['failures'] - backend_results['errors']}</p>
            <p>Failed: {backend_results['failures']}</p>
            <p>Errors: {backend_results['errors']}</p>
            <p>Skipped: {backend_results['skipped']}</p>
            <p>Success Rate: <span class="success-rate {success_rate_class}">{backend_results['success_rate']:.2f}%</span></p>
"""
    else:
        html += """
            <p class="no-results">No backend test results available</p>
"""

    html += """
        </div>
"""

    # Frontend summary
    html += """
        <div class="summary-box">
            <h3>Frontend Tests</h3>
"""

    if frontend_results:
        success_rate_class = 'high' if frontend_results['success_rate'] >= 90 else ('medium' if frontend_results['success_rate'] >= 70 else 'low')
        html += f"""
            <p>Total Tests: {frontend_results['tests']}</p>
            <p>Passed: {frontend_results['tests'] - frontend_results['failures'] - frontend_results['errors']}</p>
            <p>Failed: {frontend_results['failures']}</p>
            <p>Errors: {frontend_results['errors']}</p>
            <p>Skipped: {frontend_results['skipped']}</p>
            <p>Success Rate: <span class="success-rate {success_rate_class}">{frontend_results['success_rate']:.2f}%</span></p>
"""
    else:
        html += """
            <p class="no-results">No frontend test results available</p>
"""

    html += """
        </div>
"""

    # Integration summary
    html += """
        <div class="summary-box">
            <h3>Integration Tests</h3>
"""

    if integration_results:
        success_rate_class = 'high' if integration_results['success_rate'] >= 90 else ('medium' if integration_results['success_rate'] >= 70 else 'low')
        html += f"""
            <p>Total Tests: {integration_results['tests']}</p>
            <p>Passed: {integration_results['tests'] - integration_results['failures'] - integration_results['errors']}</p>
            <p>Failed: {integration_results['failures']}</p>
            <p>Errors: {integration_results['errors']}</p>
            <p>Skipped: {integration_results['skipped']}</p>
            <p>Success Rate: <span class="success-rate {success_rate_class}">{integration_results['success_rate']:.2f}%</span></p>
"""
    else:
        html += """
            <p class="no-results">No integration test results available</p>
"""

    html += """
        </div>
    </div>
"""

    # Backend test details
    html += """
    <h2>Backend Test Details</h2>
"""

    if backend_results and backend_results['test_cases']:
        html += """
    <table>
        <thead>
            <tr>
                <th>Test</th>
                <th>Class</th>
                <th>Status</th>
                <th>Time (s)</th>
            </tr>
        </thead>
        <tbody>
"""

        for test_case in backend_results['test_cases']:
            html += f"""
            <tr>
                <td>{test_case['name']}</td>
                <td>{test_case['classname']}</td>
                <td class="{test_case['status']}">{test_case['status'].upper()}</td>
                <td>{test_case['time']:.3f}</td>
            </tr>
"""

            if test_case['status'] in ['failed', 'error'] and 'message' in test_case:
                html += f"""
            <tr>
                <td colspan="4">
                    <div class="details error-details">
                        <strong>{test_case.get('type', 'Error')}:</strong> {test_case.get('message', '')}
                        {test_case.get('text', '')}
                    </div>
                </td>
            </tr>
"""

        html += """
        </tbody>
    </table>
"""
    else:
        html += """
    <p class="no-results">No backend test details available</p>
"""

    # Frontend test details
    html += """
    <h2>Frontend Test Details</h2>
"""

    if frontend_results and frontend_results['test_cases']:
        html += """
    <table>
        <thead>
            <tr>
                <th>Test</th>
                <th>Class</th>
                <th>Status</th>
                <th>Time (s)</th>
            </tr>
        </thead>
        <tbody>
"""

        for test_case in frontend_results['test_cases']:
            html += f"""
            <tr>
                <td>{test_case['name']}</td>
                <td>{test_case['classname']}</td>
                <td class="{test_case['status']}">{test_case['status'].upper()}</td>
                <td>{test_case['time']:.3f}</td>
            </tr>
"""

            if test_case['status'] in ['failed', 'error'] and 'message' in test_case:
                html += f"""
            <tr>
                <td colspan="4">
                    <div class="details error-details">
                        <strong>{test_case.get('type', 'Error')}:</strong> {test_case.get('message', '')}
                        {test_case.get('text', '')}
                    </div>
                </td>
            </tr>
"""

        html += """
        </tbody>
    </table>
"""
    else:
        html += """
    <p class="no-results">No frontend test details available</p>
"""

    # Integration test details
    html += """
    <h2>Integration Test Details</h2>
"""

    if integration_results and integration_results['test_cases']:
        html += """
    <table>
        <thead>
            <tr>
                <th>Test</th>
                <th>Class</th>
                <th>Status</th>
                <th>Time (s)</th>
            </tr>
        </thead>
        <tbody>
"""

        for test_case in integration_results['test_cases']:
            html += f"""
            <tr>
                <td>{test_case['name']}</td>
                <td>{test_case['classname']}</td>
                <td class="{test_case['status']}">{test_case['status'].upper()}</td>
                <td>{test_case['time']:.3f}</td>
            </tr>
"""

            if test_case['status'] in ['failed', 'error'] and 'message' in test_case:
                html += f"""
            <tr>
                <td colspan="4">
                    <div class="details error-details">
                        <strong>{test_case.get('type', 'Error')}:</strong> {test_case.get('message', '')}
                        {test_case.get('text', '')}
                    </div>
                </td>
            </tr>
"""

        html += """
        </tbody>
    </table>
"""
    else:
        html += """
    <p class="no-results">No integration test details available</p>
"""

    html += """
</body>
</html>
"""

    return html

def main():
    """Main function"""
    # Ensure test_results directory exists
    os.makedirs('test_results', exist_ok=True)

    # Parse test results
    backend_results = parse_junit_xml('test_results/backend_results.xml')
    frontend_results = parse_junit_xml('test_results/frontend_results.xml')
    integration_results = parse_junit_xml('test_results/integration_results.xml')

    # Ensure all results are valid dictionaries with required fields
    if not backend_results:
        backend_results = {
            'tests': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'success_rate': 0.0,
            'test_cases': []
        }

    if not frontend_results:
        frontend_results = {
            'tests': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'success_rate': 0.0,
            'test_cases': []
        }

    if not integration_results:
        integration_results = {
            'tests': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'success_rate': 0.0,
            'test_cases': []
        }

    # Always update test counts based on test_cases
    if backend_results and 'test_cases' in backend_results:
        backend_results['tests'] = len(backend_results['test_cases'])
        passed = sum(1 for case in backend_results['test_cases'] if case['status'] == 'passed')
        failed = sum(1 for case in backend_results['test_cases'] if case['status'] == 'failed')
        errors = sum(1 for case in backend_results['test_cases'] if case['status'] == 'error')
        skipped = sum(1 for case in backend_results['test_cases'] if case['status'] == 'skipped')

        backend_results['failures'] = failed
        backend_results['errors'] = errors
        backend_results['skipped'] = skipped
        backend_results['success_rate'] = (passed / backend_results['tests'] * 100) if backend_results['tests'] > 0 else 0.0

        print(f"Updated backend test counts: {backend_results['tests']} tests, {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")

    # Do the same for integration tests
    if integration_results and 'test_cases' in integration_results:
        integration_results['tests'] = len(integration_results['test_cases'])
        passed = sum(1 for case in integration_results['test_cases'] if case['status'] == 'passed')
        failed = sum(1 for case in integration_results['test_cases'] if case['status'] == 'failed')
        errors = sum(1 for case in integration_results['test_cases'] if case['status'] == 'error')
        skipped = sum(1 for case in integration_results['test_cases'] if case['status'] == 'skipped')

        integration_results['failures'] = failed
        integration_results['errors'] = errors
        integration_results['skipped'] = skipped
        integration_results['success_rate'] = (passed / integration_results['tests'] * 100) if integration_results['tests'] > 0 else 0.0

        print(f"Updated integration test counts: {integration_results['tests']} tests, {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")

    # Add timestamp to the report
    timestamp = datetime.now().isoformat()
    backend_results['timestamp'] = timestamp
    frontend_results['timestamp'] = timestamp
    integration_results['timestamp'] = timestamp

    # Generate HTML report
    html_report = generate_html_report(backend_results, frontend_results, integration_results)

    # Write report to file
    report_path = 'test_results/test_report.html'
    with open(report_path, 'w') as f:
        f.write(html_report)

    # Also save a timestamped copy for historical reference
    timestamped_path = f'test_results/test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    with open(timestamped_path, 'w') as f:
        f.write(html_report)

    print(f"Test report generated: {os.path.abspath(report_path)}")
    print(f"Timestamped copy saved: {os.path.abspath(timestamped_path)}")

    # Open the report in a browser if possible
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(report_path))
    except Exception as e:
        print(f"Could not open report in browser: {e}")

if __name__ == '__main__':
    main()
