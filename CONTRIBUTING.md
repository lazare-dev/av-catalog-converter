# Contributing to AV Catalog Converter

Thank you for your interest in contributing to the AV Catalog Converter project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Branching Strategy](#branching-strategy)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/av-catalog-converter.git
   cd av-catalog-converter
   ```
3. Add the original repository as a remote:
   ```bash
   git remote add upstream https://github.com/originalowner/av-catalog-converter.git
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

### Setting Up

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For frontend development, install Node.js and npm:
   ```bash
   cd web/frontend
   npm install
   ```

### Running the Application

```bash
# Run the backend API
python app.py --api --port 8080

# In another terminal, run the frontend
cd web/frontend
npm start
```

## Branching Strategy

- `main` - Main branch, contains stable code
- `develop` - Development branch, contains code for the next release
- `feature/feature-name` - Feature branches, for developing new features
- `bugfix/bug-name` - Bug fix branches, for fixing bugs
- `release/version` - Release branches, for preparing releases

## Making Changes

1. Make sure you're working on the latest code:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

## Testing

Before submitting a pull request, make sure all tests pass:

```bash
# Run all tests
python run_tests.py --all

# Run unit tests
python run_tests.py --unit

# Run integration tests
python run_tests.py --integration

# Generate coverage report
python run_tests.py --coverage
```

If you're adding new functionality, please add appropriate tests.

## Pull Requests

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Create a pull request from your branch to the `develop` branch of the original repository
3. Provide a clear description of the changes and reference any related issues
4. Wait for the code review and address any feedback

## Code Style

We follow PEP 8 for Python code. Please ensure your code adheres to these standards:

```bash
# Install flake8
pip install flake8

# Run flake8 to check your code
flake8 your_changed_files.py
```

For JavaScript/TypeScript code, we follow the ESLint configuration in the frontend directory.

## Documentation

If you're adding new features or changing existing functionality, please update the documentation accordingly:

1. Update the relevant Markdown files in the `docs/` directory
2. Update the HTML documentation if applicable
3. Add docstrings to your code
4. Update the README.md if necessary

## Issue Reporting

If you find a bug or have a feature request:

1. Check if the issue already exists in the GitHub issue tracker
2. If not, create a new issue with a clear description and steps to reproduce (for bugs)
3. Include relevant information such as:
   - Operating system
   - Python version
   - Dependencies versions
   - Error messages and stack traces
   - Expected vs. actual behavior

Thank you for contributing to the AV Catalog Converter project!
