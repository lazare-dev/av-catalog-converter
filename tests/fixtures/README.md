# Test Fixtures

This directory contains test fixtures for the AV Catalog Converter tests.

## File Types

- `small_catalog.csv`: A small CSV catalog for basic tests
- `medium_catalog.csv`: A medium-sized CSV catalog for performance tests
- `large_catalog.csv`: A large CSV catalog for parallel processing tests
- `small_catalog.xlsx`: A small Excel catalog for basic tests
- `medium_catalog.xlsx`: A medium-sized Excel catalog for performance tests
- `large_catalog.xlsx`: A large Excel catalog for parallel processing tests

## Usage

These fixtures are automatically excluded from the `.gitignore` file to ensure they are available for testing.

To use a fixture in a test:

```python
import os
import pytest
from pathlib import Path

# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def test_with_fixture():
    # Use a fixture file
    fixture_path = FIXTURES_DIR / "small_catalog.csv"
    
    # Test with the fixture
    assert fixture_path.exists()
    # ... rest of test
```

## Adding New Fixtures

When adding new fixtures, make sure to:

1. Keep them small (< 1MB if possible)
2. Document them in this README
3. Add them to the `.gitignore` exceptions if needed
