# AV Catalog Converter Tests

This directory contains tests for the AV Catalog Converter application.

## Test Structure

- `unit/`: Unit tests for individual components
  - `core/`: Tests for core components
  - `services/`: Tests for services
  - `utils/`: Tests for utilities
- `integration/`: Integration tests for multiple components
- `fixtures/`: Test fixtures (sample files, etc.)

## New Optimized Components Tests

The following tests have been added for the optimized components:

### Parallel Processing

- `tests/unit/utils/parallel/test_parallel_processor.py`: Tests for the ParallelProcessor class
- `tests/unit/core/file_parser/test_csv_parser_parallel.py`: Tests for the CSV parser with parallel processing
- `tests/unit/core/file_parser/test_excel_parser_parallel.py`: Tests for the Excel parser with parallel processing

### Rate Limiting

- `tests/unit/utils/rate_limiting/test_rate_limiter.py`: Tests for the RateLimiter class
- `tests/unit/core/llm/test_phi_client.py`: Tests for the PhiClient with rate limiting

### Adaptive Caching

- `tests/unit/utils/caching/test_adaptive_cache.py`: Tests for the AdaptiveCache class

### Integration Tests

- `tests/integration/test_optimized_components.py`: Integration tests for all optimized components
- `tests/integration/test_api_with_optimizations.py`: Integration tests for the API with optimized components

## Running Tests

### Running All Tests

```bash
python -m pytest tests/
```

### Running Specific Test Categories

```bash
# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/

# Run tests for parallel processing
python -m pytest -m parallel

# Run tests for rate limiting
python -m pytest -m rate_limiting

# Run tests for caching
python -m pytest -m caching
```

### Running Tests with Coverage

```bash
python -m pytest --cov=. --cov-report=term --cov-report=html
```

### Running Tests in Parallel

```bash
python -m pytest -n auto
```

## Test Fixtures

Test fixtures are located in the `fixtures/` directory. These include:

- `small_catalog.csv`: A small CSV catalog (10 rows)
- `medium_catalog.csv`: A medium-sized CSV catalog (1,000 rows)
- `small_catalog.xlsx`: A small Excel catalog (10 rows)
- `medium_catalog.xlsx`: A medium-sized Excel catalog (1,000 rows)

## Docker Integration

The tests are automatically run when the Docker container is started. This is configured in the `docker/entrypoint.sh` script.

To run the tests manually in the Docker container:

```bash
docker exec -it av-catalog-converter python -m pytest
```
