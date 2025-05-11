# AV Catalog Converter Architecture Guide

This document provides a detailed overview of the AV Catalog Converter architecture, explaining the design decisions, component interactions, and data flow.

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Component Interactions](#component-interactions)
- [Extension Points](#extension-points)
- [Performance Considerations](#performance-considerations)

## System Overview

The AV Catalog Converter is designed with a modular, extensible architecture that separates concerns and allows for easy maintenance and enhancement. The system follows a pipeline architecture where data flows through various processing stages:

1. **File Parsing**: Convert input files from various formats into a standardized internal representation
2. **Structure Analysis**: Analyze the structure of the data to identify columns and data types
3. **Field Mapping**: Map input fields to standardized output fields
4. **Value Normalization**: Normalize values to ensure consistency
5. **Output Generation**: Generate the standardized output in the requested format

![Architecture Overview](../images/architecture.png)

## Core Components

### File Parsers

The file parsing subsystem is responsible for reading input files in various formats and converting them into a standardized internal representation (pandas DataFrame).

**Key Classes:**
- `BaseParser`: Abstract base class for all parsers
- `CSVParser`: Parser for CSV/TSV files
- `ExcelParser`: Parser for Excel files
- `PDFParser`: Parser for PDF files
- `JSONParser`: Parser for JSON files
- `XMLParser`: Parser for XML files
- `ParserFactory`: Factory for creating parser instances based on file type

### LLM Integration

The LLM (Language Learning Model) integration provides intelligent field mapping and text analysis capabilities.

**Key Classes:**
- `BaseLLMClient`: Abstract base class for LLM clients
- `PhiClient`: Client for Microsoft's Phi-2 model
- `LLMFactory`: Factory for creating LLM client instances
- `PromptManager`: Manages prompts for different LLM tasks

### Services

The services layer contains the business logic for analyzing, mapping, and normalizing catalog data.

**Key Classes:**
- `StructureAnalyzer`: Analyzes the structure of input data
- `FieldMapper`: Maps input fields to standardized output fields
- `CategoryExtractor`: Extracts and normalizes product categories
- `ValueNormalizer`: Normalizes values for consistency

### Utilities

The utilities layer provides common functionality used across the application.

**Key Classes:**
- `ParallelProcessor`: Handles parallel processing of data
- `CacheManager`: Manages caching of results
- `RateLimiter`: Implements rate limiting for API and LLM calls
- `Logger`: Centralized logging functionality

### Web Components

The web components provide a RESTful API and a React-based UI for interacting with the application.

**Key Classes:**
- `APIController`: Handles API requests
- `JobManager`: Manages processing jobs
- `ReactApp`: Frontend React application

## Data Flow

The data flows through the system as follows:

1. **Input**: User provides a catalog file through the UI, API, or command line
2. **Parsing**: The appropriate parser converts the file into a DataFrame
3. **Analysis**: The structure analyzer identifies columns and data types
4. **Mapping**: The field mapper maps input fields to standardized output fields
5. **Normalization**: Values are normalized for consistency
6. **Output**: The standardized data is exported in the requested format

![Data Flow Diagram](../images/data_flow.png)

## Component Interactions

### Parser and Structure Analyzer

The parser converts the input file into a DataFrame, which is then passed to the structure analyzer. The structure analyzer examines the data to identify columns, data types, and sample values.

```python
# Example interaction
parser = ParserFactory.create_parser(file_path)
data_frame = parser.parse()
analyzer = StructureAnalyzer()
structure_info = analyzer.analyze(data_frame)
```

### Structure Analyzer and Field Mapper

The structure analyzer provides information about the input data structure to the field mapper. The field mapper uses this information, along with the LLM client, to map input fields to standardized output fields.

```python
# Example interaction
field_mapper = FieldMapper(llm_client)
mappings = field_mapper.map_fields(structure_info)
```

### Field Mapper and Value Normalizer

The field mapper provides the mappings to the value normalizer, which uses them to normalize the values in the DataFrame.

```python
# Example interaction
normalizer = ValueNormalizer()
normalized_data = normalizer.normalize(data_frame, mappings)
```

## Extension Points

The AV Catalog Converter is designed to be easily extensible. Here are the main extension points:

### Adding a New Parser

To add support for a new file format, create a new parser class that inherits from `BaseParser` and implement the required methods.

```python
class NewFormatParser(BaseParser):
    def parse(self, file_path):
        # Implementation for parsing the new format
        return data_frame
    
    def get_headers(self, file_path):
        # Implementation for getting headers from the new format
        return headers
```

Then register the new parser in the `ParserFactory`.

### Adding a New Normalizer

To add a new normalization strategy, create a new normalizer class that inherits from `BaseNormalizer` and implement the required methods.

```python
class NewNormalizer(BaseNormalizer):
    def normalize(self, data, field_mappings):
        # Implementation for the new normalization strategy
        return normalized_data
```

Then register the new normalizer in the `NormalizerFactory`.

### Adding a New LLM Client

To integrate with a different LLM, create a new LLM client class that inherits from `BaseLLMClient` and implement the required methods.

```python
class NewLLMClient(BaseLLMClient):
    def generate(self, prompt, **kwargs):
        # Implementation for generating text with the new LLM
        return generated_text
    
    def get_embeddings(self, text, **kwargs):
        # Implementation for getting embeddings from the new LLM
        return embeddings
```

Then register the new LLM client in the `LLMFactory`.

## Performance Considerations

The AV Catalog Converter includes several features to optimize performance:

### Parallel Processing

The `ParallelProcessor` utility enables parallel processing of data, which can significantly improve performance for large files.

```python
from utils.parallel import ParallelProcessor

processor = ParallelProcessor()
results = processor.process_dataframe(data_frame, process_function)
```

### Caching

The `CacheManager` utility provides caching capabilities to avoid redundant computations.

```python
from utils.caching import CacheManager

cache = CacheManager()
result = cache.get_or_compute(key, compute_function)
```

### Rate Limiting

The `RateLimiter` utility implements rate limiting for API and LLM calls to prevent overloading external services.

```python
from utils.rate_limiting import RateLimiter

limiter = RateLimiter(max_calls=10, time_period=60)
with limiter:
    result = external_api_call()
```

For more detailed information on performance optimization, see the [Performance Optimization Guide](../performance_optimization.md).
