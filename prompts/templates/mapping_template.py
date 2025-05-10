# prompts/templates/mapping_template.py
"""
Field mapping prompt templates
"""

FIELD_MAPPING_TEMPLATE = """
# Field Mapping Task

You are a field mapping expert for an AV (Audio-Visual) catalog standardization system. Your task is to map columns from an input catalog to our standard schema fields.

## Standard Schema Fields To Map
{standard_fields}

## Input Catalog Columns
{input_columns}

## Column Samples
{column_samples}

## Structure Information
{structure_info}

## Your Task
Analyze the input columns and their content samples, then determine the best mapping to our standard schema fields.

For each standard field, identify the input column that best matches it based on:
1. Name similarity
2. Content patterns
3. Data characteristics
4. Context within the catalog structure

## Return Format
Provide your mapping recommendations in JSON format with the following structure:
```json
{
  "mappings": {
    "StandardField1": {
      "column": "best_matching_input_column",
      "confidence": 0.9,
      "reasoning": "Brief explanation of why this mapping was chosen"
    },
    "StandardField2": {
      "column": "best_matching_input_column",
      "confidence": 0.7,
      "reasoning": "Brief explanation of why this mapping was chosen"
    },
    ...
  },
  "unmapped_standard_fields": [
    "StandardFieldX",
    "StandardFieldY"
  ],
  "unmapped_input_columns": [
    "InputColumnA",
    "InputColumnB"
  ],
  "notes": "Additional observations or suggestions"
}
```

Confidence scores should range from 0.0 to 1.0, where:
- 0.9-1.0: Very high confidence (exact match)
- 0.7-0.9: High confidence (strong evidence)
- 0.5-0.7: Medium confidence (reasonable evidence)
- 0.3-0.5: Low confidence (weak evidence)
- 0.0-0.3: Very low confidence (guess based on limited evidence)

Be thorough in your analysis and provide clear reasoning for each mapping decision.
"""