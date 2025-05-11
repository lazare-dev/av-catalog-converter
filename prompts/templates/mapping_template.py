# prompts/templates/mapping_template.py
"""
Field mapping prompt templates for AV Catalog Converter
Optimized for Phi-2 model
"""

FIELD_MAPPING_TEMPLATE = """
# AV Catalog Field Mapping Task

You are a data mapping expert for an Audio-Visual equipment catalog standardization system. Your task is to map columns from an input catalog to our standardized schema fields.

## Standard Schema Fields
{standard_fields}

## Input Catalog Columns
{input_columns}

## Column Samples
{column_samples}

## Structure Information
{structure_info}

## Instructions
1. Analyze the input columns and their content in the sample data
2. Map each input column to the most appropriate target schema field
3. Focus especially on the REQUIRED fields - these must be mapped
4. Only map columns you're confident about (confidence > 0.6)
5. If a column doesn't clearly match any target field, don't map it

## Mapping Criteria
For each standard field, identify the input column that best matches it based on:
1. Name similarity (e.g., "product_id" → "SKU")
2. Content patterns (e.g., alphanumeric codes → "SKU")
3. Data characteristics (e.g., price values → price fields)
4. Context within the catalog structure

## Output Format
Return your mapping as a JSON object with this structure:
```json
{
  "field_mappings": {
    "SKU": {
      "column": "product_id",
      "confidence": 0.95,
      "reasoning": "Column contains unique alphanumeric product identifiers"
    },
    "Short Description": {
      "column": "product_name",
      "confidence": 0.9,
      "reasoning": "Contains brief product descriptions"
    }
  },
  "unmapped_standard_fields": [
    "Document URL",
    "MSRP EUR"
  ],
  "unmapped_input_columns": [
    "created_date",
    "internal_notes"
  ]
}
```

## Confidence Scoring
Confidence scores should range from 0.0 to 1.0, where:
- 0.9-1.0: Very high confidence (exact match or perfect pattern match)
- 0.7-0.9: High confidence (strong evidence from content and naming)
- 0.5-0.7: Medium confidence (reasonable evidence but some uncertainty)
- 0.3-0.5: Low confidence (weak evidence, significant uncertainty)
- 0.0-0.3: Very low confidence (guess based on limited evidence)

## Important Notes
- The output CSV must include these columns in this exact order: SKU, Short Description, Long Description, Model, Category Group, Category, Manufacturer, Manufacturer SKU, Image URL, Document Name, Document URL, Unit Of Measure, Buy Cost, Trade Price, MSRP GBP, MSRP USD, MSRP EUR, Discontinued
- Required fields (must be mapped): SKU, Short Description, Manufacturer, Trade Price
- Provide clear reasoning for each mapping decision
"""