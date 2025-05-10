# prompts/templates/structure_template.py
"""
Structure analysis prompt templates
"""

STRUCTURE_ANALYSIS_TEMPLATE = """
# Structure Analysis Task

You are a data structure analyzer for an AV (Audio-Visual) catalog standardization system. Your task is to analyze the structure of an input catalog and identify key information about its organization.

## Input Catalog Sample
```
{data_sample}
```

## Column Information
{column_info}

## Your Task
Analyze the structure of this catalog and provide the following information in JSON format:

1. Identify the main purpose of each column
2. Detect any patterns in the data organization
3. Suggest potential field mappings to our standard schema
4. Note any data quality issues that might affect mapping

## Standard Schema Fields (Target Format)
These are the fields we need to map to:
- SKU: Unique product identifier
- Short Description: Brief product description
- Long Description: Detailed product description
- Model: Manufacturer's model number
- Category Group: Top-level category
- Category: Specific product category
- Manufacturer: Name of manufacturer
- Manufacturer SKU: Original manufacturer part number
- Image URL: URL to product image
- Document Name: Name of related documen
- Document URL: URL to product documentation
- Unit Of Measure: Unit by which product is sold
- Buy Cost: Dealer cost price
- Trade Price: Standard dealer/reseller price
- MSRP GBP: Manufacturer's suggested retail price in GBP
- MSRP USD: Manufacturer's suggested retail price in USD
- MSRP EUR: Manufacturer's suggested retail price in EUR
- Discontinued: Discontinuation status flag

## Return Format
Provide your analysis in JSON format with the following structure:
```json
{
  "column_analysis": {
    "column_name1": {
      "purpose": "...",
      "data_characteristics": "...",
      "potential_mapping": "..."
    },
    ...
  },
  "structure_notes": "Overall observations about catalog structure",
  "possible_field_mappings": {
    "SKU": {"column": "best_match_column", "confidence": 0.9},
    "Short Description": {"column": "best_match_column", "confidence": 0.8},
    ...
  },
  "data_quality_issues": [
    "Issue 1...",
    "Issue 2..."
  ]
}
```

Be precise and thorough in your analysis. The goal is to understand how to map this catalog to our standard format with high accuracy.
"""