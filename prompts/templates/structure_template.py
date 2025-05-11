# prompts/templates/structure_template.py
"""
Structure analysis prompt templates for AV Catalog Converter
Optimized for Phi-2 model
"""

STRUCTURE_ANALYSIS_TEMPLATE = """
# AV Catalog Structure Analysis Task

You are a data structure analyzer for an Audio-Visual equipment catalog standardization system. Your task is to analyze the structure of an input catalog and identify key information about its organization to help with field mapping.

## Input Catalog Sample
```
{data_sample}
```

## Column Information
{column_info}

## Header Information
{header_info}

## Data Quality Information
{data_quality}

## Your Task
Analyze the structure of this catalog and provide the following information in JSON format:

1. Identify the main purpose of each column
2. Detect any patterns in the data organization
3. Suggest potential field mappings to our standard schema
4. Note any data quality issues that might affect mapping

## Standard Schema Fields (Target Format)
These are the fields we need to map to in the exact order required for the output:
1. SKU: Unique product identifier (REQUIRED)
2. Short Description: Brief product description (REQUIRED)
3. Long Description: Detailed product description
4. Model: Manufacturer's model number
5. Category Group: Top-level category
6. Category: Specific product category
7. Manufacturer: Name of manufacturer (REQUIRED)
8. Manufacturer SKU: Original manufacturer part number
9. Image URL: URL to product image
10. Document Name: Name of related document
11. Document URL: URL to product documentation
12. Unit Of Measure: Unit by which product is sold
13. Buy Cost: Dealer cost price
14. Trade Price: Standard dealer/reseller price (REQUIRED)
15. MSRP GBP: Manufacturer's suggested retail price in GBP
16. MSRP USD: Manufacturer's suggested retail price in USD
17. MSRP EUR: Manufacturer's suggested retail price in EUR
18. Discontinued: Discontinuation status flag

## Analysis Guidelines
- Look for patterns in column names that suggest their purpose
- Examine data values to determine column types and purposes
- Identify unique identifiers that could serve as SKUs
- Look for price columns that might map to cost/price fields
- Identify text columns that might contain product descriptions
- Check for URL patterns that might be image or document links
- Note any data quality issues like missing values, inconsistent formats, etc.

## Return Format
Provide your analysis in JSON format with the following structure:
```json
{
  "column_purpose": {
    "column_name1": "Purpose of this column (e.g., product identifier, description, price, etc.)",
    "column_name2": "Purpose of this column"
  },
  "dataset_type": "The type of dataset (e.g., product catalog, inventory list, etc.)",
  "primary_keys": ["List of columns that could serve as unique identifiers"],
  "data_organization": "Description of how the data is organized",
  "possible_field_mappings": {
    "SKU": {"column": "best_match_column", "confidence": 0.9, "reasoning": "Brief explanation"},
    "Short Description": {"column": "best_match_column", "confidence": 0.8, "reasoning": "Brief explanation"}
  },
  "data_quality_issues": [
    "Issue 1: Description of the issue and affected columns",
    "Issue 2: Description of the issue and affected columns"
  ]
}
```

Focus on providing accurate field mappings with high confidence scores (0.7+) for the REQUIRED fields: SKU, Short Description, Manufacturer, and Trade Price. Only include mappings you're reasonably confident about.
"""