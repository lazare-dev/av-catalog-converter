# prompts/examples/structure_examples.py
"""
Structure analysis examples
"""

STRUCTURE_ANALYSIS_EXAMPLE_INPUT = """
Model,Description,Manufacturer,Price,Category
XG500,4K Laser Projector,Sony,3499.99,Projectors
LS800,Wireless Speaker,Bose,299.95,Audio
MX2000,Video Conferencing System,Polycom,4599.00,Conferencing
"""

STRUCTURE_ANALYSIS_EXAMPLE_OUTPUT = """
{
  "column_analysis": {
    "Model": {
      "purpose": "Manufacturer's model number",
      "data_characteristics": "Short alphanumeric codes",
      "potential_mapping": "Model"
    },
    "Description": {
      "purpose": "Brief product description",
      "data_characteristics": "Short text describing key product features",
      "potential_mapping": "Short Description"
    },
    "Manufacturer": {
      "purpose": "Product manufacturer name",
      "data_characteristics": "Company names, low cardinality",
      "potential_mapping": "Manufacturer"
    },
    "Price": {
      "purpose": "Product price",
      "data_characteristics": "Numeric values with decimal points",
      "potential_mapping": "Trade Price"
    },
    "Category": {
      "purpose": "Product category",
      "data_characteristics": "General product type, low cardinality",
      "potential_mapping": "Category"
    }
  },
  "structure_notes": "This is a simple product catalog with basic information. It has a clear structure with one row per product and consistent data formats. The catalog lacks some fields from the standard schema like SKU, Long Description, and Category Group.",
  "possible_field_mappings": {
    "Model": {"column": "Model", "confidence": 0.95},
    "Short Description": {"column": "Description", "confidence": 0.9},
    "Manufacturer": {"column": "Manufacturer", "confidence": 0.95},
    "Trade Price": {"column": "Price", "confidence": 0.8},
    "Category": {"column": "Category", "confidence": 0.9}
  },
  "data_quality_issues": [
    "No unique product identifier (SKU) is present",
    "No category hierarchy (missing Category Group)",
    "Limited product information (missing detailed specifications)"
  ]
}
"""