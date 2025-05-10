# prompts/examples/mapping_examples.py
"""
Field mapping examples
"""

FIELD_MAPPING_EXAMPLE_INPUT = """
Standard fields to map:
- SKU: Unique product identifier
  Examples: AV-PRJ-1001, SPK-WL-500
  Hints: sku, item_number, product_code
- Short Description: Brief product description
  Examples: 4K Laser Projector - 5000 lumens, Wireless Ceiling Speaker - White
  Hints: short_desc, product_name, title
- Manufacturer: Name of manufacturer
  Examples: Sony, Bose
  Hints: brand, vendor, supplier

Input columns:
- Item Number
- Product Name
- Brand
- List Price
- Category

Column samples:
- Item Number: PRJ-1001, SPK-500, CAB-HDMI-10
- Product Name: 4K Laser Projector, Wireless Speaker, HDMI Cable 10m
- Brand: Sony, Bose, Generic
- List Price: 3499.99, 299.95, 49.99
- Category: Projectors, Speakers, Cables
"""

FIELD_MAPPING_EXAMPLE_OUTPUT = """
{
  "mappings": {
    "SKU": {
      "column": "Item Number",
      "confidence": 0.9,
      "reasoning": "Column name is a common synonym for SKU and content matches expected pattern"
    },
    "Short Description": {
      "column": "Product Name",
      "confidence": 0.95,
      "reasoning": "Content matches expected short descriptions with product type and key features"
    },
    "Manufacturer": {
      "column": "Brand",
      "confidence": 0.95,
      "reasoning": "Column name is a synonym for Manufacturer and contains expected company names"
    }
  },
  "unmapped_standard_fields": [],
  "unmapped_input_columns": [
    "List Price",
    "Category"
  ],
  "notes": "The mapping for the required fields is very confident. The remaining input columns List Price and Category could potentially map to Trade Price and Category in the standard schema if needed."
}
"""