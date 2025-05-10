# prompts/templates/category_template.py
"""
Category extraction prompt templates
"""

CATEGORY_EXTRACTION_TEMPLATE = """
# Category Extraction Task

You are a category classification expert for an AV (Audio-Visual) catalog standardization system. Your task is to extract and standardize product categories from product information.

## Products to Categorize
{products_json}

## Standard AV Industry Categories
Top-level Category Groups:
- Display (projectors, screens, displays, monitors)
- Audio (speakers, amplifiers, microphones, audio processors)
- Video (cameras, video processors, media players)
- Control (control systems, touch panels, remote controls)
- Infrastructure (cables, connectors, racks, signal distribution)
- Conferencing (video conferencing, collaboration systems)
- Accessories (brackets, adapters, cases, tools)

## Your Task
For each product, determine:
1. The most appropriate Category Group (from the list above)
2. A specific Category within that group

Use the available product information (descriptions, model numbers, etc.) to make your determination.

## Return Format
Provide your category assignments in JSON format with the following structure:
```json
{
  "product_index1": {
    "category_group": "Top-level category",
    "category": "Specific category"
  },
  "product_index2": {
    "category_group": "Top-level category",
    "category": "Specific category"
  },
  ...
}
```

Be consistent in your categorization and use standard industry terminology.
"""