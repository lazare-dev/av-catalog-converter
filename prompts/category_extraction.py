# prompts/category_extraction.py
"""
Category extraction prompt builder
"""
import json
from prompts.templates.category_template import CATEGORY_EXTRACTION_TEMPLATE

def get_category_extraction_prompt(products: list) -> str:
    """
    Generate category extraction prompt
    
    Args:
        products (list): List of products to categorize
        
    Returns:
        str: Formatted prompt
    """
    # Format products information as JSON
    products_json = json.dumps(products, indent=2)
    
    # Prepare prompt
    prompt = CATEGORY_EXTRACTION_TEMPLATE.format(
        products_json=products_json
    )
    
    return prompt