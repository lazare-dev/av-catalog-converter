# prompts/structure_analysis.py
"""
Structure analysis prompt builder for AV Catalog Converter
Generates prompts for analyzing input catalog structure
"""
import json
from prompts.templates.structure_template import STRUCTURE_ANALYSIS_TEMPLATE
from config.schema import REQUIRED_FIELDS

def get_structure_analysis_prompt(data_sample: str, column_info: dict,
                                header_info: dict, data_quality: dict) -> str:
    """
    Generate structure analysis prompt optimized for Phi-2 model

    Args:
        data_sample (str): Sample data string
        column_info (dict): Column types and characteristics
        header_info (dict): Header detection information
        data_quality (dict): Data quality information

    Returns:
        str: Formatted prompt
    """
    # Format column information with more details
    column_info_str = "Column information:\n"
    for col, info in column_info.items():
        # Get sample values, handling potential None values
        samples = info.get('samples', [])
        valid_samples = [str(s) for s in samples[:3] if s is not None and str(s).strip()]
        sample_values = ', '.join(valid_samples) if valid_samples else '[empty]'

        # Include more detailed information about each column
        column_info_str += f"- {col} (type: {info.get('type', 'unknown')})\n"
        column_info_str += f"  Uniqueness: {info.get('unique_ratio', 0):.2f}, "
        column_info_str += f"Empty ratio: {info.get('empty_ratio', 0):.2f}\n"
        column_info_str += f"  Sample values: {sample_values}\n"

        # Include pattern information if available
        if 'pattern' in info and info['pattern']:
            column_info_str += f"  Pattern: {info['pattern']}\n"

    # Format header information
    header_info_str = "Header information:\n"
    if isinstance(header_info, dict):
        if 'has_header' in header_info:
            header_info_str += f"- Has header: {header_info.get('has_header', False)}\n"
        if 'header_row' in header_info:
            header_info_str += f"- Header row: {header_info.get('header_row', 0)}\n"
        if 'confidence' in header_info:
            header_info_str += f"- Confidence: {header_info.get('confidence', 0):.2f}\n"
    else:
        header_info_str += json.dumps(header_info, indent=2)

    # Format data quality information
    data_quality_str = "Data quality information:\n"
    if isinstance(data_quality, dict):
        # Handle missing values
        if 'missing_values' in data_quality:
            data_quality_str += "- Missing values by column:\n"
            for col, count in data_quality.get('missing_values', {}).items():
                data_quality_str += f"  {col}: {count} missing values\n"

        # Handle other quality metrics
        if 'duplicate_rows' in data_quality:
            data_quality_str += f"- Duplicate rows: {data_quality.get('duplicate_rows', 0)}\n"

        if 'inconsistent_formats' in data_quality:
            data_quality_str += "- Inconsistent formats:\n"
            for col, details in data_quality.get('inconsistent_formats', {}).items():
                data_quality_str += f"  {col}: {details}\n"
    else:
        data_quality_str += json.dumps(data_quality, indent=2)

    # Add information about required fields
    data_quality_str += "\nRequired fields in output:\n"
    for field in REQUIRED_FIELDS:
        data_quality_str += f"- {field}\n"

    # Prepare prompt with all information
    prompt = STRUCTURE_ANALYSIS_TEMPLATE.format(
        data_sample=data_sample,
        column_info=column_info_str,
        header_info=header_info_str,
        data_quality=data_quality_str
    )

    return prompt
