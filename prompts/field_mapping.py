# prompts/field_mapping.py
"""
Field mapping prompt builder for AV Catalog Converter
Generates prompts for mapping input columns to standardized schema fields
"""
from prompts.templates.mapping_template import FIELD_MAPPING_TEMPLATE
from config.schema import REQUIRED_FIELDS

def get_field_mapping_prompt(standard_fields, input_columns: list,
                           column_samples: dict, structure_info: dict) -> str:
    """
    Generate field mapping prompt optimized for Phi-2 model

    Args:
        standard_fields: Standard field definitions to map to (list or dict)
        input_columns (list): Input column names to map from
        column_samples (dict): Sample values for each input column
        structure_info (dict): Structure analysis information

    Returns:
        str: Formatted prompt
    """
    # Format standard fields information with required fields highlighted
    standard_fields_str = "Standard fields to map (in output order):\n"

    # Handle different types of standard_fields input
    if isinstance(standard_fields, dict):
        # Dictionary format
        for field_name, field_info in standard_fields.items():
            required_marker = " (REQUIRED)" if field_name in REQUIRED_FIELDS else ""
            description = field_info.get('description', '') if isinstance(field_info, dict) else ''
            standard_fields_str += f"- {field_name}{required_marker}: {description}\n"

            # Add examples if available
            if isinstance(field_info, dict):
                examples = field_info.get('examples', [])
                if examples and len(examples) > 0:
                    examples_str = ', '.join([str(ex) for ex in examples[:3] if ex])
                    if examples_str:
                        standard_fields_str += f"  Examples: {examples_str}\n"

                # Add mapping hints if available
                if field_info.get('mapping_hints'):
                    hints = field_info.get('mapping_hints', [])
                    if len(hints) > 3:
                        hints = hints[:3]
                    hints_str = ', '.join([str(h) for h in hints if h])
                    if hints_str:
                        standard_fields_str += f"  Common column names: {hints_str}\n"
    elif isinstance(standard_fields, list):
        # List format
        for field_name in standard_fields:
            required_marker = " (REQUIRED)" if field_name in REQUIRED_FIELDS else ""
            standard_fields_str += f"- {field_name}{required_marker}\n"
    else:
        # Fallback for other formats
        standard_fields_str += "- Unable to format standard fields\n"

    # Format input columns with more context
    input_columns_str = f"Input columns ({len(input_columns)}):\n"
    for col in input_columns:
        input_columns_str += f"- {col}\n"

    # Format column samples with more examples
    column_samples_str = "Column samples (first few values):\n"
    for col, samples in column_samples.items():
        # Filter out None values and empty strings
        valid_samples = [str(s) for s in samples[:5] if s is not None and str(s).strip()]
        if valid_samples:
            samples_str = ', '.join(valid_samples)
            column_samples_str += f"- {col}: {samples_str}\n"
        else:
            column_samples_str += f"- {col}: [empty]\n"

    # Format structure info with more details
    structure_info_str = "Structure analysis:\n"

    # Handle different structure info formats
    if structure_info:
        if 'column_purpose' in structure_info:
            structure_info_str += "Column purposes:\n"
            for col, purpose in structure_info.get('column_purpose', {}).items():
                if col in input_columns:
                    structure_info_str += f"- {col}: {purpose}\n"

        if 'possible_field_mappings' in structure_info:
            structure_info_str += "\nSuggested mappings from structure analysis:\n"
            for field, mapping in structure_info.get('possible_field_mappings', {}).items():
                if isinstance(mapping, dict) and 'column' in mapping:
                    col = mapping.get('column')
                    confidence = mapping.get('confidence', 0)
                    structure_info_str += f"- {field} → {col} (Confidence: {confidence:.2f})\n"

        if 'primary_keys' in structure_info:
            primary_keys = structure_info.get('primary_keys', [])
            if primary_keys:
                structure_info_str += f"\nPotential unique identifiers: {', '.join(primary_keys)}\n"

        if 'data_quality_issues' in structure_info:
            issues = structure_info.get('data_quality_issues', [])
            if issues:
                structure_info_str += "\nData quality issues:\n"
                for issue in issues[:3]:
                    structure_info_str += f"- {issue}\n"
    else:
        structure_info_str += "No detailed structure analysis available.\n"

    # Prepare prompt with all information
    # Get the template
    template = FIELD_MAPPING_TEMPLATE

    # Create a copy of the template to avoid modifying the original
    prompt_template = template

    # Replace the placeholders with our data
    prompt = prompt_template.replace('{standard_fields}', standard_fields_str)
    prompt = prompt.replace('{input_columns}', input_columns_str)
    prompt = prompt.replace('{column_samples}', column_samples_str)
    prompt = prompt.replace('{structure_info}', structure_info_str)

    return prompt