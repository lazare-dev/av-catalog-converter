"""
Semantic field mapping using AI assistance for AV Catalog Converter
Maps input columns to standardized schema fields using LLM
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
import re
import traceback

from config.schema import OUTPUT_SCHEMA, FIELD_ORDER, REQUIRED_FIELDS
from core.llm.llm_factory import LLMFactory
from utils.caching.disk_cache import DiskCache
from config.settings import CACHE_DIR, MAPPING_THRESHOLDS

class SemanticMapper:
    """Maps fields using semantic understanding and AI assistance"""

    def __init__(self):
        """Initialize the semantic mapper"""
        self.logger = logging.getLogger(__name__)
        self.llm_client = None  # Lazy initialization
        self.cache = DiskCache(f"{CACHE_DIR}/semantic_mapping", ttl=604800)  # 7 days

        # Track mapping statistics
        self.total_mapping_requests = 0
        self.cache_hits = 0
        self.successful_mappings = 0
        self.failed_mappings = 0

    def map_fields(self, data: pd.DataFrame, structure_info: Dict[str, Any] = None,
                  already_mapped: List[str] = None) -> Dict[str, str]:
        """
        Map input columns to standard schema fields using semantic understanding

        Args:
            data (pd.DataFrame): Input data
            structure_info (Dict[str, Any], optional): Structure analysis results
            already_mapped (List[str], optional): Fields already mapped

        Returns:
            Dict[str, str]: Mapping from input columns to standard fields
        """
        self.total_mapping_requests += 1

        # Initialize parameters
        already_mapped = already_mapped or []
        structure_info = structure_info or {}
        columns = list(data.columns)

        self.logger.info(f"Performing semantic mapping for {len(columns)} columns")

        # Prepare sample data - use more rows for better context
        sample_rows = min(15, len(data))
        sample_data = data.head(sample_rows).fillna("").to_dict(orient='records')

        # Create cache key from input data
        cache_key = {
            "columns": sorted(columns),
            "already_mapped": sorted(already_mapped),
            "sample_hash": hash(str(sample_data[:3]) if sample_data else "")  # Use first 3 rows for hash
        }

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.info("Using cached semantic mapping")
            self.cache_hits += 1
            return cached_result

        # Initialize LLM client if needed
        if not self.llm_client:
            try:
                self.llm_client = LLMFactory.create_client()
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM client: {str(e)}")
                self.failed_mappings += 1
                return {}

        # Prepare unmapped fields to focus on
        unmapped_fields = [f for f in FIELD_ORDER if f not in already_mapped]
        required_unmapped = [f for f in REQUIRED_FIELDS if f in unmapped_fields]

        # Prepare prompt with schema information and sample data
        prompt = self._create_mapping_prompt(columns, sample_data, unmapped_fields, required_unmapped, structure_info)

        # Get mapping from LLM
        try:
            # Set a longer timeout for mapping which can be complex
            response = self.llm_client.generate_response(prompt)
            mapping = self._parse_mapping_response(response, columns)

            # Validate the mapping
            mapping = self._validate_mapping(mapping, columns)

            # Cache the result
            self.cache.set(cache_key, mapping)

            self.successful_mappings += 1
            return mapping

        except Exception as e:
            self.logger.error(f"Error in semantic mapping: {str(e)}")
            self.logger.debug(traceback.format_exc())
            self.failed_mappings += 1
            return {}

    def _create_mapping_prompt(self, columns: List[str], sample_data: List[Dict[str, Any]],
                              unmapped_fields: List[str], required_unmapped: List[str],
                              structure_info: Dict[str, Any] = None) -> str:
        """
        Create prompt for the LLM to map fields

        Args:
            columns (List[str]): Input column names
            sample_data (List[Dict[str, Any]]): Sample data rows
            unmapped_fields (List[str]): Fields that need mapping
            required_unmapped (List[str]): Required fields that need mapping
            structure_info (Dict[str, Any], optional): Structure analysis information

        Returns:
            str: Formatted prompt
        """
        # Format sample data for display - include more rows for better context
        sample_str = json.dumps(sample_data[:5], indent=2)

        # Format schema information with more details
        schema_info = []
        for field in unmapped_fields:
            field_def = next((f for f in OUTPUT_SCHEMA if f.name == field), None)
            if field_def:
                required = "REQUIRED" if field in required_unmapped else "Optional"
                schema_info.append(f"- {field_def.name} ({required}): {field_def.description}")
                if hasattr(field_def, 'examples') and field_def.examples:
                    schema_info.append(f"  Examples: {', '.join(str(ex) for ex in field_def.examples[:3])}")
                if hasattr(field_def, 'mapping_hints') and field_def.mapping_hints:
                    schema_info.append(f"  Common column names: {', '.join(field_def.mapping_hints[:5])}")
                if hasattr(field_def, 'patterns') and field_def.patterns:
                    # Convert regex patterns to strings if needed
                    pattern_strs = [p.pattern if hasattr(p, 'pattern') else str(p) for p in field_def.patterns[:2]]
                    if pattern_strs:
                        schema_info.append(f"  Patterns: {', '.join(pattern_strs)}")

        schema_str = "\n".join(schema_info)

        # Format structure information if available
        structure_info_str = ""
        if structure_info:
            structure_info_str = "STRUCTURE ANALYSIS:\n"

            # Handle different structure info formats
            if 'column_analysis' in structure_info:
                for col, analysis in structure_info.get('column_analysis', {}).items():
                    if col in columns:
                        purpose = analysis.get('purpose', 'Unknown purpose')
                        potential_mapping = analysis.get('potential_mapping', 'No suggestion')
                        structure_info_str += f"- {col}: {purpose} (Suggested mapping: {potential_mapping})\n"

            elif 'column_types' in structure_info:
                for col, info in structure_info.get('column_types', {}).items():
                    if col in columns:
                        col_type = info.get('type', 'unknown')
                        samples = info.get('sample_values', [])
                        sample_str = ', '.join([str(s) for s in samples[:3]]) if samples else 'No samples'
                        structure_info_str += f"- {col}: Type: {col_type}, Samples: {sample_str}\n"

            elif 'possible_field_mappings' in structure_info:
                for field, mapping in structure_info.get('possible_field_mappings', {}).items():
                    if isinstance(mapping, dict) and 'column' in mapping:
                        col = mapping.get('column')
                        confidence = mapping.get('confidence', 0)
                        structure_info_str += f"- {field} -> {col} (Confidence: {confidence:.2f})\n"

        # Create the prompt with clear instructions
        prompt = f"""
# AV Catalog Field Mapping Task

You are a data mapping expert for an Audio-Visual equipment catalog standardization system. Your task is to map columns from an input catalog to our standardized schema fields.

## Input Catalog Columns
{', '.join(columns)}

## Sample Data (first few rows)
{sample_str}

## Target Schema Fields
{schema_str}

## Additional Context
{structure_info_str}

## Instructions
1. Analyze the input columns and their content in the sample data
2. Map each input column to the most appropriate target schema field
3. Focus especially on the REQUIRED fields - these must be mapped
4. Only map columns you're confident about (confidence > 0.6)
5. If a column doesn't clearly match any target field, don't map it

## Output Format
Return your mapping as a JSON object with this format:
```json
{{
  "input_column_name": "Target_Field_Name",
  "another_column": "Another_Field"
}}
```

For example:
```json
{{
  "product_id": "SKU",
  "name": "Short Description",
  "brand": "Manufacturer"
}}
```

Only include the JSON mapping object in your response, no additional text or explanations.
"""
        return prompt

    def _parse_mapping_response(self, response: str, columns: List[str]) -> Dict[str, str]:
        """
        Parse the LLM response into a mapping dictionary

        Args:
            response (str): LLM response text
            columns (List[str]): Original input columns for validation

        Returns:
            Dict[str, str]: Mapping from input columns to standard fields
        """
        mapping = {}

        # Clean up the response
        response = response.strip()

        # Try multiple parsing strategies
        parsing_methods = [
            self._parse_json_with_code_block,
            self._parse_json_without_code_block,
            self._parse_with_regex,
            self._parse_line_by_line
        ]

        for parse_method in parsing_methods:
            try:
                result = parse_method(response)
                if result:
                    # Validate the mapping
                    for source, target in result.items():
                        if source in columns and target in FIELD_ORDER:
                            mapping[source] = target
                        else:
                            self.logger.warning(f"Invalid mapping: {source} -> {target}")

                    if mapping:
                        self.logger.info(f"Successfully parsed mapping using {parse_method.__name__}")
                        break
            except Exception as e:
                self.logger.debug(f"Parsing method {parse_method.__name__} failed: {str(e)}")
                continue

        self.logger.info(f"Extracted {len(mapping)} mappings from semantic analysis")
        return mapping

    def _parse_json_with_code_block(self, response: str) -> Dict[str, str]:
        """Parse JSON from response with code block markers"""
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        return {}

    def _parse_json_without_code_block(self, response: str) -> Dict[str, str]:
        """Parse JSON from response without code block markers"""
        # Look for JSON object pattern
        json_match = re.search(r'({[\s\S]*?})(?:\s*$|\n)', response)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        return {}

    def _parse_with_regex(self, response: str) -> Dict[str, str]:
        """Parse mappings using regex patterns"""
        mapping = {}
        # Look for "key": "value" patterns
        pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, response)

        for source, target in matches:
            mapping[source] = target

        return mapping

    def _parse_line_by_line(self, response: str) -> Dict[str, str]:
        """Parse mappings line by line for simple formats"""
        mapping = {}
        lines = response.split('\n')

        for line in lines:
            # Look for patterns like "source -> target" or "source: target"
            arrow_match = re.search(r'["\']?([^"\']+)["\']?\s*->\s*["\']?([^"\']+)["\']?', line)
            if arrow_match:
                source, target = arrow_match.groups()
                mapping[source.strip()] = target.strip()
                continue

            colon_match = re.search(r'["\']?([^"\']+)["\']?\s*:\s*["\']?([^"\']+)["\']?', line)
            if colon_match:
                source, target = colon_match.groups()
                mapping[source.strip()] = target.strip()

        return mapping

    def _validate_mapping(self, mapping: Dict[str, str], columns: List[str]) -> Dict[str, str]:
        """
        Validate and clean up the mapping

        Args:
            mapping (Dict[str, str]): Raw mapping from LLM
            columns (List[str]): Valid input columns

        Returns:
            Dict[str, str]: Validated mapping
        """
        validated = {}

        # Check for duplicate target fields
        target_counts = {}
        for source, target in mapping.items():
            if target in target_counts:
                target_counts[target].append(source)
            else:
                target_counts[target] = [source]

        # Resolve duplicates by keeping the best match
        for target, sources in target_counts.items():
            if len(sources) > 1:
                self.logger.warning(f"Multiple columns mapped to {target}: {sources}")
                # For now, just keep the first one
                best_source = sources[0]
                validated[best_source] = target
            else:
                validated[sources[0]] = target

        # Ensure all sources are valid columns
        final_mapping = {src: tgt for src, tgt in validated.items() if src in columns}

        return final_mapping
