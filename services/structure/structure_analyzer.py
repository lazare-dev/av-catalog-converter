# services/structure/structure_analyzer.py
"""
Structure analysis service for understanding input catalog format
"""
import logging
import pandas as pd
import json
import re
import traceback
from typing import Dict, Any, List

from services.structure.header_detector import HeaderDetector
from services.structure.data_boundary_detector import DataBoundaryDetector
from core.llm.llm_factory import LLMFactory
from prompts.structure_analysis import get_structure_analysis_prompt
from config.schema import FIELD_ORDER, REQUIRED_FIELDS

class StructureAnalyzer:
    """Service for analyzing the structure of input data"""

    def __init__(self):
        """Initialize the structure analyzer"""
        self.logger = logging.getLogger(__name__)
        self.header_detector = HeaderDetector()
        self.boundary_detector = DataBoundaryDetector()
        self.llm_client = LLMFactory.create_client()

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the structure of the input data

        Args:
            data (pd.DataFrame): Input data to analyze

        Returns:
            Dict[str, Any]: Analysis results containing structure information
        """
        self.logger.info("Starting structure analysis")

        # Initialize results
        analysis = {
            "original_columns": list(data.columns),
            "data_shape": data.shape,
            "column_types": {},
            "data_quality": {},
            "nested_structure": {},
            "header_info": {},
            "possible_field_mappings": {}
        }

        # Detect headers and clean columns
        header_info = self.header_detector.detect_headers(data)
        analysis["header_info"] = header_info

        # Detect data boundaries
        boundaries = self.boundary_detector.detect_boundaries(data)
        analysis["data_boundaries"] = boundaries

        # Analyze column types and data patterns
        analysis["column_types"] = self._analyze_column_types(data)

        # Analyze data quality
        analysis["data_quality"] = self._analyze_data_quality(data)

        # Detect nested structure (if any)
        analysis["nested_structure"] = self._detect_nested_structure(data)

        # Perform AI-powered structure analysis
        ai_analysis = self._perform_ai_analysis(data, analysis)
        analysis.update(ai_analysis)

        self.logger.info("Structure analysis completed")
        return analysis

    def _analyze_column_types(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze types and patterns in each column

        Args:
            data (pd.DataFrame): Input data

        Returns:
            Dict[str, Dict[str, Any]]: Column type information
        """
        column_types = {}

        for col in data.columns:
            col_data = data[col].dropna()

            if len(col_data) == 0:
                column_types[col] = {
                    "type": "unknown",
                    "empty": True
                }
                continue

            # Determine basic type
            inferred_type = "string"
            numeric_count = pd.to_numeric(col_data, errors='coerce').notna().sum()
            numeric_ratio = numeric_count / len(col_data)

            if numeric_ratio > 0.9:
                # Check if values look like prices
                price_pattern = col_data.astype(str).str.contains(r'[$£€]|price|cost', case=False)
                if price_pattern.sum() / len(col_data) > 0.3:
                    inferred_type = "price"
                else:
                    # Check for decimal points (prices often have them)
                    decimal_pattern = col_data.astype(str).str.contains(r'\.\d+')
                    if decimal_pattern.sum() / len(col_data) > 0.5:
                        inferred_type = "decimal"
                    else:
                        inferred_type = "integer"

            # Check for URL patterns
            url_pattern = col_data.astype(str).str.contains(r'https?://|www\.|\.(com|org|net)')
            if url_pattern.sum() / len(col_data) > 0.7:
                inferred_type = "url"

            # Look for ID/SKU patterns
            id_pattern = col_data.astype(str).str.match(r'^[A-Za-z0-9\-_]{3,20}$')
            if id_pattern.sum() / len(col_data) > 0.8:
                # Check for column name hints
                if any(hint in str(col).lower() for hint in ['sku', 'id', 'code', 'part', 'model']):
                    inferred_type = "id"

            # Count unique values ratio
            unique_ratio = col_data.nunique() / len(col_data)

            # Store the analysis
            column_types[col] = {
                "type": inferred_type,
                "unique_count": col_data.nunique(),
                "unique_ratio": unique_ratio,
                "sample_values": col_data.sample(min(5, len(col_data))).tolist(),
                "empty": False
            }

        return column_types

    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality issues

        Args:
            data (pd.DataFrame): Input data

        Returns:
            Dict[str, Any]: Data quality information
        """
        quality = {
            "missing_values": {},
            "duplicates": {},
            "inconsistencies": {}
        }

        # Check missing values by column
        missing = data.isnull().sum()
        for col in data.columns:
            missing_count = missing[col]
            if missing_count > 0:
                quality["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_count / len(data) * 100)
                }

        # Check for duplicate rows
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            quality["duplicates"]["rows"] = {
                "count": int(duplicate_rows),
                "percentage": float(duplicate_rows / len(data) * 100)
            }

        # Check for inconsistencies in column values
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            # Check for mixed data types
            numeric_count = pd.to_numeric(col_data, errors='coerce').notna().sum()
            numeric_ratio = numeric_count / len(col_data)

            if 0.1 < numeric_ratio < 0.9:
                quality["inconsistencies"][col] = {
                    "issue": "mixed_types",
                    "details": f"Column contains both numeric ({numeric_ratio:.1%}) and non-numeric values"
                }

            # For price columns, check currency consistency
            if any(hint in str(col).lower() for hint in ['price', 'cost', 'msrp']):
                currency_symbols = col_data.astype(str).str.extract(r'([$£€])')[0].dropna()
                if len(currency_symbols) > 0 and currency_symbols.nunique() > 1:
                    quality["inconsistencies"][col] = {
                        "issue": "mixed_currencies",
                        "details": f"Column contains mixed currency symbols: {', '.join(currency_symbols.unique())}"
                    }

        return quality

    def _detect_nested_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if the data has nested structure (e.g., categories with subcategories)

        Args:
            data (pd.DataFrame): Input data

        Returns:
            Dict[str, Any]: Nested structure information
        """
        nested_info = {
            "has_nested_structure": False,
            "hierarchy_columns": [],
            "level_counts": {}
        }

        # Look for potential hierarchy columns
        hierarchy_candidates = []

        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            # Category columns typically have:
            # 1. Relatively few unique values compared to row count
            # 2. String values
            # 3. Names often contain 'category', 'group', 'type', etc.

            unique_ratio = col_data.nunique() / len(col_data)
            is_string_type = col_data.dtype == object
            category_name_hint = any(hint in str(col).lower() for hint in
                                    ['category', 'group', 'type', 'class', 'department'])

            # Score this column as a category candidate
            category_score = 0
            if unique_ratio < 0.2:  # Few unique values
                category_score += 3
            if is_string_type:
                category_score += 1
            if category_name_hint:
                category_score += 2

            if category_score >= 3:
                hierarchy_candidates.append((col, category_score))

        # Sort candidates by score (descending)
        hierarchy_candidates.sort(key=lambda x: x[1], reverse=True)

        # Extract top candidates
        hierarchy_columns = [col for col, score in hierarchy_candidates[:3]]

        if len(hierarchy_columns) >= 2:
            # Check if there's a relationship between columns that suggests hierarchy
            # For each pair of potential hierarchy columns, see if values in one column
            # correspond consistently to values in another (parent-child relationship)

            for i in range(len(hierarchy_columns)):
                for j in range(i+1, len(hierarchy_columns)):
                    col1 = hierarchy_columns[i]
                    col2 = hierarchy_columns[j]

                    # Group by col1 and count unique values in col2
                    grouped = data.groupby(col1)[col2].nunique()

                    # If col1 values consistently map to few col2 values,
                    # there might be a hierarchy
                    if grouped.mean() < 5 and grouped.max() < 10:
                        nested_info["has_nested_structure"] = True
                        nested_info["hierarchy_columns"] = [col1, col2]

                        # Count items at each level
                        nested_info["level_counts"][col1] = data[col1].nunique()
                        nested_info["level_counts"][col2] = data[col2].nunique()
                        break

        return nested_info

    def _perform_ai_analysis(self, data, basic_analysis):
        """
        Perform AI-based structure analysis

        Args:
            data (pd.DataFrame): Input data
            basic_analysis (Dict): Basic analysis results

        Returns:
            Dict[str, Any]: AI-augmented analysis results
        """
        self.logger.info("Performing AI-based structure analysis")

        try:
            # Prepare a sample of the data for the model - include more rows for better context
            sample_rows = min(25, len(data))
            data_sample = data.head(sample_rows).fillna("").to_string(index=False)

            # Get columns with their types and examples
            column_info = {}
            for col, info in basic_analysis["column_types"].items():
                # Include more detailed information for each column
                column_info[col] = {
                    "type": info["type"],
                    "samples": info.get("sample_values", []),
                    "unique_ratio": info.get("unique_ratio", 0),
                    "empty_ratio": info.get("empty_ratio", 0) if "empty_ratio" in info else 0,
                    "pattern": info.get("pattern", "")
                }

            # Add more context about data quality
            data_quality = basic_analysis.get("data_quality", {})
            if "missing_values" in basic_analysis:
                data_quality["missing_values"] = basic_analysis["missing_values"]

            # Generate the prompt for structure analysis
            prompt = get_structure_analysis_prompt(
                data_sample=data_sample,
                column_info=column_info,
                header_info=basic_analysis["header_info"],
                data_quality=data_quality
            )

            # Get response from LLM with a longer timeout for complex analysis
            response = self.llm_client.generate_response(prompt)

            # Parse the response (expecting JSON format)
            ai_analysis = self._parse_ai_response(response)

            # Validate and enhance the analysis
            ai_analysis = self._validate_ai_analysis(ai_analysis, basic_analysis)

            self.logger.info("AI structure analysis completed successfully")
            return ai_analysis

        except Exception as e:
            self.logger.error(f"Error in AI structure analysis: {str(e)}")
            self.logger.debug(f"Raw response: {response if 'response' in locals() else 'No response generated'}")

            # Return a minimal valid structure to avoid downstream errors
            return {
                "ai_analysis_error": str(e),
                "column_purpose": {},
                "dataset_type": "unknown",
                "primary_keys": [],
                "data_organization": "unknown",
                "possible_field_mappings": {},
                "data_quality_issues": ["AI analysis failed: " + str(e)]
            }

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the AI response into structured data

        Args:
            response (str): AI model response

        Returns:
            Dict[str, Any]: Parsed analysis results

        Raises:
            ValueError: If response cannot be parsed
        """
        # Extract JSON part from the response (if not already JSON)
        try:
            # Try to see if the whole response is valid JSON
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # If not, try to extract JSON block from the response
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response)

            if matches:
                try:
                    result = json.loads(matches[0])
                    return result
                except json.JSONDecodeError:
                    # Try to clean up the JSON string
                    cleaned_json = re.sub(r'[\n\r\t]', ' ', matches[0])
                    try:
                        result = json.loads(cleaned_json)
                        return result
                    except json.JSONDecodeError:
                        pass

            # Try to find any JSON-like structure in the response
            json_pattern = r'{[\s\S]*?}'
            matches = re.findall(json_pattern, response)

            if matches:
                for match in matches:
                    try:
                        result = json.loads(match)
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except json.JSONDecodeError:
                        continue

            self.logger.error("Failed to parse AI response")
            self.logger.debug(f"AI response: {response}")
            raise ValueError("Could not parse structured data from AI response")

    def _validate_ai_analysis(self, ai_analysis: Dict[str, Any], basic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance the AI analysis results

        Args:
            ai_analysis (Dict[str, Any]): AI analysis results
            basic_analysis (Dict[str, Any]): Basic analysis results

        Returns:
            Dict[str, Any]: Validated and enhanced analysis
        """
        # Ensure all required keys are present
        required_keys = [
            "column_purpose",
            "dataset_type",
            "primary_keys",
            "data_organization",
            "possible_field_mappings",
            "data_quality_issues"
        ]

        for key in required_keys:
            if key not in ai_analysis:
                self.logger.warning(f"Missing key in AI analysis: {key}")
                if key in ["column_purpose", "possible_field_mappings", "data_quality_issues"]:
                    ai_analysis[key] = {}
                elif key in ["primary_keys"]:
                    ai_analysis[key] = []
                else:
                    ai_analysis[key] = "Unknown"

        # Validate field mappings
        if "possible_field_mappings" in ai_analysis:
            valid_mappings = {}
            for field, mapping in ai_analysis["possible_field_mappings"].items():
                # Check if field is in our schema
                if field in FIELD_ORDER:
                    # Ensure mapping has required structure
                    if isinstance(mapping, dict) and "column" in mapping:
                        # Add confidence if missing
                        if "confidence" not in mapping:
                            mapping["confidence"] = 0.5

                        # Add reasoning if missing
                        if "reasoning" not in mapping:
                            mapping["reasoning"] = "No reasoning provided"

                        valid_mappings[field] = mapping
                    elif isinstance(mapping, str):
                        # Convert simple string mappings to proper format
                        valid_mappings[field] = {
                            "column": mapping,
                            "confidence": 0.5,
                            "reasoning": "Direct mapping from AI analysis"
                        }

            ai_analysis["possible_field_mappings"] = valid_mappings

            # Prioritize required fields
            for field in REQUIRED_FIELDS:
                if field not in ai_analysis["possible_field_mappings"]:
                    # Try to find a mapping for this required field
                    self._suggest_mapping_for_required_field(field, ai_analysis, basic_analysis)

        return ai_analysis

    def _suggest_mapping_for_required_field(self, field: str, ai_analysis: Dict[str, Any], basic_analysis: Dict[str, Any]):
        """
        Suggest a mapping for a required field that wasn't mapped by the AI

        Args:
            field (str): Required field to map
            ai_analysis (Dict[str, Any]): AI analysis results
            basic_analysis (Dict[str, Any]): Basic analysis results
        """
        self.logger.info(f"Attempting to suggest mapping for required field: {field}")

        # Get column types from basic analysis
        column_types = basic_analysis.get("column_types", {})

        # Define field-specific heuristics
        if field == "SKU":
            # Look for ID-like columns
            for col, info in column_types.items():
                if info.get("type") in ["id", "string"]:
                    unique_ratio = info.get("unique_ratio", 0)
                    if unique_ratio > 0.9:  # Highly unique values
                        ai_analysis["possible_field_mappings"][field] = {
                            "column": col,
                            "confidence": 0.6,
                            "reasoning": "Column contains unique values that could serve as identifiers"
                        }
                        return

        elif field == "Short Description":
            # Look for name/title columns
            for col, info in column_types.items():
                if info.get("type") in ["string", "text"]:
                    if any(name in col.lower() for name in ["name", "title", "desc", "product"]):
                        ai_analysis["possible_field_mappings"][field] = {
                            "column": col,
                            "confidence": 0.6,
                            "reasoning": "Column name suggests it contains product names/descriptions"
                        }
                        return

        elif field == "Manufacturer":
            # Look for manufacturer/brand columns
            for col, info in column_types.items():
                if info.get("type") in ["string", "text"]:
                    if any(name in col.lower() for name in ["manufacturer", "brand", "vendor", "supplier", "make"]):
                        ai_analysis["possible_field_mappings"][field] = {
                            "column": col,
                            "confidence": 0.7,
                            "reasoning": "Column name suggests it contains manufacturer information"
                        }
                        return

        elif field == "Trade Price":
            # Look for price columns
            for col, info in column_types.items():
                if info.get("type") in ["price", "numeric", "decimal"]:
                    if any(name in col.lower() for name in ["price", "cost", "msrp", "rrp"]):
                        ai_analysis["possible_field_mappings"][field] = {
                            "column": col,
                            "confidence": 0.6,
                            "reasoning": "Column appears to contain price information"
                        }
                        return

        self.logger.warning(f"Could not suggest mapping for required field: {field}")
