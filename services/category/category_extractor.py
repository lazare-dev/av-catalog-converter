# services/category/category_extractor.py
"""
Category extraction service for standardizing product categories
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional
import re

from services.category.hierarchy_analyzer import HierarchyAnalyzer
from services.category.taxonomy_mapper import TaxonomyMapper
from core.llm.llm_factory import LLMFactory
from prompts.category_extraction import get_category_extraction_prompt

class CategoryExtractor:
    """Service for extracting and standardizing product categories"""
    
    def __init__(self):
        """Initialize the category extractor"""
        self.logger = logging.getLogger(__name__)
        self.hierarchy_analyzer = HierarchyAnalyzer()
        self.taxonomy_mapper = TaxonomyMapper()
        self.llm_client = LLMFactory.create_client()
        
    def extract_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract, normalize, and assign standard categories to products
        
        Args:
            data (pd.DataFrame): Input data with mapped fields
            
        Returns:
            pd.DataFrame: Data with standardized categories
        """
        self.logger.info("Starting category extraction")
        
        # Make a copy to avoid modifying the original
        result_df = data.copy()
        
        # Check if we already have category fields
        has_category = "Category" in result_df.columns and result_df["Category"].notna().sum() > 0
        has_group = "Category Group" in result_df.columns and result_df["Category Group"].notna().sum() > 0
        
        if has_category and has_group:
            # We already have categories, just standardize them
            self.logger.info("Using existing category fields")
            return self._standardize_existing_categories(result_df)
            
        # If we have only one of the category fields, try to fill in the other
        if has_category and not has_group:
            self.logger.info("Deriving Category Group from existing Category")
            result_df = self._derive_category_group(result_df)
            return result_df
            
        if has_group and not has_category:
            self.logger.info("Deriving Category from existing Category Group")
            result_df = self._derive_category(result_df)
            return result_df
            
        # No category fields found, need to infer from product data
        self.logger.info("No category fields found, inferring from product data")
        return self._infer_categories(result_df)
    
    def _standardize_existing_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize existing category values
        
        Args:
            data (pd.DataFrame): Input data with category fields
            
        Returns:
            pd.DataFrame: Data with standardized categories
        """
        result_df = data.copy()
        
        # Extract current categories
        current_categories = result_df["Category"].dropna().unique()
        current_groups = result_df["Category Group"].dropna().unique()
        
        # Map to standard taxonomy
        category_mapping = self.taxonomy_mapper.map_to_standard_taxonomy(
            categories=current_categories,
            category_groups=current_groups
        )
        
        # Apply mappings
        for idx, row in result_df.iterrows():
            current_cat = row.get("Category")
            current_group = row.get("Category Group")
            
            if pd.notna(current_cat) and current_cat in category_mapping:
                result_df.at[idx, "Category"] = category_mapping[current_cat]["standard_category"]
                result_df.at[idx, "Category Group"] = category_mapping[current_cat]["standard_group"]
                
            elif pd.notna(current_group) and current_group in category_mapping:
                result_df.at[idx, "Category Group"] = category_mapping[current_group]["standard_group"]
                
                # Only set category if it's currently empty
                if pd.isna(current_cat) and "standard_category" in category_mapping[current_group]:
                    result_df.at[idx, "Category"] = category_mapping[current_group]["standard_category"]
        
        return result_df
    
    def _derive_category_group(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive Category Group from existing Category
        
        Args:
            data (pd.DataFrame): Input data with Category field
            
        Returns:
            pd.DataFrame: Data with both category fields
        """
        result_df = data.copy()
        
        # Get unique categories
        categories = result_df["Category"].dropna().unique()
        
        # Map to standard taxonomy to get groups
        category_mapping = self.taxonomy_mapper.map_to_standard_taxonomy(
            categories=categories,
            infer_groups=True
        )
        
        # Apply group mappings
        for idx, row in result_df.iterrows():
            current_cat = row.get("Category")
            
            if pd.notna(current_cat) and current_cat in category_mapping:
                result_df.at[idx, "Category"] = category_mapping[current_cat]["standard_category"]
                result_df.at[idx, "Category Group"] = category_mapping[current_cat]["standard_group"]
        
        return result_df
    
    def _derive_category(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive Category from existing Category Group
        
        Args:
            data (pd.DataFrame): Input data with Category Group field
            
        Returns:
            pd.DataFrame: Data with both category fields
        """
        result_df = data.copy()
        
        # Get unique category groups
        groups = result_df["Category Group"].dropna().unique()
        
        # Map to standard taxonomy
        group_mapping = self.taxonomy_mapper.map_to_standard_taxonomy(
            category_groups=groups,
            infer_categories=True
        )
        
        # Try to infer specific categories from product descriptions
        description_fields = ["Short Description", "Long Description", "Model"]
        present_fields = [f for f in description_fields if f in result_df.columns]
        
        if present_fields:
            # Use AI to extract categories from descriptions
            for idx, row in result_df.iterrows():
                group = row.get("Category Group")
                
                if pd.isna(group) or group not in group_mapping:
                    continue
                    
                std_group = group_mapping[group]["standard_group"]
                
                # Extract relevant text
                text = " ".join(str(row.get(field, "")) for field in present_fields if pd.notna(row.get(field)))
                
                if text:
                    # Get potential categories for this group
                    potential_categories = self.taxonomy_mapper.get_categories_for_group(std_group)
                    
                    # Find best matching category
                    best_category = self._find_best_category(text, potential_categories, std_group)
                    
                    if best_category:
                        result_df.at[idx, "Category"] = best_category
        
        return result_df
    
    def _infer_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Infer categories from product data when no category fields exist
        
        Args:
            data (pd.DataFrame): Input data without category fields
            
        Returns:
            pd.DataFrame: Data with inferred categories
        """
        result_df = data.copy()
        
        # Initialize category columns if not present
        if "Category" not in result_df.columns:
            result_df["Category"] = None
            
        if "Category Group" not in result_df.columns:
            result_df["Category Group"] = None
            
        # Look for description fields
        description_fields = ["Short Description", "Long Description", "Model", "Manufacturer"]
        present_fields = [f for f in description_fields if f in result_df.columns]
        
        if not present_fields:
            self.logger.warning("No description fields found for category inference")
            return result_df
            
        # Use AI to batch process products
        batch_size = 25  # Process in batches to avoid context limits
        
        for batch_start in range(0, len(result_df), batch_size):
            batch_end = min(batch_start + batch_size, len(result_df))
            batch = result_df.iloc[batch_start:batch_end]
            
            # Prepare data for the prompt
            products = []
            for idx, row in batch.iterrows():
                product_info = {}
                for field in present_fields:
                    if pd.notna(row.get(field)):
                        product_info[field] = row.get(field)
                products.append({"idx": idx, "data": product_info})
                
            if not products:
                continue
                
            # Generate prompt
            prompt = get_category_extraction_prompt(products)
            
            # Get AI response
            response = self.llm_client.generate_response(prompt)
            
            # Parse and apply categories
            category_results = self._parse_category_response(response)
            
            for idx, categories in category_results.items():
                if isinstance(idx, str):
                    # Convert string index to integer if needed
                    try:
                        idx = int(idx)
                    except ValueError:
                        continue
                        
                if idx >= len(result_df):
                    continue
                    
                result_df.at[idx, "Category"] = categories.get("category")
                result_df.at[idx, "Category Group"] = categories.get("category_group")
        
        # Finally, standardize all categories
        return self._standardize_existing_categories(result_df)
    
    def _find_best_category(self, text: str, categories: List[str], category_group: str) -> Optional[str]:
        """
        Find the best matching category for text
        
        Args:
            text (str): Product description text
            categories (List[str]): Potential categories
            category_group (str): Category group
            
        Returns:
            Optional[str]: Best matching category or None
        """
        if not categories:
            return None
            
        # Simple keyword matching approach
        text = text.lower()
        scores = {}
        
        for category in categories:
            # Convert category to search terms
            terms = re.findall(r'\b[a-z]{3,}\b', category.lower())
            
            # Score based on term presence
            score = sum(1 for term in terms if term in text)
            scores[category] = score
            
        # Get best match
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            if best_match[1] > 0:
                return best_match[0]
                
        # Fall back to first category if no clear match
        return categories[0]
    
    def _parse_category_response(self, response: str) -> Dict[int, Dict[str, str]]:
        """
        Parse the AI response for category extraction
        
        Args:
            response (str): AI model response
            
        Returns:
            Dict[int, Dict[str, str]]: Parsed categories by index
        """
        result = {}
        
        # Try to extract JSON from response
        try:
            import json
            import re
            
            # Look for JSON block
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response)
            
            if matches:
                data = json.loads(matches[0])
                if isinstance(data, dict):
                    for idx, categories in data.items():
                        result[int(idx)] = categories
                    return result
                    
            # Try to extract JSON without code blocks
            data = json.loads(response)
            if isinstance(data, dict):
                for idx, categories in data.items():
                    result[int(idx)] = categories
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from response: {str(e)}")
            
        # Fallback: try to extract categories line by line
        try:
            lines = response.strip().split('\n')
            current_idx = None
            
            for line in lines:
                # Look for index markers
                idx_match = re.search(r'Product[:\s]+(\d+)', line)
                if idx_match:
                    current_idx = int(idx_match.group(1))
                    result[current_idx] = {}
                    continue
                    
                # Look for category information
                if current_idx is not None:
                    if 'category group' in line.lower() or 'main category' in line.lower():
                        group_match = re.search(r'[:\s]+([\w\s]+)$', line)
                        if group_match:
                            result[current_idx]['category_group'] = group_match.group(1).strip()
                            
                    elif 'category' in line.lower() or 'subcategory' in line.lower():
                        cat_match = re.search(r'[:\s]+([\w\s]+)$', line)
                        if cat_match:
                            result[current_idx]['category'] = cat_match.group(1).strip()
        except Exception as e:
            self.logger.error(f"Failed to parse categories from text: {str(e)}")
            
        return result
