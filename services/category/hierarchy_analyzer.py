# services/category/hierarchy_analyzer.py
"""
Hierarchy detection for categorization
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

class HierarchyAnalyzer:
    """Analyzes category hierarchies in catalogs"""
    
    def __init__(self):
        """Initialize the hierarchy analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def detect_hierarchy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect category hierarchies in catalog data
        
        Args:
            data (pd.DataFrame): Input data with category fields
            
        Returns:
            Dict[str, Any]: Hierarchy information
        """
        self.logger.info("Detecting category hierarchy")
        
        # Find potential category columns
        category_columns = self._find_category_columns(data)
        
        if not category_columns:
            return {"has_hierarchy": False}
            
        # Analyze relationships between columns
        hierarchy = self._analyze_column_relationships(data, category_columns)
        
        return hierarchy
    
    def _find_category_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Find columns that may contain category information
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[str]: List of potential category columns
        """
        category_columns = []
        
        # Check for standard category fields
        if "Category" in data.columns:
            category_columns.append("Category")
            
        if "Category Group" in data.columns:
            category_columns.append("Category Group")
            
        # Look for other potential category columns
        for col in data.columns:
            col_lower = str(col).lower()
            
            # Skip already identified columns
            if col in category_columns:
                continue
                
            # Skip non-category columns
            if any(kw in col_lower for kw in ['category', 'group', 'department', 'class', 'type']):
                # Category columns typically have low cardinality
                unique_ratio = data[col].nunique() / len(data)
                
                if unique_ratio < 0.3:  # Less than 30% unique values
                    category_columns.append(col)
        
        return category_columns
    
    def _analyze_column_relationships(self, data: pd.DataFrame, 
                                   category_columns: List[str]) -> Dict[str, Any]:
        """
        Analyze relationships between potential category columns
        
        Args:
            data (pd.DataFrame): Input data
            category_columns (List[str]): Potential category columns
            
        Returns:
            Dict[str, Any]: Hierarchy information
        """
        if len(category_columns) < 2:
            return {"has_hierarchy": False}
            
        # Initialize hierarchy info
        hierarchy = {
            "has_hierarchy": False,
            "hierarchy_levels": [],
            "parent_child_map": {},
            "level_counts": {}
        }
        
        # Try all pairs of columns to find hierarchical relationships
        for i in range(len(category_columns)):
            for j in range(i+1, len(category_columns)):
                col1 = category_columns[i]
                col2 = category_columns[j]
                
                relationship = self._check_column_relationship(data, col1, col2)
                
                if relationship["is_hierarchical"]:
                    hierarchy["has_hierarchy"] = True
                    
                    if relationship["parent_col"] == col1:
                        parent = col1
                        child = col2
                    else:
                        parent = col2
                        child = col1
                        
                    # Add to hierarchy levels in proper order
                    if parent not in hierarchy["hierarchy_levels"]:
                        hierarchy["hierarchy_levels"].append(parent)
                        
                    if child not in hierarchy["hierarchy_levels"]:
                        # Insert child after its parent
                        parent_idx = hierarchy["hierarchy_levels"].index(parent)
                        hierarchy["hierarchy_levels"].insert(parent_idx + 1, child)
                        
                    # Store parent-child mapping
                    hierarchy["parent_child_map"][parent] = child
                    
                    # Store level counts
                    hierarchy["level_counts"][parent] = data[parent].nunique()
                    hierarchy["level_counts"][child] = data[child].nunique()
        
        # If we identified hierarchical relationships, format the result
        if hierarchy["has_hierarchy"]:
            # Ensure hierarchy levels are in proper order (may need topological sort for complex hierarchies)
            ordered_levels = []
            visited = set()
            
            def visit(node):
                if node in visited:
                    return
                visited.add(node)
                ordered_levels.append(node)
                
                # Visit child if exists
                if node in hierarchy["parent_child_map"]:
                    visit(hierarchy["parent_child_map"][node])
            
            # Start with nodes that aren't children
            child_nodes = set(hierarchy["parent_child_map"].values())
            root_nodes = [col for col in hierarchy["hierarchy_levels"] if col not in child_nodes]
            
            for node in root_nodes:
                visit(node)
                
            hierarchy["hierarchy_levels"] = ordered_levels
            
        return hierarchy
    
    def _check_column_relationship(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Check if two columns have a hierarchical relationship
        
        Args:
            data (pd.DataFrame): Input data
            col1 (str): First column name
            col2 (str): Second column name
            
        Returns:
            Dict[str, Any]: Relationship information
        """
        # Initialize result
        result = {
            "is_hierarchical": False,
            "parent_col": None,
            "child_col": None,
            "confidence": 0.0
        }
        
        # Skip if either column has too many missing values
        if data[col1].isna().sum() > len(data) * 0.3 or data[col2].isna().sum() > len(data) * 0.3:
            return result
            
        # Group by each column and count unique values in the other
        grouped1 = data.groupby(col1)[col2].nunique()
        grouped2 = data.groupby(col2)[col1].nunique()
        
        # Average number of unique values when grouping
        avg_unique1 = grouped1.mean()  # avg unique col2 values per col1 value
        avg_unique2 = grouped2.mean()  # avg unique col1 values per col2 value
        
        # Columns are hierarchical if one column has low average unique values
        # Parent -> Child: Each parent has multiple children, 
        # but each child has only one parent
        
        if avg_unique1 > 1.5 and avg_unique2 < 1.5:
            # col1 is parent, col2 is child
            result["is_hierarchical"] = True
            result["parent_col"] = col1
            result["child_col"] = col2
            result["confidence"] = 0.8
            
        elif avg_unique2 > 1.5 and avg_unique1 < 1.5:
            # col2 is parent, col1 is child
            result["is_hierarchical"] = True
            result["parent_col"] = col2
            result["child_col"] = col1
            result["confidence"] = 0.8
            
        return result
    
    def flatten_hierarchy(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Flatten hierarchical categories into standard format
        
        Args:
            data (pd.DataFrame): Input data with hierarchical categories
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 
                Flattened data and hierarchy mapping information
        """
        # Detect hierarchy
        hierarchy_info = self.detect_hierarchy(data)
        
        if not hierarchy_info["has_hierarchy"]:
            return data, {}
            
        # Create output dataframe
        result_df = data.copy()
        
        # If standard category columns don't exist, add them
        if "Category Group" not in result_df.columns:
            result_df["Category Group"] = None
            
        if "Category" not in result_df.columns:
            result_df["Category"] = None
            
        # Map hierarchical columns to standard ones
        hierarchy_levels = hierarchy_info["hierarchy_levels"]
        
        if len(hierarchy_levels) >= 2:
            # Map top level to Category Group
            top_level = hierarchy_levels[0]
            result_df["Category Group"] = data[top_level]
            
            # Map second level to Category
            second_level = hierarchy_levels[1]
            result_df["Category"] = data[second_level]
            
        elif len(hierarchy_levels) == 1:
            # Only one level - map to Category and derive Group
            result_df["Category"] = data[hierarchy_levels[0]]
            
        # Return mapping information
        mapping_info = {
            "original_hierarchy": hierarchy_info,
            "mapping": {
                "Category Group": hierarchy_levels[0] if len(hierarchy_levels) >= 1 else None,
                "Category": hierarchy_levels[1] if len(hierarchy_levels) >= 2 else hierarchy_levels[0]
            }
        }
        
        return result_df, mapping_info
