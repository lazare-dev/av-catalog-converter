# services/category/taxonomy_mapper.py
"""
Industry taxonomy mapping
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional
from difflib import SequenceMatcher
import re

class TaxonomyMapper:
    """Maps categories to standard AV industry taxonomy"""
    
    def __init__(self):
        """Initialize the taxonomy mapper"""
        self.logger = logging.getLogger(__name__)
        
        # Define standard AV industry taxonomy
        self._init_standard_taxonomy()
    
    def _init_standard_taxonomy(self):
        """Initialize the standard taxonomy structure"""
        # Standard category groups and their categories
        self.standard_taxonomy = {
            "Display": [
                "Projectors",
                "Flat Panel Displays",
                "Video Walls",
                "Screens",
                "Digital Signage",
                "Display Mounts"
            ],
            "Audio": [
                "Speakers",
                "Amplifiers",
                "Microphones",
                "Audio Processors",
                "Audio Mixers",
                "Headphones",
                "Audio Recorders"
            ],
            "Video": [
                "Cameras",
                "Video Processors",
                "Video Switchers",
                "Media Players",
                "Video Recorders",
                "Video Encoders/Decoders"
            ],
            "Control": [
                "Control Systems",
                "Touch Panels",
                "Remote Controls",
                "Control Processors",
                "Automation Systems"
            ],
            "Infrastructure": [
                "Cables",
                "Connectors",
                "Racks",
                "Power Management",
                "Signal Converters",
                "Signal Extenders",
                "KVM Systems"
            ],
            "Conferencing": [
                "Video Conferencing Systems",
                "Conference Phones",
                "Wireless Presentation Systems",
                "Collaboration Tools",
                "Webcams"
            ],
            "Accessories": [
                "Brackets",
                "Adapters",
                "Cases",
                "Cleaning Supplies",
                "Installation Tools"
            ]
        }
        
        # Create flat list for easy lookup
        self.all_standard_categories = []
        self.category_to_group_map = {}
        
        for group, categories in self.standard_taxonomy.items():
            for category in categories:
                self.all_standard_categories.append(category)
                self.category_to_group_map[category] = group
        
        # Common synonyms and alternative terms
        self.category_synonyms = {
            "Projectors": ["Beamers", "Video Projectors", "Projection Systems"],
            "Flat Panel Displays": ["TVs", "Monitors", "LCD Displays", "LED Displays", "OLED Displays"],
            "Screens": ["Projection Screens", "Motorized Screens", "Fixed Screens"],
            "Speakers": ["Loudspeakers", "Sound Reinforcement", "PA Speakers", "Ceiling Speakers", "In-Wall Speakers"],
            "Cameras": ["PTZ Cameras", "IP Cameras", "CCTV Cameras", "Webcams"],
            "Cables": ["Cabling", "Wire", "HDMI Cables", "Network Cables", "Audio Cables"],
            "Control Systems": ["Control Units", "Automation Systems", "Room Control"],
            "Signal Converters": ["Format Converters", "Signal Transformers", "Scalers"],
            "Signal Extenders": ["Extenders", "Baluns", "Over IP Extenders", "Signal Distribution"],
            "Video Conferencing Systems": ["VC Systems", "Telepresence", "Video Conference Units"]
        }
        
        self.group_synonyms = {
            "Display": ["Displays", "Visual", "Video Display", "Screens"],
            "Audio": ["Sound", "Audio Equipment", "Sound Systems", "Audio Systems"],
            "Video": ["Video Equipment", "Video Systems", "Visual", "Video Production"],
            "Control": ["Automation", "Control Systems", "Controllers", "Remote"],
            "Infrastructure": ["AV Infrastructure", "Backbone", "Foundation", "Connection", "Connectivity"],
            "Conferencing": ["Meeting Room", "Collaboration", "Communications", "Conference Systems"],
            "Accessories": ["Add-ons", "Peripherals", "Supplies", "Auxiliary"]
        }
    
    def map_to_standard_taxonomy(self, categories: List[str] = None, 
                               category_groups: List[str] = None,
                               infer_groups: bool = False,
                               infer_categories: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Map custom categories to standard taxonomy
        
        Args:
            categories (List[str], optional): List of categories to map
            category_groups (List[str], optional): List of category groups to map
            infer_groups (bool): Whether to infer groups for categories
            infer_categories (bool): Whether to infer categories for groups
            
        Returns:
            Dict[str, Dict[str, str]]: Mapping of original values to standard ones
        """
        mapping = {}
        
        # Map categories
        if categories:
            for category in categories:
                if not category or pd.isna(category):
                    continue
                    
                std_category = self._find_matching_category(category)
                
                mapping[category] = {
                    "original": category,
                    "standard_category": std_category
                }
                
                # Add group if requested or category was mapped
                if infer_groups or std_category:
                    group = self.category_to_group_map.get(std_category)
                    if group:
                        mapping[category]["standard_group"] = group
        
        # Map groups
        if category_groups:
            for group in category_groups:
                if not group or pd.isna(group):
                    continue
                    
                std_group = self._find_matching_group(group)
                
                if group not in mapping:
                    mapping[group] = {
                        "original": group,
                        "standard_group": std_group
                    }
                else:
                    mapping[group]["standard_group"] = std_group
                
                # Add a default category if requested
                if infer_categories and std_group:
                    categories = self.standard_taxonomy.get(std_group, [])
                    if categories:
                        mapping[group]["standard_category"] = categories[0]
        
        return mapping
    
    def get_categories_for_group(self, group: str) -> List[str]:
        """
        Get standard categories for a group
        
        Args:
            group (str): Category group
            
        Returns:
            List[str]: List of standard categories
        """
        return self.standard_taxonomy.get(group, [])
    
    def _find_matching_category(self, category: str) -> Optional[str]:
        """
        Find the best matching standard category
        
        Args:
            category (str): Input category
            
        Returns:
            Optional[str]: Matching standard category or None
        """
        if not category:
            return None
            
        category = str(category).strip()
        
        # Check for exact match
        for std_category in self.all_standard_categories:
            if category.lower() == std_category.lower():
                return std_category
                
        # Check against synonyms
        for std_category, synonyms in self.category_synonyms.items():
            if any(category.lower() == syn.lower() for syn in synonyms):
                return std_category
                
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        # Normalize input for matching
        category_norm = self._normalize_for_matching(category)
        
        for std_category in self.all_standard_categories:
            # Check direct match
            std_norm = self._normalize_for_matching(std_category)
            score = SequenceMatcher(None, category_norm, std_norm).ratio()
            
            # Check synonym matches
            for synonym in self.category_synonyms.get(std_category, []):
                syn_norm = self._normalize_for_matching(synonym)
                syn_score = SequenceMatcher(None, category_norm, syn_norm).ratio()
                score = max(score, syn_score)
                
            # Check for keyword overlap
            keywords = [self._normalize_for_matching(word) 
                       for word in re.findall(r'\b[a-zA-Z]{3,}\b', std_category)]
            
            keyword_in_category = any(keyword in category_norm for keyword in keywords)
            if keyword_in_category:
                score += 0.1
                
            if score > best_score and score > 0.7:  # Threshold to avoid poor matches
                best_score = score
                best_match = std_category
                
        return best_match
    
    def _find_matching_group(self, group: str) -> Optional[str]:
        """
        Find the best matching standard category group
        
        Args:
            group (str): Input category group
            
        Returns:
            Optional[str]: Matching standard group or None
        """
        if not group:
            return None
            
        group = str(group).strip()
        
        # Check for exact match
        for std_group in self.standard_taxonomy.keys():
            if group.lower() == std_group.lower():
                return std_group
                
        # Check against synonyms
        for std_group, synonyms in self.group_synonyms.items():
            if any(group.lower() == syn.lower() for syn in synonyms):
                return std_group
                
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        # Normalize input for matching
        group_norm = self._normalize_for_matching(group)
        
        for std_group in self.standard_taxonomy.keys():
            # Check direct match
            std_norm = self._normalize_for_matching(std_group)
            score = SequenceMatcher(None, group_norm, std_norm).ratio()
            
            # Check synonym matches
            for synonym in self.group_synonyms.get(std_group, []):
                syn_norm = self._normalize_for_matching(synonym)
                syn_score = SequenceMatcher(None, group_norm, syn_norm).ratio()
                score = max(score, syn_score)
                
            if score > best_score and score > 0.6:  # Slightly lower threshold for groups
                best_score = score
                best_match = std_group
                
        return best_match
    
    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for fuzzy matching
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized