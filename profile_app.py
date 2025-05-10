#!/usr/bin/env python
"""
Script to profile the application performance
"""
import os
import sys
import argparse
import time
from pathlib import Path
import pandas as pd
import cProfile
import pstats
import io

from utils.profiling.profiler import Profiler
from utils.logging.logger import Logger
from core.file_parser.parser_factory import ParserFactory
from services.structure.structure_analyzer import StructureAnalyzer
from services.mapping.field_mapper import FieldMapper
from services.category.category_extractor import CategoryExtractor
from services.normalization.value_normalizer import ValueNormalizer

# Initialize logger
logger = Logger.get_logger(__name__)

def profile_parsing(file_path: Path) -> None:
    """
    Profile the file parsing process
    
    Args:
        file_path: Path to the file to parse
    """
    logger.info(f"Profiling file parsing for {file_path}")
    
    # Profile parser creation
    with Profiler(name="parser_creation"):
        parser = ParserFactory.create_parser(file_path)
    
    # Profile parsing
    with Profiler(name="file_parsing") as profiler:
        data = parser.parse()
    
    # Save profiling results
    profiler.save_stats(f"parsing_{file_path.name}.prof")
    
    # Log basic info about the parsed data
    if hasattr(data, 'shape'):
        logger.info(f"Parsed data shape: {data.shape}")
    
    return data

def profile_structure_analysis(data: pd.DataFrame) -> None:
    """
    Profile the structure analysis process
    
    Args:
        data: DataFrame to analyze
    """
    logger.info("Profiling structure analysis")
    
    analyzer = StructureAnalyzer()
    
    with Profiler(name="structure_analysis") as profiler:
        structure_info = analyzer.analyze(data)
    
    # Save profiling results
    profiler.save_stats("structure_analysis.prof")
    
    # Log basic info about the structure
    logger.info(f"Structure analysis results: {len(structure_info.get('column_types', {}))} columns analyzed")
    
    return structure_info

def profile_field_mapping(data: pd.DataFrame, structure_info: dict) -> None:
    """
    Profile the field mapping process
    
    Args:
        data: DataFrame to map
        structure_info: Structure information from the analyzer
    """
    logger.info("Profiling field mapping")
    
    mapper = FieldMapper()
    
    with Profiler(name="field_mapping") as profiler:
        mapped_data = mapper.map(data, structure_info)
    
    # Save profiling results
    profiler.save_stats("field_mapping.prof")
    
    # Log basic info about the mapping
    if hasattr(mapped_data, 'shape'):
        logger.info(f"Mapped data shape: {mapped_data.shape}")
    
    return mapped_data

def profile_category_extraction(data: pd.DataFrame) -> None:
    """
    Profile the category extraction process
    
    Args:
        data: DataFrame to extract categories from
    """
    logger.info("Profiling category extraction")
    
    extractor = CategoryExtractor()
    
    with Profiler(name="category_extraction") as profiler:
        categories = extractor.extract(data)
    
    # Save profiling results
    profiler.save_stats("category_extraction.prof")
    
    # Log basic info about the categories
    logger.info(f"Extracted {len(categories)} categories")
    
    return categories

def profile_value_normalization(data: pd.DataFrame) -> None:
    """
    Profile the value normalization process
    
    Args:
        data: DataFrame to normalize
    """
    logger.info("Profiling value normalization")
    
    normalizer = ValueNormalizer()
    
    with Profiler(name="value_normalization") as profiler:
        normalized_data = normalizer.normalize(data)
    
    # Save profiling results
    profiler.save_stats("value_normalization.prof")
    
    # Log basic info about the normalization
    if hasattr(normalized_data, 'shape'):
        logger.info(f"Normalized data shape: {normalized_data.shape}")
    
    return normalized_data

def profile_end_to_end(file_path: Path) -> None:
    """
    Profile the entire processing pipeline
    
    Args:
        file_path: Path to the file to process
    """
    logger.info(f"Profiling end-to-end processing for {file_path}")
    
    with Profiler(name="end_to_end_processing") as profiler:
        # Parse file
        parser = ParserFactory.create_parser(file_path)
        data = parser.parse()
        
        # Analyze structure
        analyzer = StructureAnalyzer()
        structure_info = analyzer.analyze(data)
        
        # Map fields
        mapper = FieldMapper()
        mapped_data = mapper.map(data, structure_info)
        
        # Extract categories
        extractor = CategoryExtractor()
        categories = extractor.extract(mapped_data)
        
        # Normalize values
        normalizer = ValueNormalizer()
        normalized_data = normalizer.normalize(mapped_data)
    
    # Save profiling results
    profiler.save_stats(f"end_to_end_{file_path.name}.prof")
    
    # Log basic info about the result
    if hasattr(normalized_data, 'shape'):
        logger.info(f"Final data shape: {normalized_data.shape}")
    
    return normalized_data

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Profile the AV Catalog Converter')
    parser.add_argument('--file', '-f', required=True, help='Path to the file to process')
    parser.add_argument('--component', '-c', choices=['parsing', 'structure', 'mapping', 'category', 'normalization', 'all'], 
                        default='all', help='Component to profile')
    args = parser.parse_args()
    
    file_path = Path(args.file)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 1
    
    # Create profiling directory
    profiling_dir = Path("profiling")
    profiling_dir.mkdir(exist_ok=True)
    
    # Profile the specified component
    if args.component == 'parsing' or args.component == 'all':
        data = profile_parsing(file_path)
    else:
        # Parse the file without profiling
        logger.info(f"Parsing file: {file_path}")
        parser = ParserFactory.create_parser(file_path)
        data = parser.parse()
    
    if args.component == 'structure' or args.component == 'all':
        structure_info = profile_structure_analysis(data)
    elif args.component in ['mapping', 'category', 'normalization', 'all']:
        # Analyze structure without profiling
        logger.info("Analyzing structure")
        analyzer = StructureAnalyzer()
        structure_info = analyzer.analyze(data)
    
    if args.component == 'mapping' or args.component == 'all':
        mapped_data = profile_field_mapping(data, structure_info)
    elif args.component in ['category', 'normalization', 'all']:
        # Map fields without profiling
        logger.info("Mapping fields")
        mapper = FieldMapper()
        mapped_data = mapper.map(data, structure_info)
    
    if args.component == 'category' or args.component == 'all':
        categories = profile_category_extraction(mapped_data)
    
    if args.component == 'normalization' or args.component == 'all':
        normalized_data = profile_value_normalization(mapped_data)
    
    if args.component == 'all':
        # Profile the entire pipeline
        profile_end_to_end(file_path)
    
    logger.info("Profiling complete. Results saved to the 'profiling' directory.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
