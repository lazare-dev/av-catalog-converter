# core/file_parser/excel_parser.py
"""
Excel-specific parsing implementation
"""
import pandas as pd
import logging
from pathlib import Path

from core.file_parser.base_parser import BaseParser
from config.settings import PARSER_CONFIG

class ExcelParser(BaseParser):
    """Parser for Excel files (XLSX, XLS)"""
    
    def __init__(self, file_path):
        """
        Initialize the Excel parser
        
        Args:
            file_path (str): Path to the Excel file
        """
        super().__init__(file_path)
        self.config = PARSER_CONFIG['excel']
        self.sheet_name = None
        self.header_row = 0
        self.logger = logging.getLogger(__name__)
    
    def _get_sheet_names(self):
        """
        Get all sheet names from the Excel file
        
        Returns:
            list: List of sheet names
        """
        try:
            xl = pd.ExcelFile(self.file_path)
            sheet_names = xl.sheet_names
            self.logger.debug(f"Found sheets: {sheet_names}")
            return sheet_names
        except Exception as e:
            self.logger.error(f"Error reading Excel sheets: {str(e)}")
            return []
    
    def _select_best_sheet(self):
        """
        Select the most relevant sheet from the workbook
        
        Returns:
            str: Selected sheet name
        """
        sheet_names = self._get_sheet_names()
        
        if not sheet_names:
            raise ValueError("No sheets found in Excel file")
            
        if len(sheet_names) == 1:
            # Only one sheet available
            return sheet_names[0]
        
        # Look for sheets with catalog-like names
        catalog_keywords = ['product', 'catalog', 'catalogue', 'price', 'inventory', 'stock', 'item']
        for keyword in catalog_keywords:
            for sheet in sheet_names:
                if keyword.lower() in sheet.lower():
                    self.logger.debug(f"Selected sheet '{sheet}' based on keyword '{keyword}'")
                    return sheet
        
        # If no catalog-like sheet names, look for the sheet with most data
        max_rows = 0
        selected_sheet = sheet_names[0]
        
        for sheet in sheet_names:
            try:
                # Read a sample to determine row count
                sample = pd.read_excel(self.file_path, sheet_name=sheet, nrows=1)
                sheet_info = pd.read_excel(self.file_path, sheet_name=sheet, header=None, nrows=0)
                num_columns = len(sheet_info.columns)
                
                # Read last row to determine total rows
                tail_df = pd.read_excel(self.file_path, sheet_name=sheet, skiprows=0, usecols=range(num_columns), nrows=None)
                num_rows = len(tail_df)
                
                if num_rows > max_rows:
                    max_rows = num_rows
                    selected_sheet = sheet
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing sheet '{sheet}': {str(e)}")
                continue
                
        self.logger.debug(f"Selected sheet '{selected_sheet}' with {max_rows} rows")
        return selected_sheet
    
    def _detect_header_row(self, sheet_name):
        """
        Detect which row contains the headers
        
        Args:
            sheet_name (str): Sheet name to analyze
            
        Returns:
            int: Row index (0-based) of the header row
        """
        if not self.config['header_detection']:
            return 0
            
        try:
            # Read first few rows
            sample = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None, nrows=10)
            
            # Analyze each row as a potential header
            for row_idx in range(min(5, len(sample))):
                row = sample.iloc[row_idx]
                
                # Check if row has mostly string values (typical for headers)
                str_count = sum(isinstance(val, str) for val in row if not pd.isna(val))
                non_empty = sum(1 for val in row if not pd.isna(val))
                
                # If most non-empty values are strings, it's likely a header
                if non_empty > 0 and str_count / non_empty > 0.7:
                    self.logger.debug(f"Detected header at row {row_idx}")
                    return row_idx
                    
            # Default to first row
            return 0
            
        except Exception as e:
            self.logger.warning(f"Error detecting header row: {str(e)}")
            return 0
    
    def parse(self):
        """
        Parse the Excel file into a pandas DataFrame
        
        Returns:
            pd.DataFrame: Parsed data
        """
        self.logger.info(f"Parsing Excel file: {self.file_path}")
        
        # Select the sheet to parse
        if not self.sheet_name:
            self.sheet_name = self._select_best_sheet()
            
        # Detect header row
        header_row = self._detect_header_row(self.sheet_name)
        
        try:
            # Read the Excel file
            df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=header_row
            )
            
            # Clean up column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            self.logger.info(f"Successfully parsed Excel sheet '{self.sheet_name}' with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing Excel file: {str(e)}")
            raise
    
    def get_headers(self):
        """
        Extract headers from the Excel file
        
        Returns:
            list: List of header names
        """
        # Select the sheet to use
        if not self.sheet_name:
            self.sheet_name = self._select_best_sheet()
            
        # Detect header row
        header_row = self._detect_header_row(self.sheet_name)
        
        try:
            # Read just the header row
            df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=header_row,
                nrows=0
            )
            
            headers = list(df.columns)
            self.logger.debug(f"Extracted headers: {headers}")
            return headers
            
        except Exception as e:
            self.logger.error(f"Error extracting Excel headers: {str(e)}")
            return []
    
    def get_sample(self, rows=5):
        """
        Get a sample of data for preview
        
        Args:
            rows (int): Number of rows to sample
            
        Returns:
            pd.DataFrame: Sample data
        """
        # Select the sheet to use
        if not self.sheet_name:
            self.sheet_name = self._select_best_sheet()
            
        # Detect header row
        header_row = self._detect_header_row(self.sheet_name)
        
        try:
            # Read sample rows from the file
            sample_df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=header_row,
                nrows=rows
            )
            
            return sample_df
            
        except Exception as e:
            self.logger.error(f"Error getting Excel sample: {str(e)}")
            return pd.DataFrame()