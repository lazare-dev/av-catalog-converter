# core/file_parser/pdf_parser.py
"""
PDF-specific parsing implementation
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import io
import pytesseract
from PIL import Image
import tempfile
import subprocess
import os

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import tabula
except ImportError:
    tabula = None

from core.file_parser.base_parser import BaseParser
from config.settings import PARSER_CONFIG

class PDFParser(BaseParser):
    """Parser for PDF files with table extraction capabilities"""
    
    def __init__(self, file_path):
        """
        Initialize the PDF parser
        
        Args:
            file_path (str): Path to the PDF file
        """
        super().__init__(file_path)
        self.config = PARSER_CONFIG['pdf']
        self.logger = logging.getLogger(__name__)
        
        if pdfplumber is None and tabula is None:
            self.logger.error("PDF parsing requires either pdfplumber or tabula-py libraries")
            raise ImportError("PDF parsing requires either pdfplumber or tabula-py libraries")
    
    def _extract_tables_with_pdfplumber(self):
        """
        Extract tables using pdfplumber
        
        Returns:
            list: List of pandas DataFrames, one per table
        """
        tables = []
        
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    self.logger.debug(f"Processing page {i+1}/{len(pdf.pages)}")
                    
                    # Extract tables from the page
                    page_tables = page.extract_tables()
                    
                    for j, table in enumerate(page_tables):
                        if not table:
                            continue
                            
                        # Convert to DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0])
                        
                        # Clean up DataFrame
                        df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')
                        
                        if len(df) > 0:
                            self.logger.debug(f"Extracted table {j+1} from page {i+1} with {len(df)} rows")
                            tables.append(df)
                
        except Exception as e:
            self.logger.error(f"Error extracting tables with pdfplumber: {str(e)}")
            
        return tables
    
    def _extract_tables_with_tabula(self):
        """
        Extract tables using tabula-py
        
        Returns:
            list: List of pandas DataFrames, one per table
        """
        tables = []
        
        try:
            # Extract all tables from the PDF
            dfs = tabula.read_pdf(
                self.file_path,
                pages='all',
                multiple_tables=True,
                guess=True,
                lattice=True
            )
            
            for i, df in enumerate(dfs):
                if len(df) > 0:
                    # Clean up DataFrame
                    df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')
                    
                    if len(df) > 0:
                        self.logger.debug(f"Extracted table {i+1} with {len(df)} rows using tabula")
                        tables.append(df)
                
        except Exception as e:
            self.logger.error(f"Error extracting tables with tabula: {str(e)}")
            
        return tables
    
    def _perform_ocr_if_needed(self, tables):
        """
        Perform OCR if configured and no tables were found
        
        Args:
            tables (list): List of tables extracted from PDF
            
        Returns:
            list: Updated list of tables
        """
        if not self.config['ocr_enabled'] or len(tables) > 0:
            return tables
            
        self.logger.info("No tables found with PDF extraction, attempting OCR")
        
        try:
            # Convert PDF to images
            images = self._convert_pdf_to_images()
            
            # Process each image with OCR
            for i, img in enumerate(images):
                self.logger.debug(f"Performing OCR on page {i+1}")
                
                # Extract text from image
                text = pytesseract.image_to_string(img)
                
                # Extract tables from text using image_to_data
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
                
                # Group by line number to reconstruct rows
                if 'line_num' in data.columns and not data.empty:
                    grouped = data.groupby('line_num')
                    
                    # Create a DataFrame with rows corresponding to lines
                    rows = []
                    for line_num, group in grouped:
                        text_line = ' '.join(group['text'].astype(str))
                        rows.append([text_line])
                    
                    if rows:
                        ocr_df = pd.DataFrame(rows)
                        tables.append(ocr_df)
            
        except Exception as e:
            self.logger.error(f"Error performing OCR: {str(e)}")
            
        return tables
    
    def _convert_pdf_to_images(self):
        """Convert PDF pages to PIL Image objects"""
        images = []
        
        try:
            # Primary conversion method
            # Implementation details...
            pass  # Add this placeholder if the implementation is missing
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}")
            
            # Fallback using pdfplumber if available
            if pdfplumber is not None:
                self.logger.info("Attempting fallback conversion with pdfplumber")
                try:
                    with pdfplumber.open(self.file_path) as pdf:
                        for page in pdf.pages:
                            img = page.to_image()
                            pil_img = Image.open(io.BytesIO(img.original.tobytes()))
                            images.append(pil_img)
                    if images:
                        self.logger.info("Fallback conversion successful")
                        return images
                except Exception as inner_e:
                    self.logger.error(f"Fallback PDF conversion failed: {str(inner_e)}")
            
            # If we get here, both methods failed
            self.logger.warning("All PDF conversion methods failed")
        
        return images
    
    def _select_best_table(self, tables):
        """
        Select the most appropriate table from extracted tables
        
        Args:
            tables (list): List of extracted tables
            
        Returns:
            pd.DataFrame: Best table or merged table
        """
        if not tables:
            return pd.DataFrame()
            
        if len(tables) == 1:
            return tables[0]
            
        # Scoring function for tables
        def score_table(df):
            # Prefer tables with more data
            row_score = min(len(df), 100)  # Cap at 100 for very large tables
            col_score = min(len(df.columns), 20)  # Cap at 20 for many columns
            
            # Prefer tables with standard headers
            header_keywords = ['sku', 'item', 'product', 'description', 'price', 'model']
            header_score = sum(1 for col in df.columns if any(
                kw in str(col).lower() for kw in header_keywords
            ))
            
            return row_score + col_score * 2 + header_score * 5
        
        # Score each table
        table_scores = [(table, score_table(table)) for table in tables]
        best_table = max(table_scores, key=lambda x: x[1])[0]
        
        return best_table
    
    def parse(self):
        """
        Parse the PDF file into a pandas DataFrame
        
        Returns:
            pd.DataFrame: Parsed data
        """
        self.logger.info(f"Parsing PDF file: {self.file_path}")
        
        # Extract tables using available methods
        tables = []
        
        if pdfplumber is not None:
            self.logger.debug("Attempting extraction with pdfplumber")
            tables.extend(self._extract_tables_with_pdfplumber())
            
        if not tables and tabula is not None:
            self.logger.debug("Attempting extraction with tabula")
            tables.extend(self._extract_tables_with_tabula())
            
        # Try OCR if no tables found and OCR is enabled
        tables = self._perform_ocr_if_needed(tables)
        
        if not tables:
            self.logger.warning("No tables extracted from PDF")
            return pd.DataFrame()
            
        # Select or merge tables
        final_table = self._select_best_table(tables)
        
        # Clean up column names
        final_table.columns = [str(col).strip() for col in final_table.columns]
        
        self.logger.info(f"Successfully extracted table with {final_table.shape[0]} rows and {final_table.shape[1]} columns")
        return final_table
    
    def get_headers(self):
        """
        Extract headers from the PDF file
        
        Returns:
            list: List of header names
        """
        try:
            # Parse to get the table
            df = self.parse()
            
            # Return headers if available
            if not df.empty:
                headers = list(df.columns)
                self.logger.debug(f"Extracted headers: {headers}")
                return headers
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting PDF headers: {str(e)}")
            return []
    
    def get_sample(self, rows=5):
        """
        Get a sample of data for preview
        
        Args:
            rows (int): Number of rows to sample
            
        Returns:
            pd.DataFrame: Sample data
        """
        try:
            # Parse to get the table
            df = self.parse()
            
            # Return sample
            if not df.empty:
                return df.head(rows)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting PDF sample: {str(e)}")
            return pd.DataFrame()
