# utils/helpers/string_helpers.py
"""
String manipulation utilities
"""
import re
from typing import List, Dict, Any, Optional
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize text by removing special characters, excess whitespace, etc.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
        
    # Convert to unicode normalized form
    text = unicodedata.normalize('NFKD', text)
    
    # Remove control characters
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('C'))
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_html(html: str) -> str:
    """
    Remove HTML tags from text
    
    Args:
        html (str): HTML text
        
    Returns:
        str: Clean text
    """
    if not html:
        return ""
        
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', html)
    
    # Handle HTML entities
    clean = clean.replace('&nbsp;', ' ')
    clean = clean.replace('&amp;', '&')
    clean = clean.replace('&lt;', '<')
    clean = clean.replace('&gt;', '>')
    clean = clean.replace('&quot;', '"')
    clean = clean.replace('&apos;', "'")
    
    # Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    return clean

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text
    
    Args:
        text (str): Input text
        min_length (int): Minimum keyword length
        
    Returns:
        List[str]: Extracted keywords
    """
    if not text:
        return []
        
    # Normalize text
    normalized = normalize_text(text).lower()
    
    # Extract words
    words = re.findall(r'\b[a-z]{%d,}\b' % min_length, normalized)
    
    # Remove common stop words
    stop_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'has', 'are', 'were', 'was', 'will', 'would'}
    keywords = [word for word in words if word not in stop_words]
    
    return keywords

def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug
    
    Args:
        text (str): Input text
        
    Returns:
        str: URL-friendly slug
    """
    if not text:
        return ""
        
    # Normalize text
    slug = unicodedata.normalize('NFKD', text.lower())
    
    # Remove non-alphanumeric characters
    slug = re.sub(r'[^\w\s-]', '', slug)
    
    # Replace whitespace with hyphens
    slug = re.sub(r'\s+', '-', slug).strip('-')
    
    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    
    return slug