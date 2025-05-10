"""
AV Catalog Standardizer - Field Definitions
-------------------------------------------
Standard field definitions for AV catalog data.
"""

import re
from typing import Dict, Any, List, Set, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class FieldDefinition:
    """Definition of a standard field with mapping information"""
    
    name: str
    description: str
    required: bool = False
    data_type: str = "string"
    default_value: Any = None
    mapping_hints: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    common_names: List[str] = field(default_factory=list)
    example_values: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize any missing fields with defaults"""
        if not self.mapping_hints:
            self.mapping_hints = [f"Often labeled as '{self.name}' or similar"]
        
        if not self.patterns:
            self.patterns = []
            
        if not self.common_names:
            name_parts = re.findall(r'[A-Z][a-z]*', self.name)
            self.common_names = [self.name.lower()] + [part.lower() for part in name_parts]
            
        if not self.example_values:
            self.example_values = []

# Define all standard fields with mapping information
FIELD_DEFINITIONS = {
    "SKU": FieldDefinition(
        name="SKU",
        description="Unique identifier for the product in your system",
        required=True,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'SKU', 'Item Number', 'Product ID', or 'Product Code'",
            "Usually alphanumeric and unique across products",
            "May contain dashes or underscores as separators"
        ],
        patterns=[
            r'^[A-Za-z0-9\-_]{3,20}$',
            r'^[A-Z]{2,5}-\d{3,6}$'
        ],
        common_names=["sku", "item_number", "product_id", "product_code", "item_id", "item_sku", "id"],
        example_values=["AV-12345", "SPKR-001", "MIC-2022-B"]
    ),
    
    "Short_Description": FieldDefinition(
        name="Short_Description",
        description="Brief product name or title",
        required=True,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Product Name', 'Title', 'Short Description', or 'Name'",
            "Usually 5-100 characters in length",
            "Contains the essential product identifier for humans"
        ],
        common_names=["short_description", "product_name", "title", "name", "item_name", "product", "short_desc"],
        example_values=["4K HDR Projector", "Wireless Conference Microphone", "HDMI 2.1 Cable 3m"]
    ),
    
    "Long_Description": FieldDefinition(
        name="Long_Description",
        description="Detailed product description with features",
        required=False,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Description', 'Long Description', 'Full Description', or 'Details'",
            "Usually longer than 100 characters",
            "Contains detailed product information and features"
        ],
        common_names=["long_description", "description", "full_description", "details", "product_description", "desc", "long_desc"],
        example_values=["The 4K HDR projector offers stunning image quality with true 4K resolution..."]
    ),
    
    "Model": FieldDefinition(
        name="Model",
        description="Manufacturer's model number/name",
        required=False,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Model', 'Model Number', or 'Model Name'",
            "May follow manufacturer-specific patterns",
            "Different from both SKU and Manufacturer SKU"
        ],
        patterns=[
            r'^[A-Z0-9]{2,}[\-]?\d{2,}$',
            r'^[A-Z]{1,3}[\-]?\d{3,}$'
        ],
        common_names=["model", "model_number", "model_name", "model_no"],
        example_values=["VPL-VW325ES", "MX-50", "XDR-55A80J"]
    ),
    
    "Category_Group": FieldDefinition(
        name="Category_Group",
        description="Top-level product category",
        required=False,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Category Group', 'Product Group', 'Department', or 'Main Category'",
            "Represents the highest level of categorization",
            "Examples include 'Audio', 'Video', 'Control', etc."
        ],
        common_names=["category_group", "product_group", "department", "main_category", "group", "category_type"],
        example_values=["Audio", "Video", "Control", "Infrastructure"]
    ),
    
    "Category": FieldDefinition(
        name="Category",
        description="Specific product category",
        required=False,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Category', 'Sub-Category', or 'Product Type'",
            "More specific than Category Group",
            "Examples include 'Projectors', 'Speakers', 'Microphones', etc."
        ],
        common_names=["category", "sub_category", "product_type", "subcategory", "product_category"],
        example_values=["Projectors", "Speakers", "Mixers", "Amplifiers"]
    ),
    
    "Manufacturer": FieldDefinition(
        name="Manufacturer",
        description="Product manufacturer/brand name",
        required=True,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Manufacturer', 'Brand', 'Vendor', or 'Make'",
            "Company that produces the product",
            "Examples include 'Sony', 'Bose', 'Extron', etc."
        ],
        common_names=["manufacturer", "brand", "vendor", "make", "supplier", "company"],
        example_values=["Sony", "Bose", "Extron", "Crestron", "Epson"]
    ),
    
    "Manufacturer_SKU": FieldDefinition(
        name="Manufacturer_SKU",
        description="Manufacturer's own product code",
        required=False,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Manufacturer SKU', 'Manufacturer Part Number', 'MPN', or 'OEM Part Number'",
            "The product code used by the manufacturer",
            "May differ from your internal SKU"
        ],
        patterns=[
            r'^[A-Z]{2,4}-[A-Z0-9]{2,}',
            r'^[A-Z]{1,3}\d{4,}',
            r'^[A-Z]\d{2}[A-Z]\d{3,}'
        ],
        common_names=["manufacturer_sku", "mpn", "mfr_part_number", "oem_part_number", "manufacturer_part_number", "mfr_sku"],
        example_values=["VPL-VW325ES", "841667", "23-056-03"]
    ),
    
    "Image_URL": FieldDefinition(
        name="Image_URL",
        description="URL to product image",
        required=False,
        data_type="url",
        mapping_hints=[
            "Often labeled as 'Image URL', 'Image', 'Product Image', or 'Photo URL'",
            "Contains a full URL to the product image",
            "Should start with 'http://' or 'https://'"
        ],
        patterns=[
            r'^https?://.+\.(jpg|jpeg|png|gif|webp)$'
        ],
        common_names=["image_url", "image", "product_image", "photo_url", "img", "image_link"],
        example_values=["https://example.com/images/product123.jpg"]
    ),
    
    "Document_Name": FieldDefinition(
        name="Document_Name",
        description="Name of associated document (manual, spec sheet, etc.)",
        required=False,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Document Name', 'Manual Name', or 'File Name'",
            "The name of the product documentation file",
            "Related to Document URL field"
        ],
        common_names=["document_name", "manual_name", "file_name", "doc_name", "document"],
        example_values=["User Manual", "Specification Sheet", "Quick Start Guide"]
    ),
    
    "Document_URL": FieldDefinition(
        name="Document_URL",
        description="URL to product documentation",
        required=False,
        data_type="url",
        mapping_hints=[
            "Often labeled as 'Document URL', 'Manual URL', 'Specs URL', or 'Documentation Link'",
            "Contains a full URL to product documentation",
            "Should start with 'http://' or 'https://'"
        ],
        patterns=[
            r'^https?://.+\.(pdf|doc|docx|txt)$'
        ],
        common_names=["document_url", "manual_url", "specs_url", "documentation_link", "doc_url", "pdf_url"],
        example_values=["https://example.com/manuals/product123.pdf"]
    ),
    
    "Unit_Of_Measure": FieldDefinition(
        name="Unit_Of_Measure",
        description="How product is sold (each, pair, box, etc.)",
        required=True,
        data_type="string",
        mapping_hints=[
            "Often labeled as 'Unit of Measure', 'UOM', 'Unit', or 'Selling Unit'",
            "Describes how the product is sold (e.g., 'EACH', 'PAIR', 'BOX')",
            "Usually short uppercase text"
        ],
        common_names=["unit_of_measure", "uom", "unit", "selling_unit", "measure_unit", "unit_type"],
        example_values=["EACH", "PAIR", "BOX", "SET", "CASE"]
    ),
    
    "Buy_Cost": FieldDefinition(
        name="Buy_Cost",
        description="Your cost to purchase from manufacturer/distributor",
        required=False,
        data_type="decimal",
        mapping_hints=[
            "Often labeled as 'Buy Cost', 'Cost', 'Dealer Cost', or 'Purchase Price'",
            "The price you pay to acquire the product",
            "May include currency symbols"
        ],
        common_names=["buy_cost", "cost", "dealer_cost", "purchase_price", "buy_price", "wholesale_price"],
        example_values=["123.45", "$199.99", "1,299.00"]
    ),
    
    "Trade_Price": FieldDefinition(
        name="Trade_Price",
        description="Price sold to trade customers",
        required=False,
        data_type="decimal",
        mapping_hints=[
            "Often labeled as 'Trade Price', 'Dealer Price', 'Wholesale Price', or 'B2B Price'",
            "Price for business or trade customers",
            "Typically higher than Buy Cost but lower than MSRP"
        ],
        common_names=["trade_price", "dealer_price", "wholesale_price", "b2b_price", "reseller_price"],
        example_values=["199.99", "$249.99", "1,499.00"]
    ),
    
    "MSRP_GBP": FieldDefinition(
        name="MSRP_GBP",
        description="Manufacturer's suggested retail price in GBP (£)",
        required=False,
        data_type="decimal",
        mapping_hints=[
            "Often labeled as 'MSRP GBP', 'RRP (£)', 'UK Price', or 'Retail Price GBP'",
            "The recommended selling price in British Pounds",
            "May include £ symbol or 'GBP' text"
        ],
        common_names=["msrp_gbp", "rrp_gbp", "uk_price", "retail_price_gbp", "gbp_price", "pound_price"],
        example_values=["£299.99", "349.99 GBP", "1,999.00"]
    ),
    
    "MSRP_USD": FieldDefinition(
        name="MSRP_USD",
        description="Manufacturer's suggested retail price in USD ($)",
        required=False,
        data_type="decimal",
        mapping_hints=[
            "Often labeled as 'MSRP USD', 'US Price', 'Retail Price USD', or 'Dollar Price'",
            "The recommended selling price in US Dollars",
            "May include $ symbol or 'USD' text"
        ],
        common_names=["msrp_usd", "us_price", "retail_price_usd", "usd_price", "dollar_price"],
        example_values=["$399.99", "499.99 USD", "2,499.00"]
    ),
    
    "MSRP_EUR": FieldDefinition(
        name="MSRP_EUR",
        description="Manufacturer's suggested retail price in EUR (€)",
        required=False,
        data_type="decimal",
        mapping_hints=[
            "Often labeled as 'MSRP EUR', 'EU Price', 'Retail Price EUR', or 'Euro Price'",
            "The recommended selling price in Euros",
            "May include € symbol or 'EUR' text"
        ],
        common_names=["msrp_eur", "eu_price", "retail_price_eur", "eur_price", "euro_price"],
        example_values=["€349.99", "399.99 EUR", "2,299.00"]
    ),
    
    "Discontinued": FieldDefinition(
        name="Discontinued",
        description="Whether product is discontinued by manufacturer",
        required=False,
        data_type="boolean",
        mapping_hints=[
            "Often labeled as 'Discontinued', 'Active', 'Status', or 'Available'",
            "Indicates if product is no longer available from manufacturer",
            "Usually boolean (Yes/No) or status code"
        ],
        common_names=["discontinued", "active", "status", "available", "in_production", "eol", "end_of_life"],
        example_values=["Yes", "No", "TRUE", "FALSE", "1", "0", "Active", "Discontinued"]
    )
}

# List of known AV manufacturers
KNOWN_MANUFACTURERS = [
    "Sony", "Panasonic", "JVC", "Epson", "Barco", "Christie", "Optoma", "BenQ", "Vivitek",  # Projectors/Displays
    "Samsung", "LG", "Sharp", "NEC", "ViewSonic", "Philips", "Vizio", "Hisense", "TCL",  # Displays/TVs
    "Bose", "JBL", "Harman Kardon", "Klipsch", "Polk Audio", "KEF", "Sonos", "Definitive Technology",  # Speakers
    "Shure", "Sennheiser", "Audio-Technica", "Rode", "AKG", "Blue", "Neumann", "Beyerdynamic",  # Microphones
    "Crestron", "Extron", "AMX", "Control4", "Savant", "RTI", "Elan", "URC",  # Control Systems
    "Yamaha", "Denon", "Marantz", "Onkyo", "Pioneer", "Anthem", "NAD", "Rotel",  # Receivers/Amps
    "QSC", "Crown", "Lab.gruppen", "Powersoft", "Ashly", "Peavey", "Behringer",  # Pro Amplifiers
    "Blackmagic Design", "AJA", "Atomos", "Teradek", "Roland", "NewTek", "Datavideo",  # Video Production
    "Cisco", "Logitech", "Polycom", "Zoom", "Avaya", "Microsoft", "Google", "BlueJeans",  # Conferencing
    "Kramer", "Gefen", "Atlona", "HDanywhere", "Just Add Power", "WyreStorm", "Key Digital",  # Signal Distribution
    "Middle Atlantic", "Chief", "Sanus", "OmniMount", "Peerless", "Vogel's",  # Mounting
    "Neutrik", "Canare", "Belden", "West Penn", "Gepco", "Liberty", "Mogami"  # Connectivity
]

# Common manufacturer prefixes for product codes
MANUFACTURER_PREFIXES = {
    "Sony": ["VPL", "BVM", "HT", "STR", "UBP"],
    "Epson": ["EH", "EB", "TW", "HC", "LS"],
    "Panasonic": ["PT", "TH", "AW", "AG"],
    "JVC": ["DLA", "SR", "KW", "GY"],
    "Crestron": ["DM", "HD", "PRO", "TSW", "CP"],
    "Extron": ["DTP", "XTP", "SMP", "DSC", "IPL"],
    "Shure": ["SM", "BETA", "ULX", "SLX", "MX"],
    "Bose": ["AM", "SM", "FreeSpace", "EdgeMax", "PowerMatch"],
    "QSC": ["K.", "KW", "KLA", "E", "GX", "RMX", "PLD", "CX"],
    "Yamaha": ["RX", "CX", "MX", "NS", "HS", "DXR", "TF", "QL"]
}

def normalize_header(header: str) -> str:
    """
    Normalize a header name for comparison
    
    Args:
        header (str): Header name to normalize
        
    Returns:
        str: Normalized header name
    """
    # Convert to lowercase
    normalized = str(header).lower()
    
    # Replace special characters with underscores
    normalized = re.sub(r'[^a-z0-9]', '_', normalized)
    
    # Replace multiple underscores with a single one
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    # Common abbreviation expansions
    abbrev_map = {
        'desc': 'description',
        'id': 'identifier',
        'num': 'number',
        'qty': 'quantity',
        'uom': 'unit_of_measure',
        'msrp': 'retail_price',
        'gbp': 'pounds',
        'usd': 'dollars',
        'eur': 'euros',
        'mfr': 'manufacturer',
        'manuf': 'manufacturer'
    }
    
    # Apply abbreviation expansions
    for abbrev, expansion in abbrev_map.items():
        if normalized == abbrev or normalized.startswith(f"{abbrev}_") or normalized.endswith(f"_{abbrev}"):
            normalized = normalized.replace(abbrev, expansion)
    
    return normalized

def get_field_mapping(header: str) -> Optional[str]:
    """
    Get standard field mapping for a header
    
    Args:
        header (str): Header name
        
    Returns:
        Optional[str]: Mapped standard field name or None
    """
    normalized = normalize_header(header)
    
    # Direct match with common names
    for field_name, field_def in FIELD_DEFINITIONS.items():
        if normalized in field_def.common_names:
            return field_name
            
    # Check for fuzzy matches with common names
    for field_name, field_def in FIELD_DEFINITIONS.items():
        for common_name in field_def.common_names:
            if common_name in normalized or normalized in common_name:
                return field_name
                
    return None

# Common field name to standard field mapping
COMMON_FIELD_MAPPINGS = {
    # SKU/ID fields
    'sku': 'SKU',
    'item_number': 'SKU',
    'item_id': 'SKU',
    'product_id': 'SKU',
    'product_code': 'SKU',
    'product_number': 'SKU',
    'item_code': 'SKU',
    'part_number': 'SKU',
    'id': 'SKU',
    
    # Description fields
    'name': 'Short_Description',
    'product_name': 'Short_Description',
    'title': 'Short_Description',
    'short_description': 'Short_Description',
    'item_name': 'Short_Description',
    'heading': 'Short_Description',
    'product_title': 'Short_Description',
    
    'description': 'Long_Description',
    'long_description': 'Long_Description',
    'full_description': 'Long_Description',
    'detailed_description': 'Long_Description',
    'details': 'Long_Description',
    'product_details': 'Long_Description',
    'product_description': 'Long_Description',
    
    # Model fields
    'model': 'Model',
    'model_number': 'Model',
    'model_name': 'Model',
    'model_no': 'Model',
    
    # Category fields
    'category': 'Category',
    'product_category': 'Category',
    'product_type': 'Category',
    'subcategory': 'Category',
    'sub_category': 'Category',
    
    'category_group': 'Category_Group',
    'main_category': 'Category_Group',
    'product_group': 'Category_Group',
    'department': 'Category_Group',
    'family': 'Category_Group',
    
    # Manufacturer fields
    'manufacturer': 'Manufacturer',
    'brand': 'Manufacturer',
    'vendor': 'Manufacturer',
    'make': 'Manufacturer',
    'supplier': 'Manufacturer',
    'mfr': 'Manufacturer',
    
    'manufacturer_sku': 'Manufacturer_SKU',
    'manufacturer_part_number': 'Manufacturer_SKU',
    'mfr_part_number': 'Manufacturer_SKU',
    'mpn': 'Manufacturer_SKU',
    'vendor_part_number': 'Manufacturer_SKU',
    'oem_part_number': 'Manufacturer_SKU',
    
    # Media fields
    'image_url': 'Image_URL',
    'image': 'Image_URL',
    'photo_url': 'Image_URL',
    'product_image': 'Image_URL',
    'img_url': 'Image_URL',
    'pic_url': 'Image_URL',
    
    'document_name': 'Document_Name',
    'doc_name': 'Document_Name',
    'manual_name': 'Document_Name',
    'document_title': 'Document_Name',
    'file_name': 'Document_Name',
    
    'document_url': 'Document_URL',
    'doc_url': 'Document_URL',
    'manual_url': 'Document_URL',
    'pdf_url': 'Document_URL',
    'documentation_url': 'Document_URL',
    'spec_sheet_url': 'Document_URL',
    
    # Unit of Measure fields
    'unit_of_measure': 'Unit_Of_Measure',
    'uom': 'Unit_Of_Measure',
    'unit': 'Unit_Of_Measure',
    'selling_unit': 'Unit_Of_Measure',
    'measure_unit': 'Unit_Of_Measure',
    'unit_type': 'Unit_Of_Measure',
    'quantity_unit': 'Unit_Of_Measure',
    
    # Price fields
    'cost': 'Buy_Cost',
    'buy_cost': 'Buy_Cost',
    'purchase_price': 'Buy_Cost',
    'dealer_cost': 'Buy_Cost',
    'wholesale_cost': 'Buy_Cost',
    'buy_price': 'Buy_Cost',
    
    'trade_price': 'Trade_Price',
    'dealer_price': 'Trade_Price',
    'wholesale_price': 'Trade_Price',
    'b2b_price': 'Trade_Price',
    'reseller_price': 'Trade_Price',
    'trade_cost': 'Trade_Price',
    
    'msrp_gbp': 'MSRP_GBP',
    'rrp_gbp': 'MSRP_GBP',
    'uk_price': 'MSRP_GBP',
    'gbp_price': 'MSRP_GBP',
    'retail_price_gbp': 'MSRP_GBP',
    'price_gbp': 'MSRP_GBP',
    
    'msrp_usd': 'MSRP_USD',
    'us_price': 'MSRP_USD',
    'usd_price': 'MSRP_USD',
    'retail_price_usd': 'MSRP_USD',
    'price_usd': 'MSRP_USD',
    'dollar_price': 'MSRP_USD',
    
    'msrp_eur': 'MSRP_EUR',
    'eu_price': 'MSRP_EUR',
    'eur_price': 'MSRP_EUR',
    'retail_price_eur': 'MSRP_EUR',
    'price_eur': 'MSRP_EUR',
    'euro_price': 'MSRP_EUR',
    
    # Status fields
    'discontinued': 'Discontinued',
    'active': 'Discontinued',
    'status': 'Discontinued',
    'available': 'Discontinued',
    'in_production': 'Discontinued',
    'eol': 'Discontinued',
    'end_of_life': 'Discontinued'
}

# Common content patterns by field
CONTENT_PATTERNS = {
    'SKU': [
        r'^[A-Z]{2,5}-\d{3,6}$',  # Format like AV-12345
        r'^[A-Z0-9]{3,6}-[A-Z0-9]{2,6}$',  # Format like XDR-55A
        r'^\d{3,8}$'  # Simple numeric IDs
    ],
    'Short_Description': [
        r'^.{5,100}$'  # 5-100 character text
    ],
    'Long_Description': [
        r'^.{100,}$'  # >100 character text
    ],
    'Model': [
        r'^[A-Z0-9]{2,}[\-]?\d{2,}$',  # Format like XDR55 or HC-3800
        r'^[A-Z]{1,3}[\-]?\d{3,}$'  # Format like VP-450
    ],
    'Image_URL': [
        r'^https?://.+\.(jpg|jpeg|png|gif|webp)$'  # Image URLs
    ],
    'Document_URL': [
        r'^https?://.+\.(pdf|doc|docx|txt)$'  # Document URLs
    ],
    'Buy_Cost': [
        r'^\$?\d+\.?\d*$',  # Simple dollar amounts
        r'^\d{1,3}(?:,\d{3})*\.\d{2}$'  # Formatted dollars
    ],
    'MSRP_GBP': [
        r'^£?\d+\.?\d*$',  # Pound amounts
        r'^\d+\.?\d*\s?GBP$'  # GBP notation
    ],
    'MSRP_USD': [
        r'^\$?\d+\.?\d*$',  # Dollar amounts
        r'^\d+\.?\d*\s?USD$'  # USD notation
    ],
    'MSRP_EUR': [
        r'^€?\d+\.?\d*$',  # Euro amounts
        r'^\d+\.?\d*\s?EUR$'  # EUR notation
    ],
    'Discontinued': [
        r'^(?:yes|no|true|false|1|0)$',  # Boolean values
        r'^(?:discontinued|active|available|obsolete)$'  # Status values
    ]
}

def update_field_definition(field_name: str, **kwargs) -> None:
    """
    Update a field definition with new attributes
    
    Args:
        field_name (str): Name of field to update
        **kwargs: Attributes to update
    """
    if field_name not in FIELD_DEFINITIONS:
        logger.warning(f"Field {field_name} does not exist, cannot update")
        return
        
    field_def = FIELD_DEFINITIONS[field_name]
    
    for key, value in kwargs.items():
        if hasattr(field_def, key):
            setattr(field_def, key, value)
            
    logger.debug(f"Updated field definition for {field_name}")

def add_manufacturer(name: str, prefixes: List[str] = None) -> None:
    """
    Add a new manufacturer to known manufacturers
    
    Args:
        name (str): Manufacturer name
        prefixes (List[str], optional): Product code prefixes
    """
    if name not in KNOWN_MANUFACTURERS:
        KNOWN_MANUFACTURERS.append(name)
        logger.debug(f"Added manufacturer: {name}")
        
    if prefixes and name not in MANUFACTURER_PREFIXES:
        MANUFACTURER_PREFIXES[name]