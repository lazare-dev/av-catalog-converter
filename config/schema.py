# config/schema.py
"""Output schema definition for AV Catalog Standardizer"""

from typing import List, Dict, Any, Optional, Pattern, Union
from dataclasses import dataclass
import re

@dataclass
class FieldDefinition:
    name: str
    position: int
    required: bool
    description: str
    examples: List[str]
    patterns: List[Union[str, Pattern]]
    mapping_hints: List[str]
    default_value: Any = None

# Updated schema with correct field order
OUTPUT_SCHEMA = [
    FieldDefinition(
        name="SKU",
        position=1,
        required=True,
        description="Unique product identifier used by the reseller",
        examples=["AV-PRJ-1001", "SPK-WL-500", "CAB-HDMI-10M"],
        patterns=[r"[A-Z]{2,3}-[A-Z]{2,3}-\d{3,4}", r"[A-Z0-9]{6,10}"],
        mapping_hints=["sku", "item_number", "product_code", "part_number", "item_code", "article_number"],
        default_value=None
    ),
    FieldDefinition(
        name="Short Description",
        position=2,
        required=True,
        description="Brief product description (50-100 characters)",
        examples=["4K Laser Projector - 5000 lumens", "Wireless Ceiling Speaker - White"],
        patterns=[],
        mapping_hints=["short_desc", "product_name", "title", "item_name", "name", "description", "product"],
        default_value=None
    ),
    FieldDefinition(
        name="Long Description",
        position=3,
        required=False,
        description="Detailed product description",
        examples=["Professional 4K laser projector with 5000 lumens brightness, HDR support, and 20,000 hour lamp life"],
        patterns=[],
        mapping_hints=["long_desc", "full_description", "detailed_description", "product_description", "description", "details"],
        default_value=""
    ),
    FieldDefinition(
        name="Model",
        position=4,
        required=False,
        description="Manufacturer's model number/name",
        examples=["XG-500", "LS10500", "HDMI-4K-10M"],
        patterns=[],
        mapping_hints=["model", "model_number", "model_name", "model_no", "type", "product_model"],
        default_value=""
    ),
    FieldDefinition(
        name="Category Group",
        position=5,
        required=False,
        description="Top-level product category",
        examples=["Display", "Audio", "Cables", "Control Systems"],
        patterns=[],
        mapping_hints=["category_group", "main_category", "product_group", "group", "department"],
        default_value=""
    ),
    FieldDefinition(
        name="Category",
        position=6,
        required=False,
        description="Specific product category",
        examples=["Projectors", "Speakers", "HDMI Cables", "Control Panels"],
        patterns=[],
        mapping_hints=["category", "sub_category", "product_category", "product_type", "type"],
        default_value=""
    ),
    FieldDefinition(
        name="Manufacturer",
        position=7,
        required=True,
        description="Product manufacturer/brand name",
        examples=["Sony", "Bose", "Crestron", "Extron"],
        patterns=[],
        mapping_hints=["manufacturer", "brand", "vendor", "supplier", "make", "producer", "company"],
        default_value=None
    ),
    FieldDefinition(
        name="Manufacturer SKU",
        position=8,
        required=False,
        description="Manufacturer's own product code",
        examples=["VPL-VW915ES", "SB-1000-BLK", "DM-MD16X16"],
        patterns=[],
        mapping_hints=["manufacturer_sku", "mfr_part_number", "oem_sku", "vendor_sku", "original_sku", "mfg_code"],
        default_value=""
    ),
    FieldDefinition(
        name="Image URL",
        position=9,
        required=False,
        description="URL to product image",
        examples=["https://example.com/images/product1.jpg"],
        patterns=[r"https?://.*\.(jpg|jpeg|png|gif|webp)"],
        mapping_hints=["image_url", "image", "picture", "photo", "img_url", "product_image"],
        default_value=""
    ),
    FieldDefinition(
        name="Document Name",
        position=10,
        required=False,
        description="Name of product documentation file",
        examples=["User Manual", "Specification Sheet", "Installation Guide"],
        patterns=[],
        mapping_hints=["document_name", "doc_name", "manual", "guide", "documentation", "spec_sheet"],
        default_value=""
    ),
    FieldDefinition(
        name="Document URL",
        position=11,
        required=False,
        description="URL to product documentation",
        examples=["https://example.com/docs/manual.pdf"],
        patterns=[r"https?://.*\.(pdf|doc|docx|txt)"],
        mapping_hints=["document_url", "doc_url", "manual_url", "documentation_link", "spec_url"],
        default_value=""
    ),
    FieldDefinition(
        name="Unit Of Measure",
        position=12,
        required=False,
        description="Unit of measurement for the product",
        examples=["Each", "Pair", "Box of 10", "Meter"],
        patterns=[],
        mapping_hints=["unit_of_measure", "uom", "unit", "measure", "quantity_unit", "sales_unit"],
        default_value="Each"
    ),
    FieldDefinition(
        name="Buy Cost",
        position=13,
        required=False,
        description="Wholesale cost price (numeric only)",
        examples=["1000.00", "249.99", "5.75"],
        patterns=[r"\d+\.?\d*"],
        mapping_hints=["buy_cost", "cost", "wholesale_price", "dealer_price", "cost_price", "net_price"],
        default_value="0.00"
    ),
    FieldDefinition(
        name="Trade Price",
        position=14,
        required=True,
        description="Standard trade/dealer price (numeric only)",
        examples=["1200.00", "299.99", "6.50"],
        patterns=[r"\d+\.?\d*"],
        mapping_hints=["trade_price", "dealer_price", "wholesale", "price", "net_price", "distributor_price"],
        default_value=None
    ),
    FieldDefinition(
        name="MSRP GBP",
        position=15,
        required=False,
        description="Manufacturer's suggested retail price in GBP (numeric only)",
        examples=["1499.99", "349.99", "7.99"],
        patterns=[r"\d+\.?\d*"],
        mapping_hints=["msrp_gbp", "rrp_gbp", "retail_price_gbp", "uk_price", "gbp_price", "pound_price"],
        default_value="0.00"
    ),
    FieldDefinition(
        name="MSRP USD",
        position=16,
        required=False,
        description="Manufacturer's suggested retail price in USD (numeric only)",
        examples=["1999.99", "399.99", "9.99"],
        patterns=[r"\d+\.?\d*"],
        mapping_hints=["msrp_usd", "rrp_usd", "retail_price_usd", "us_price", "usd_price", "dollar_price"],
        default_value="0.00"
    ),
    FieldDefinition(
        name="MSRP EUR",
        position=17,
        required=False,
        description="Manufacturer's suggested retail price in EUR (numeric only)",
        examples=["1799.99", "379.99", "8.99"],
        patterns=[r"\d+\.?\d*"],
        mapping_hints=["msrp_eur", "rrp_eur", "retail_price_eur", "eu_price", "eur_price", "euro_price"],
        default_value="0.00"
    ),
    FieldDefinition(
        name="Discontinued",
        position=18,
        required=False,
        description="Whether the product is discontinued (Yes/No)",
        examples=["Yes", "No"],
        patterns=[r"(Yes|No|Y|N|True|False|1|0)"],
        mapping_hints=["discontinued", "obsolete", "eol", "end_of_life", "active", "status"],
        default_value="No"
    ),
]

# Create a dictionary for easier lookup
SCHEMA_DICT = {field.name: field for field in OUTPUT_SCHEMA}

# Define the order of fields in output
FIELD_ORDER = [field.name for field in sorted(OUTPUT_SCHEMA, key=lambda x: x.position)]

# Define required fields
REQUIRED_FIELDS = [field.name for field in OUTPUT_SCHEMA if field.required]

# Define numeric fields
NUMERIC_FIELDS = ["Buy Cost", "Trade Price", "MSRP GBP", "MSRP USD", "MSRP EUR"]
