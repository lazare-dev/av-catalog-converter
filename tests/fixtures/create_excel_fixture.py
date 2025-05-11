"""
Script to create Excel test fixtures
"""
import pandas as pd
import os
from pathlib import Path

# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent

def create_small_excel_fixture():
    """Create a small Excel fixture for testing"""
    # Read the CSV fixture
    csv_path = FIXTURES_DIR / "small_catalog.csv"
    df = pd.read_csv(csv_path)
    
    # Save as Excel
    excel_path = FIXTURES_DIR / "small_catalog.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Created small Excel fixture: {excel_path}")

def create_medium_excel_fixture():
    """Create a medium-sized Excel fixture for testing"""
    # Create a DataFrame with 1,000 rows
    import numpy as np
    
    # Create base data
    categories = ['Audio', 'Video', 'Lighting', 'Accessories', 'Computers', 'Networking']
    manufacturers = ['Sony', 'Panasonic', 'JBL', 'Shure', 'Yamaha', 'Aputure', 'Blackmagic', 'Apple', 'Dell', 'HP']
    product_types = ['Camera', 'Microphone', 'Speaker', 'Mixer', 'Light', 'Cable', 'Adapter', 'Computer', 'Monitor', 'Switch']
    
    # Generate data
    rows = 1000
    data = {
        'SKU': [f"SKU{i:05d}" for i in range(rows)],
        'Product Name': [f"{np.random.choice(manufacturers)} {np.random.choice(product_types)} {i}" for i in range(rows)],
        'Price': np.round(np.random.uniform(10, 5000, size=rows), 2),
        'Category': np.random.choice(categories, size=rows),
        'Manufacturer': np.random.choice(manufacturers, size=rows),
        'Description': [f"Professional {np.random.choice(product_types).lower()} for {np.random.choice(['studio', 'field', 'home', 'office', 'broadcast'])} use" for _ in range(rows)],
        'Stock': np.random.randint(0, 100, size=rows),
        'Weight': np.round(np.random.uniform(0.1, 10, size=rows), 1),
        'Dimensions': [f"{np.random.randint(1, 50)}x{np.random.randint(1, 30)}x{np.random.randint(1, 20)}" for _ in range(rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Save as Excel
    excel_path = FIXTURES_DIR / "medium_catalog.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Created medium Excel fixture: {excel_path}")
    
    # Also save as CSV for CSV tests
    csv_path = FIXTURES_DIR / "medium_catalog.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created medium CSV fixture: {csv_path}")

if __name__ == "__main__":
    create_small_excel_fixture()
    create_medium_excel_fixture()
