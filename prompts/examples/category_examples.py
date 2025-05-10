# prompts/examples/category_examples.py
"""
Category extraction examples
"""

CATEGORY_EXTRACTION_EXAMPLE_INPUT = """
[
  {
    "idx": 0,
    "data": {
      "Short Description": "4K Laser Projector - 5000 lumens",
      "Manufacturer": "Sony"
    }
  },
  {
    "idx": 1,
    "data": {
      "Short Description": "Wireless Ceiling Speaker - White",
      "Manufacturer": "Bose"
    }
  },
  {
    "idx": 2,
    "data": {
      "Short Description": "HDMI Cable - 10m",
      "Manufacturer": "Kramer"
    }
  }
]
"""

CATEGORY_EXTRACTION_EXAMPLE_OUTPUT = """
{
  "0": {
    "category_group": "Display",
    "category": "Projectors"
  },
  "1": {
    "category_group": "Audio",
    "category": "Speakers"
  },
  "2": {
    "category_group": "Infrastructure",
    "category": "Cables"
  }
}
"""