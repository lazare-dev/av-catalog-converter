"""
Simple test script to check if the LLM factory is working
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the LLM factory
from core.llm.llm_factory import LLMFactory

def main():
    """Test the LLM factory"""
    print("Testing LLM factory...")
    
    # Get the available models
    models = LLMFactory.get_available_models()
    print(f"Available models: {models}")
    
    # Create a client
    print("Creating LLM client...")
    client = LLMFactory.create_client()
    
    # Get model info
    model_info = client.get_model_info()
    print(f"Model info: {model_info}")
    
    # Generate a response
    prompt = "Extract the product category from this description: 'Sony 65-inch 4K OLED TV with HDR'"
    print(f"Generating response for prompt: {prompt}")
    response = client.generate_response(prompt)
    print(f"Response: {response}")
    
    # Get stats
    stats = LLMFactory.get_stats()
    print(f"LLM factory stats: {stats}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
