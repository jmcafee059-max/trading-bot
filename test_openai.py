import os
from dotenv import load_dotenv
from openai_market_analyzer import OpenAIMarketAnalyzer
import pandas as pd

load_dotenv()

# Test OpenAI integration
print("Testing OpenAI Integration...")
print(f"OPENAI_ENABLED: {os.getenv('OPENAI_ENABLED')}")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:20]}..." if os.getenv('OPENAI_API_KEY') else "OPENAI_API_KEY: Not set")

analyzer = OpenAIMarketAnalyzer()
print(f"Analyzer enabled: {analyzer.enabled}")
print(f"Analyzer has client: {analyzer.client is not None}")

# Test with sample data
test_data = pd.DataFrame({
    'close': [100.0, 101.0, 102.0, 103.0, 104.0, 103.5, 103.0, 104.5, 105.0, 106.0]
})

print("\nTesting market analysis...")
result = analyzer.analyze_market_conditions(test_data, 106.0)
print(f"Analysis result: {result}")
