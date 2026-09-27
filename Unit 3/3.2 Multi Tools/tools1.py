from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
import random

search_tool= DuckDuckGoSearchRun()
results= search_tool.invoke("What is the capital of Nepal?")
print(results)

def get_weather(location: str)-> str:
    """Fetched dummy weather information for a given location."""
    weather_conditions=[
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Sunny", "temp_c": 30},
        {"condition": "Windy", "temp_c": 20},
        {"condition": "Snowy", "temp_c": -5},
    ]
    data= random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}"


weather_info_tool= Tool(
    name= "get_weather_info",
    func=get_weather,
    description="Fetches dummy weather information for a given location."
)