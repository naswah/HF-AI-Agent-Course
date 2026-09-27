# Tools for the agent

from langchain_core.tools import Tool
from retriever import extract_text, get_weather
from langchain_community.tools import DuckDuckGoSearchRun


guest_info_tool = Tool(
    name="guest_info_retriever",
    description="Retrieves detailed information about the gala guests. Input should be a guest's name.",
    func=extract_text,
)


search_tool= DuckDuckGoSearchRun()


weather_info_tool= Tool(
    name= "get_weather_info",
    func=get_weather,
    description="Fetches dummy weather information for a given location."
)