# Tools for the agent

from langchain_core.tools import Tool
from retriever import extract_text


guest_info_tool = Tool(
    name="guest_info_retriever",
    description="Retrieves detailed information about the gala guests. Input should be a guest's name.",
    func=extract_text,
)