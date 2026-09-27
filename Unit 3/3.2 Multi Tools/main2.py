import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools1 import search_tool, weather_info_tool

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    temperature=0.1,
)

chat = ChatHuggingFace(llm=llm)
tools = [search_tool, weather_info_tool]
chat_with_tools = chat.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {"messages": [chat_with_tools.invoke(state["messages"])]}

# Build LangGraph workflow
builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

alfred = builder.compile()

# Run query
messages = [HumanMessage(content="Who is Meta/Facebook and what is their weather in San Francisco right now?")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)