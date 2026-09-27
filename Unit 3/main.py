import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import guest_info_tool, weather_info_tool, search_tool

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool]
chat_with_tools = chat.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])]
    }

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

alfred = builder.compile()

if __name__ == "__main__":

    # Memory for the agent
    response = alfred.invoke({"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]})
    print("🎩 Alfred's Response:")
    print(response['messages'][-1].content)
    print()

    # Second interaction (referencing the first)
    response = alfred.invoke({"messages": response["messages"] + [HumanMessage(content="What projects is she currently working on?")]})
    print("🎩 Alfred's Response:")
    print(response['messages'][-1].content)

    #Weather tool
    response = alfred.invoke({"messages": "What's the weather like in Kathmandu tonight? Will it be suitable for our fireworks display?"})
    print("🎩 Alfred's Response:")
    print(response['messages'][-1].content)

    # Combining tools
    response = alfred.invoke({"messages":"I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"})
    print("🎩 Alfred's Response:")
    print(response['messages'][-1].content)