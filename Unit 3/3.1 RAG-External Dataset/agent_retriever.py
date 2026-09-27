# Retrieval 
# builds a keyword-based search engine (retriever) over a dataset of gala invitees so LangGraph agent can query guest information at runtime.

import datasets
from httpx2 import query
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_community.retrievers import BM25Retriever

guest_list= datasets.load_dataset("agents-course/unit3-invitees", split="train")

docs= [
    Document(
        page_content="\n".join([
            f"Name:  {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest['name']}
    )
    for guest in guest_list
]


bm25= BM25Retriever.from_documents(docs)

def extract_text(query: str)-> str:
    """Retrieves detailed info about the gala"""
    results = bm25.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No guests found."