"""The graph exported for `langgraph dev` (see langgraph.json at the repo root).

No checkpointer here: when a platform serves the graph it supplies persistence.
"""
from langchain.agents import create_agent

from lib.config import MODEL
from lib.index import search_docs

agent = create_agent(
    model=MODEL,
    tools=[search_docs],
    system_prompt="You are the Acme Cloud documents assistant. Cite your sources.",
)
