"""Runs without a model or API key. Shows interrupt() and Command(resume=...)."""
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    draft: str
    approved: bool


def write_draft(state: State) -> dict:
    return {"draft": "Hello! Your refund has been approved."}


def ask_human(state: State) -> dict:
    decision = interrupt({"question": "Send this reply?", "draft": state["draft"]})
    return {"approved": decision == "yes"}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("ask_human", ask_human)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "ask_human")
builder.add_edge("ask_human", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}

first = graph.invoke({"draft": "", "approved": False}, config=config)
print(first["__interrupt__"])

final = graph.invoke(Command(resume="yes"), config=config)
print(final["approved"])
