"""Runs without a model or API key. Shows how a reducer combines updates."""
import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    notes: Annotated[list[str], operator.add]  # appends
    last: str  # replaces


def step_a(state: State) -> dict:
    return {"notes": ["A ran"], "last": "a"}


def step_b(state: State) -> dict:
    return {"notes": ["B ran"], "last": "b"}


builder = StateGraph(State)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", END)
graph = builder.compile()

print(graph.invoke({"notes": [], "last": ""}))
# notes holds both entries (appended); last holds only "b" (replaced).
