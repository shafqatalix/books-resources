from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from lib.config import MODEL, as_text

model = init_chat_model(MODEL, temperature=0)


class Route(BaseModel):
    """Where to send a customer message."""

    destination: Literal["billing", "technical", "general"] = Field(
        description="billing for payments and refunds, technical for bugs and "
        "errors, general for anything else"
    )


class State(TypedDict):
    question: str
    destination: str
    answer: str


classifier = model.with_structured_output(Route)


def classify(state: State) -> dict:
    route = classifier.invoke(state["question"])
    return {"destination": route.destination}


def pick_handler(state: State) -> Literal["billing", "technical", "general"]:
    return state["destination"]


def make_handler(instructions: str):
    def handler(state: State) -> dict:
        reply = model.invoke(
            [SystemMessage(content=instructions), HumanMessage(content=state["question"])]
        )
        return {"answer": as_text(reply)}

    return handler


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing", make_handler("You are a billing specialist. Be brief and practical."))
builder.add_node("technical", make_handler("You are a technical support engineer. Give clear steps."))
builder.add_node("general", make_handler("You are a friendly general support agent. Be brief."))

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", pick_handler)
for name in ["billing", "technical", "general"]:
    builder.add_edge(name, END)

graph = builder.compile()

if __name__ == "__main__":
    for q in [
        "I was charged twice this month.",
        "The export button gives me a 500 error.",
        "Do you have an office in Berlin?",
    ]:
        result = graph.invoke({"question": q})
        print(q, "->", result["destination"])
        print(result["answer"], "\n")
