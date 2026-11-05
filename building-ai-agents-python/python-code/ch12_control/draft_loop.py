from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from lib.config import MODEL, as_text

model = init_chat_model(MODEL, temperature=0)


class Review(BaseModel):
    """A review of a short draft."""

    approved: bool = Field(description="True if the draft is clear, accurate, and under 60 words")
    feedback: str = Field(description="One sentence on what to improve, or 'good' if approved")


class DraftState(TypedDict):
    topic: str
    draft: str
    feedback: str
    approved: bool
    attempts: int


reviewer = model.with_structured_output(Review)


def write(state: DraftState) -> dict:
    prompt = f"Write a clear explanation of {state['topic']} in under 60 words."
    if state["feedback"]:
        prompt += (
            f"\n\nImprove the previous draft using this feedback: {state['feedback']}"
            f"\n\nPrevious draft:\n{state['draft']}"
        )
    reply = model.invoke([HumanMessage(content=prompt)])
    return {"draft": as_text(reply), "attempts": state["attempts"] + 1}


def review(state: DraftState) -> dict:
    result = reviewer.invoke(
        "Review this draft. Approve only if it is clear, accurate, and under 60 words.\n\n"
        + state["draft"]
    )
    return {"approved": result.approved, "feedback": result.feedback}


def after_review(state: DraftState) -> Literal["write", "__end__"]:
    if state["approved"] or state["attempts"] >= 3:
        return END
    return "write"


builder = StateGraph(DraftState)
builder.add_node("write", write)
builder.add_node("review", review)
builder.add_edge(START, "write")
builder.add_edge("write", "review")
builder.add_conditional_edges("review", after_review)
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke(
        {"topic": "what an API is", "draft": "", "feedback": "", "approved": False, "attempts": 0}
    )
    print("Attempts:", result["attempts"], "| approved:", result["approved"])
    print(result["draft"])
