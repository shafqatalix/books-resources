from fastapi import FastAPI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from lib.config import MODEL, as_text
from lib.index import search_docs

app = FastAPI()

agent = create_agent(
    model=MODEL,
    tools=[search_docs],
    system_prompt="You are the Acme Cloud documents assistant. Cite your sources.",
    checkpointer=InMemorySaver(),
)


class Ask(BaseModel):
    question: str
    thread_id: str


@app.post("/ask")
def ask(body: Ask) -> dict:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": body.question}]},
        config={
            "configurable": {"thread_id": body.thread_id},
            "recursion_limit": 12,
        },
    )
    return {"answer": as_text(result["messages"][-1])}
