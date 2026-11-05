from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from lib.config import MODEL, as_text
from lib.index import search_docs

SYSTEM_PROMPT = (
    "You are the Acme Cloud documents assistant. Answer only from passages "
    "returned by search_docs, cite the source file, and say so if you cannot find an answer."
)

agent = create_agent(
    model=MODEL,
    tools=[search_docs],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


def ask(question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    return as_text(result["messages"][-1])


print(ask("How long do customers have to request a full refund?", "alice"))
print(ask("And for monthly plans?", "alice"))
print(ask("And for monthly plans?", "bob"))
