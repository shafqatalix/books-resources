from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lib.config import MODEL, as_text
from lib.index import search_docs
from lib.tools import save_note

SYSTEM_PROMPT = (
    "You are the Acme Cloud documents assistant. Answer only from passages "
    "returned by search_docs and cite the source file. Use save_note only when "
    "the user asks you to keep a note."
)

agent = create_agent(
    model=MODEL,
    tools=[search_docs, save_note],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_note": {"allowed_decisions": ["approve", "reject"]},
                "search_docs": False,
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "approval-demo"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Find the refund rules and save a short note called refund-summary."}]},
    config=config,
)

if "__interrupt__" in result:
    print("PAUSED. The agent wants approval for:")
    print(result["__interrupt__"])
    answer = input("approve or reject? ").strip().lower()
    decision = {"type": "approve"} if answer.startswith("a") else {"type": "reject"}
    result = agent.invoke(Command(resume={"decisions": [decision]}), config=config)

print(as_text(result["messages"][-1]))
