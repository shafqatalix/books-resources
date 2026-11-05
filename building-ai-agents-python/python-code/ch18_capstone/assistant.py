from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lib.config import MODEL, as_text
from lib.index import search_docs
from lib.tools import list_documents, save_note

SYSTEM_PROMPT = """You are the Acme Cloud documents assistant.

Answer questions using only passages returned by search_docs.
- Search before answering. Search again with different words if needed.
- Do not add facts that are not in the passages.
- End each answer with a line such as: Source: refund_policy.txt
- If the passages do not answer the question, say you could not find it.
- Use list_documents when asked what documents exist.
- Use save_note only when the user explicitly asks you to keep a note."""


def build_agent():
    return create_agent(
        model=MODEL,
        tools=[search_docs, list_documents, save_note],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "save_note": {"allowed_decisions": ["approve", "reject"]},
                    "search_docs": False,
                    "list_documents": False,
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def chat() -> None:
    agent = build_agent()
    config = {
        "configurable": {"thread_id": "cli-session"},
        "recursion_limit": 15,
        "tags": ["capstone"],
    }
    print("Acme documents assistant. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}, config=config
        )

        while "__interrupt__" in result:
            print("\nApproval needed for:", result["__interrupt__"])
            choice = input("approve or reject? ").strip().lower()
            decision = {"type": "approve"} if choice.startswith("a") else {"type": "reject"}
            result = agent.invoke(Command(resume={"decisions": [decision]}), config=config)

        print("\nAssistant:", as_text(result["messages"][-1]))


if __name__ == "__main__":
    chat()
