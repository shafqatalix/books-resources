from langchain.agents import create_agent
from langchain.tools import tool

from lib.config import MODEL, as_text
from lib.index import search_docs
from lib.tools import save_note

researcher = create_agent(
    model=MODEL,
    tools=[search_docs],
    system_prompt=(
        "You research questions in the company documents. Search as needed, "
        "then reply with a short summary and the source files you used."
    ),
)


@tool
def research(question: str) -> str:
    """Research a question in the company documents.

    Returns a short summary with sources. Use it for any factual question.
    """
    result = researcher.invoke({"messages": [{"role": "user", "content": question}]})
    return as_text(result["messages"][-1])


manager = create_agent(
    model=MODEL,
    tools=[research, save_note],
    system_prompt=(
        "You are a helpful assistant. Use the research tool for facts. "
        "Use save_note only when the user asks to keep a note."
    ),
)

if __name__ == "__main__":
    question = "What is the refund window, and what should I do about a phishing email?"
    result = manager.invoke({"messages": [{"role": "user", "content": question}]})
    print(as_text(result["messages"][-1]))
