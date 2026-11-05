from langchain.agents import create_agent

from lib.config import MODEL, as_text
from lib.index import search_docs

SYSTEM_PROMPT = """You are the Acme Cloud documents assistant.

Answer questions using only passages returned by search_docs.
- Search before answering. Search again with different words if the first
  results do not help.
- Quote or paraphrase the passages. Do not add facts that are not in them.
- End with a line like: Source: refund_policy.txt
- If the passages do not answer the question, say you could not find it."""

agent = create_agent(
    model=MODEL,
    tools=[search_docs],
    system_prompt=SYSTEM_PROMPT,
)

questions = [
    "How long do I have to ask for my money back?",
    "What should I do about a phishing email?",
    "What is the parental leave policy?",
]

for q in questions:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config={"recursion_limit": 12},
    )
    print("Q:", q)
    print("A:", as_text(result["messages"][-1]))
    print()
