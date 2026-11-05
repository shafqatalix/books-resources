from langchain.agents import create_agent

from lib.config import MODEL, as_text
from lib.index import search_docs

agent = create_agent(
    model=MODEL,
    tools=[search_docs],
    system_prompt=(
        "You are the Acme Cloud documents assistant. Answer only from passages "
        "returned by search_docs, cite the source file, and say so if you cannot find an answer."
    ),
)

question = "Can I get a refund on a monthly plan after 45 days?"

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={
        "tags": ["chunk-200", "refund-tests"],
        "metadata": {"experiment": "chunk-size", "chunk_size": 200},
        "run_name": "refund-question",
        "recursion_limit": 12,
    },
)

print(as_text(result["messages"][-1]))
