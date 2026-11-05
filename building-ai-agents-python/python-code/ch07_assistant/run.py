from langchain.agents import create_agent

from lib.config import MODEL
from lib.tools import list_documents, read_document

SYSTEM_PROMPT = """You are the Acme Cloud documents assistant.

Answer questions using only the company documents.
- If you do not know which files exist, call list_documents first.
- Call read_document to read a file before answering from it.
- Name the file or files you used at the end of your answer.
- If the documents do not contain the answer, say you could not find it.
  Do not guess."""

agent = create_agent(
    model=MODEL,
    tools=[list_documents, read_document],
    system_prompt=SYSTEM_PROMPT,
)

question = "How long do customers have to request a full refund?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
    config={"recursion_limit": 12},
):
    step["messages"][-1].pretty_print()
