from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from lib.config import MODEL, as_text
from lib.index import search_docs

SYSTEM_PROMPT = (
    "You are the Acme Cloud documents assistant. Answer only from passages "
    "returned by search_docs, cite the source file, and say so if you cannot find an answer."
)

tools = [search_docs]
model = init_chat_model(MODEL).bind_tools(tools)


def call_model(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    reply = model.invoke(messages)
    return {"messages": [reply]}


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

result = graph.invoke(
    {"messages": [{"role": "user", "content": "How long do I have to ask for a refund?"}]}
)
print(as_text(result["messages"][-1]))
print()
print(graph.get_graph().draw_mermaid())
