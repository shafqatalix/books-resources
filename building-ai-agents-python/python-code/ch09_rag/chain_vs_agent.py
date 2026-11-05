from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

from lib.config import MODEL, as_text
from lib.index import search_docs

model = init_chat_model(MODEL)


def answer_with_context(question: str) -> str:
    passages = search_docs.invoke({"query": question})
    messages = [
        SystemMessage(content="Answer only from the passages. Say so if they do not help."),
        HumanMessage(content=f"Question: {question}\n\nPassages:\n{passages}"),
    ]
    return as_text(model.invoke(messages))


if __name__ == "__main__":
    for q in [
        "How long do I have to ask for my money back?",
        "What is the parental leave policy?",
    ]:
        print("Q:", q)
        print("A:", answer_with_context(q), "\n")
