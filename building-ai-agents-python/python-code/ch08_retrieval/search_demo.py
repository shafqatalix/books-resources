from lib.index import build_vector_store

store = build_vector_store()

for question in [
    "How long do I have to ask for my money back?",
    "Who do I tell about a suspicious email?",
]:
    print("QUESTION:", question)
    for doc in store.similarity_search(question, k=2):
        print("  from", doc.metadata["source"], "->", doc.page_content[:90], "...")
    print()
