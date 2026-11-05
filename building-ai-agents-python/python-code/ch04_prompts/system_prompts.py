import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(os.environ["MODEL"], temperature=0)

WEAK = "You are helpful. Answer questions about the company."

STRONG = """You are the Acme Cloud documents assistant.
Answer questions using only the company documents.
Before answering, read the relevant document.
Name the file you used at the end of your answer.
If the documents do not contain the answer, say you could not find it.
Do not guess."""

question = "What is the parental leave policy?"

for label, prompt in [("WEAK", WEAK), ("STRONG", STRONG)]:
    reply = model.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
    )
    print(f"--- {label} prompt ---")
    print(reply.content, "\n")
