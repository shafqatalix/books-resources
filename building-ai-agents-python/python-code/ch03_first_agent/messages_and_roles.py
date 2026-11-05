import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(os.environ["MODEL"])

plain = model.invoke("Explain what an API is in one sentence.")
print("WITHOUT a system message:\n", plain.content, "\n")

messages = [
    {"role": "system", "content": "You are a patient teacher. Use an everyday analogy."},
    {"role": "user", "content": "Explain what an API is in one sentence."},
]

reply = model.invoke(messages)
print("WITH a system message:\n", reply.content)
