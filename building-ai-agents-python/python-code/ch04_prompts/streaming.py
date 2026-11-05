import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(os.environ["MODEL"])

for chunk in model.stream("Explain what an API is in three sentences."):
    print(chunk.content, end="", flush=True)
print()
