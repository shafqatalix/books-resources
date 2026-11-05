import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(os.environ["MODEL"])

reply = model.invoke("Explain what an API is in one sentence.")

print(type(reply).__name__)
print(reply.content)
