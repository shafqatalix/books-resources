import os

from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables

from langchain.chat_models import init_chat_model

model = init_chat_model(os.environ["MODEL"])
reply = model.invoke("Reply with exactly one word: ready")
print(reply.content)
