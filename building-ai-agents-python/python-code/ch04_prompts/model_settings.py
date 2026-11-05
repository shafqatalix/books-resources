import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    os.environ["MODEL"],
    temperature=0,
    max_tokens=300,
    timeout=60,
)

print(model.invoke("Give me one tip for writing a clear system prompt.").content)
