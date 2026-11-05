import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

model = init_chat_model(os.environ["MODEL"])

# Dictionary form
as_dicts = [
    {"role": "system", "content": "You are a patient teacher."},
    {"role": "user", "content": "What is an API?"},
]

# Class form
as_objects = [
    SystemMessage(content="You are a patient teacher."),
    HumanMessage(content="What is an API?"),
]

print("Dictionaries:", model.invoke(as_dicts).content, "\n")
print("Objects:", model.invoke(as_objects).content)
