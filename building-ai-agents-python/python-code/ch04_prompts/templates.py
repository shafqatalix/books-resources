import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model(os.environ["MODEL"])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {tone} assistant. Keep answers under {limit} words."),
    ("human", "{question}"),
])

messages = prompt.invoke({
    "tone": "friendly",
    "limit": 40,
    "question": "What is an API?",
})

reply = model.invoke(messages)
print(reply.content)
