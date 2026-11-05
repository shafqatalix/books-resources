import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It's sunny and 22 degrees in {city}."


agent = create_agent(
    model=os.environ["MODEL"],
    tools=[get_weather],
    system_prompt="You are a concise weather assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]}
)

print(result["messages"][-1].content)

print("\n--- the whole conversation ---")
for message in result["messages"]:
    print(type(message).__name__, "->", message.content)
