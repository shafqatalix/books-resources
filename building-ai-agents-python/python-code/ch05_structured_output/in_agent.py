import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent


class Ticket(BaseModel):
    """A triaged support ticket."""

    category: Literal["billing", "technical", "other"] = Field(
        description="The best-fitting category"
    )
    urgency: int = Field(description="Urgency from 1 (low) to 5 (critical)")
    summary: str = Field(description="One-sentence summary of the problem")


agent = create_agent(
    model=os.environ["MODEL"],
    tools=[],
    system_prompt="You triage customer support tickets.",
    response_format=Ticket,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "The app crashes whenever I open settings."}]}
)

ticket = result["structured_response"]
print(ticket)
