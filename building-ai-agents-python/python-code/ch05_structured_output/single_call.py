import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.chat_models import init_chat_model


class Ticket(BaseModel):
    """A triaged support ticket."""

    category: Literal["billing", "technical", "other"] = Field(
        description="The best-fitting category"
    )
    urgency: int = Field(description="Urgency from 1 (low) to 5 (critical)")
    summary: str = Field(description="One-sentence summary of the problem")


model = init_chat_model(os.environ["MODEL"], temperature=0)
triage = model.with_structured_output(Ticket)

ticket = triage.invoke(
    "I was charged twice for my annual plan and I need this fixed today!"
)

print(ticket)
print(ticket.category, ticket.urgency)
