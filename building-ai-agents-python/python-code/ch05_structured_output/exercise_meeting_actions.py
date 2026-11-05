"""One possible solution to Exercise 5.1."""
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.chat_models import init_chat_model


class MeetingAction(BaseModel):
    """A single action item from a meeting."""

    task: str = Field(description="What needs to be done")
    owner: str = Field(description="Who is responsible")
    due_date: str | None = Field(default=None, description="Due date if mentioned, else null")


class MeetingNotes(BaseModel):
    """All action items found in a meeting transcript."""

    actions: list[MeetingAction] = Field(description="Action items. Empty if there are none.")


model = init_chat_model(os.environ["MODEL"], temperature=0)
extractor = model.with_structured_output(MeetingNotes)

TRANSCRIPTS = [
    "Priya will send the budget draft by Friday. Marco to book the venue. Everyone agreed the demo went well.",
    "We reviewed last week's numbers and talked about the roadmap. No decisions were made.",
]

for text in TRANSCRIPTS:
    notes = extractor.invoke(text)
    print(f"{len(notes.actions)} action(s)")
    for action in notes.actions:
        print(" -", action.task, "|", action.owner, "|", action.due_date)
    print()
