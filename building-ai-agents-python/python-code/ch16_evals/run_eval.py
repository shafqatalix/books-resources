from langchain.agents import create_agent
from langsmith import Client

from lib.config import MODEL, as_text
from lib.index import search_docs

DATASET = "docs-assistant-v1"

EXAMPLES = [
    {
        "inputs": {"question": "How long do customers have to request a full refund?"},
        "outputs": {"any_of": ["30 days"]},
    },
    {
        "inputs": {"question": "Who do I email about a suspected phishing message?"},
        "outputs": {"any_of": ["security@acme.example"]},
    },
    {
        "inputs": {"question": "What time is the team standup?"},
        "outputs": {"any_of": ["9:30"]},
    },
    {
        "inputs": {"question": "Can I get a refund on a monthly plan after 45 days?"},
        "outputs": {"any_of": ["not refundable", "no refund", "cannot", "can't", "not eligible"]},
    },
    {
        "inputs": {"question": "What is the parental leave policy?"},
        "outputs": {"any_of": ["could not find", "couldn't find", "not in the documents", "no information"]},
    },
]

SYSTEM_PROMPT = (
    "You are the Acme Cloud documents assistant. Answer only from passages "
    "returned by search_docs, cite the source file, and say you could not find "
    "the answer if the passages do not contain it. Do not guess."
)

client = Client()

try:
    dataset = client.create_dataset(
        dataset_name=DATASET, description="Questions for the documents assistant."
    )
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    print("Created dataset", DATASET)
except Exception as err:
    print("Dataset not created (it probably exists already):", err)

agent = create_agent(model=MODEL, tools=[search_docs], system_prompt=SYSTEM_PROMPT)


def target(inputs: dict) -> dict:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        config={"recursion_limit": 12},
    )
    return {"answer": as_text(result["messages"][-1])}


def mentions_expected(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    answer = outputs["answer"].lower()
    return any(phrase.lower() in answer for phrase in reference_outputs["any_of"])


results = client.evaluate(
    target,
    data=DATASET,
    evaluators=[mentions_expected],
    experiment_prefix="baseline",
    max_concurrency=2,
)
print("Done. Open the experiment link in LangSmith to see the scores.")
