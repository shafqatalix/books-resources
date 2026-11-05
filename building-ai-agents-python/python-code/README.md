# Building AI Agents: companion code

Code for **Building AI Agents with LangChain, LangGraph, and LangSmith: A beginner's guide (Python)**, first edition, by Ali.

Every script in the book lives here, in the folder named for its chapter. Shared code lives in `lib/`.

> **Version note.** The code follows the LangChain 1.x documentation. These libraries change quickly, so pin your versions (`pip freeze > requirements.lock.txt`) once everything works, and check the official documentation if something differs from what the book shows.

## Quick start

```bash
git clone <your-repo-url> building-ai-agents-code
cd building-ai-agents-code   # folder that contains this README, requirements.txt, and lib/

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U -r requirements.txt

cp .env.example .env                 # then edit .env: model, provider key, LangSmith key
python -m ch02_setup.smoke_test      # should print a short reply, and a trace should appear
```

Python 3.11 or newer is recommended (3.10 is the stated minimum for LangChain 1.x).

## How to run things

Always run from the **top folder** of this repository, as modules (dots instead of slashes, no `.py`):

```bash
python -m ch07_assistant.run
```

That is what lets scripts say `from lib.config import MODEL`. If you see `No module named 'lib'`, you are in the wrong folder or ran the file by path.

## Layout

```
building-ai-agents-code/
├── README.md
├── requirements.txt            core packages
├── requirements-optional.txt   local embeddings, serving, judge evaluators
├── .env.example                copy to .env
├── langgraph.json              sketch for `langgraph dev` (chapter 17)
├── docs/                       three sample documents (also created by ch06_tools.make_docs)
├── notes/                      notes you approve in chapters 13 and 18 are written here
├── lib/
│   ├── config.py               loads .env, paths, MODEL, as_text()
│   ├── tools.py                list_documents, read_document, save_note
│   └── index.py                loading, splitting, embeddings, vector store, search_docs
├── tests/                      tests that need no API key
└── ch02_setup/ ... ch18_capstone/
```

## Chapter guide

| Chapter | Folder | Run |
|---|---|---|
| 2. Setup | `ch02_setup` | `python -m ch02_setup.smoke_test` |
| 3. First call and trace | `ch03_first_agent` | `python -m ch03_first_agent.direct_call`<br>`python -m ch03_first_agent.messages_and_roles`<br>`python -m ch03_first_agent.first_agent` |
| 4. Messages and prompts | `ch04_prompts` | `messages`, `model_settings`, `system_prompts`, `templates`, `streaming` (each as `python -m ch04_prompts.<name>`) |
| 5. Structured output | `ch05_structured_output` | `single_call`, `in_agent`, `exercise_meeting_actions` |
| 6. Tools | `ch06_tools` | `python -m ch06_tools.make_docs`, then `python -m ch06_tools.try_tools` (no model call) |
| 7. First real agent | `ch07_assistant` | `python -m ch07_assistant.run` |
| 8. Retrieval | `ch08_retrieval` | `python -m ch08_retrieval.search_demo` |
| 9. RAG as a tool | `ch09_rag` | `python -m ch09_rag.run`<br>`python -m ch09_rag.chain_vs_agent` |
| 10. Memory | `ch10_memory` | `python -m ch10_memory.run` |
| 11. LangGraph basics | `ch11_graph` | `python -m ch11_graph.hand_built_agent` |
| 12. Control flow | `ch12_control` | `router`, `draft_loop`, and `reducers_demo` (no key needed) |
| 13. Human approval | `ch13_approval` | `python -m ch13_approval.run`<br>`python -m ch13_approval.interrupt_demo` (no key needed) |
| 14. Multi-agent | `ch14_multi_agent` | `python -m ch14_multi_agent.manager` |
| 15. Debugging | `ch15_debugging` | `python -m ch15_debugging.tagged_run` |
| 16. Evaluation | `ch16_evals` | `python -m ch16_evals.run_eval` |
| 17. Serving | `ch17_serve` | `uvicorn ch17_serve.api:app --reload`, or `langgraph dev` |
| 18. Capstone | `ch18_capstone` | `python -m ch18_capstone.assistant` |

Chapters 1 and 19 have no code. The exercises in the book are left for you to do, except Exercise 5.1, which has a sample solution.

## Tests

```bash
pytest
```

`tests/test_config.py` runs anywhere. `tests/test_tools.py` runs once `langchain` is installed. Neither needs an API key.

## Notes

- **Embeddings.** Chapters 8 to 10 embed text with OpenAI by default (`OPENAI_API_KEY`). Set `EMBEDDINGS=local` in `.env` and install `requirements-optional.txt` to use a local model instead.
- **Costs.** Each run calls a real model. The examples are small, and agents have a `recursion_limit`, but keep an eye on usage.
- **`langgraph.json`** is a sketch. Follow the current LangGraph CLI documentation for exact fields.
