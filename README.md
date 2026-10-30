# Engineering Team Inbox Assistant

Demo agent (LangGraph + LangMem) that triages email, answers from project docs, and learns from feedback.

| Capability | Where |
|---|---|
| RAG | `docs/*.md` indexed into the store; `search_docs` tool (`assistant/memory.py`) |
| Short-term memory | `MemorySaver` checkpointer, per `thread_id` (`assistant/agent.py`) |
| Semantic memory | LangMem `manage_memory` / `search_memory` tools |
| People memory | Per-person profiles and notes (`remember_person_note` / `recall_person`) |
| Episodic memory | Past triage corrections used as few-shot examples |
| Procedural memory | Editable triage rules and agent instructions in the store |

Email, calendar tools are mocks (`assistant/tools.py`). Nothing is sent.

## Run

```bash
git clone https://github.com/panagranit/-team_inbox_assistant.git team_inbox_assistant
cd team_inbox_assistant
uv venv --python 3.13 .venv-py313
uv pip install --python .venv-py313/bin/python -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY to .env
.venv-py313/bin/python demo.py
```

The demo loads `.env` and `docs/` relative to `demo.py`, so it also works when
you run it by path from another directory:

```bash
/path/to/team_inbox_assistant/.venv-py313/bin/python /path/to/team_inbox_assistant/demo.py
```

The store is persistent: everything except short-term (per-thread) memory is saved to `data/memory.json` (git-ignored) with its embeddings, so semantic memory, people profiles, triage corrections, procedures and indexed docs survive restarts. Docs are re-embedded only when their text changes. Run `python demo.py --reset` to start from an empty memory.

People profiles live under `(email_assistant, "people")` and are managed by the agent through the `remember_person_note` / `recall_person` tools. For multi-process or production use, swap `PersistentStore` for a database-backed store such as `PostgresStore`.
