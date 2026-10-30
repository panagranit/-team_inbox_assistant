"""Memory layers, all backed by one LangGraph store.

- RAG:        ("team_docs",)                      chunks of project docs
- Semantic:   (app, user, "collection")           facts, managed by the agent via langmem tools
- People:     (app, "people")                     per-person profiles and notes
- Episodic:   (app, user, "examples")             past triage corrections (few-shot)
- Procedural: (user,) keys like "triage_ignore"   editable instructions
- Short-term: LangGraph checkpointer (see agent.py), scoped by thread_id

Everything in the store except short-term memory is persisted to data/memory.json.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langgraph.store.base import Item, PutOp
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

APP = "email_assistant"
DEFAULT_MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.json"

DEFAULT_PROCEDURES = {
    "triage_ignore": "Marketing newsletters, spam emails, mass company announcements",
    "triage_notify": "Team member out sick, build system notifications, project status updates",
    "triage_respond": "Direct questions from team members, meeting requests, critical bug reports",
    "agent_instructions": "Use the tools when appropriate. Ground answers about the project in search_docs results and cite the document.",
}


class PersistentStore(InMemoryStore):
    """InMemoryStore that snapshots items and their embeddings to a JSON file.

    Loading restores the vectors directly, so no embedding calls are needed on startup.
    """

    def __init__(self, path: str | Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self._load()

    def batch(self, ops):
        ops = list(ops)
        results = super().batch(ops)
        if any(isinstance(op, PutOp) for op in ops):
            self._save()
        return results

    async def abatch(self, ops):
        ops = list(ops)
        results = await super().abatch(ops)
        if any(isinstance(op, PutOp) for op in ops):
            self._save()
        return results

    def _save(self) -> None:
        entries = [
            {
                "namespace": list(ns),
                "key": key,
                "value": item.value,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "vectors": self._vectors.get(ns, {}).get(key, {}),
            }
            for ns, items in self._data.items()
            for key, item in items.items()
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries))
        os.replace(tmp, self.path)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for e in json.loads(self.path.read_text()):
            ns = tuple(e["namespace"])
            self._data[ns][e["key"]] = Item(
                value=e["value"],
                key=e["key"],
                namespace=ns,
                created_at=datetime.fromisoformat(e["created_at"]),
                updated_at=datetime.fromisoformat(e["updated_at"]),
            )
            for path, vector in e["vectors"].items():
                self._vectors[ns][e["key"]][path] = vector


def make_store(path: str | Path | None = DEFAULT_MEMORY_PATH) -> InMemoryStore:
    """Persistent by default; pass path=None for a throwaway in-memory store."""
    index = {"embed": "openai:text-embedding-3-small"}
    return PersistentStore(path, index=index) if path else InMemoryStore(index=index)


def ingest_docs(store, docs_dir: str | Path) -> int:
    """RAG ingestion: paragraph chunks, embedded only when new or changed; removed chunks are deleted."""
    wanted = {}
    for path in sorted(Path(docs_dir).glob("*.md")):
        for i, chunk in enumerate(path.read_text().split("\n\n")):
            if chunk.strip():
                wanted[f"{path.stem}-{i}"] = {"text": chunk.strip(), "source": path.name}
    existing = {item.key: item.value for item in store.search(("team_docs",), limit=10_000)}
    for key in existing.keys() - wanted.keys():
        store.delete(("team_docs",), key)
    for key, value in wanted.items():
        if existing.get(key) != value:
            store.put(("team_docs",), key, value)
    return len(wanted)


def _person_key(email_or_name: str) -> str:
    return re.sub(r"[^a-z0-9@.]+", "-", email_or_name.lower()).strip("-")


def remember_person(store, name: str, email: str = "", note: str = "") -> str:
    """Create or extend a person's profile; notes accumulate across runs."""
    key = _person_key(email or name)
    current = store.get((APP, "people"), key)
    profile = current.value if current else {"name": name, "email": email, "notes": []}
    profile["name"] = name or profile["name"]
    profile["email"] = email or profile["email"]
    if note and note not in profile["notes"]:
        profile["notes"].append(note)
    store.put((APP, "people"), key, profile)
    return key


def make_people_tools(store):
    @tool
    def remember_person_note(name: str, email: str, note: str) -> str:
        """Save a durable note about a person (role, preferences, history with them)."""
        return f"Saved profile {remember_person(store, name, email, note)}"

    @tool
    def recall_person(query: str) -> str:
        """Look up what is known about a person by name, email or topic."""
        hits = store.search((APP, "people"), query=query, limit=3)
        return "\n\n".join(
            f"{h.value['name']} <{h.value['email']}>: " + "; ".join(h.value["notes"])
            for h in hits
        ) or "No matching people."

    return [remember_person_note, recall_person]


def make_search_docs_tool(store):
    @tool
    def search_docs(query: str) -> str:
        """Search the team's project documentation and policies."""
        hits = store.search(("team_docs",), query=query, limit=3)
        return "\n\n".join(f"[{h.value['source']}] {h.value['text']}" for h in hits) or "No matches."

    return search_docs


def memory_tools():
    """Semantic memory tools (the namespace placeholder is filled from config)."""
    ns = (APP, "{langgraph_user_id}", "collection")
    return [create_manage_memory_tool(namespace=ns), create_search_memory_tool(namespace=ns)]


def get_procedure(store, user_id: str, key: str) -> str:
    item = store.get((user_id,), key)
    if item is None:
        store.put((user_id,), key, {"prompt": DEFAULT_PROCEDURES[key]})
        return DEFAULT_PROCEDURES[key]
    return item.value["prompt"]


def set_procedure(store, user_id: str, key: str, prompt: str) -> None:
    store.put((user_id,), key, {"prompt": prompt})


def add_triage_example(store, user_id: str, email: dict, label: str) -> None:
    """Episodic memory: record a correction so similar emails are triaged the same way."""
    store.put(
        (APP, user_id, "examples"),
        hashlib.sha1(str(email).encode()).hexdigest(),
        {"email": email, "label": label},
    )


def format_examples(examples) -> str:
    if not examples:
        return "No previous examples."
    return "\n\n---\n\n".join(
        f"Subject: {e.value['email']['subject']}\nFrom: {e.value['email']['author']}\n"
        f"Body: {e.value['email']['email_thread'][:400]}\n> Triage Result: {e.value['label']}"
        for e in examples
    )
