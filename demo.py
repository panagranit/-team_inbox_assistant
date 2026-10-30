"""End-to-end demo: RAG, short-term, semantic, episodic and procedural memory."""

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

from assistant.agent import build_agent  # noqa: E402
from assistant.memory import (  # noqa: E402
    DEFAULT_MEMORY_PATH,
    add_triage_example,
    ingest_docs,
    make_store,
    set_procedure,
)

USER = "john"


def email(author, subject, body):
    return {
        "author": author,
        "to": "John Doe <john.doe@company.com>",
        "subject": subject,
        "email_thread": body,
    }


def run(agent, mail, thread):
    config = {"configurable": {"langgraph_user_id": USER, "thread_id": thread}}
    result = agent.invoke({"email_input": mail}, config)
    if result.get("messages"):
        print(result["messages"][-1].content)
    return config


def main():
    reset = "--reset" in sys.argv
    if reset:
        DEFAULT_MEMORY_PATH.unlink(missing_ok=True)
    store = make_store()
    print(f"Memory file: {DEFAULT_MEMORY_PATH} ({'reset' if reset else 'persistent'})")
    print(f"RAG: indexed {ingest_docs(store, PROJECT_DIR / 'docs')} doc chunks\n")
    agent = build_agent(store)

    print("=== 1. RAG + semantic memory ===")
    run(
        agent,
        email(
            "Alice Smith <alice.smith@company.com>",
            "Quick question about API documentation",
            "Are /auth/refresh and /auth/validate missing from the specs on purpose?",
        ),
        "thread-1",
    )

    print("\n=== 2. Episodic memory: teach it a correction ===")
    build_status = email(
        "ci@company.com", "Nightly build failed on main", "The nightly build failed. See logs."
    )
    run(agent, build_status, "thread-2")
    add_triage_example(store, USER, build_status, "respond")
    print("-- after correction --")
    run(agent, build_status, "thread-3")

    print("\n=== 3. Procedural memory: update instructions ===")
    set_procedure(
        store, USER, "agent_instructions",
        "Keep replies under 3 sentences, and always propose a 30-minute meeting on a non-focus day.",
    )
    run(
        agent,
        email("Bob <bob@company.com>", "Design review", "Can we review the new auth design this week?"),
        "thread-4",
    )


if __name__ == "__main__":
    main()
