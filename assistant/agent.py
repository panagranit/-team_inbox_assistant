from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from .memory import (
    APP,
    format_examples,
    get_procedure,
    make_people_tools,
    make_search_docs_tool,
    memory_tools,
)
from .tools import check_calendar_availability, schedule_meeting, write_email

PROFILE = {
    "name": "John",
    "full_name": "John Doe",
    "background": "Senior software engineer leading a team of 5 developers",
}

TRIAGE_PROMPT = """You are {full_name}'s email assistant. {name} is: {background}.
Classify the email as ignore, notify, or respond.

Ignore: {ignore}
Notify: {notify}
Respond: {respond}

Previous examples (follow these over the rules above):
{examples}"""

AGENT_PROMPT = """You are {full_name}'s executive assistant.

Tools: search_docs (project docs/policies), write_email, schedule_meeting,
check_calendar_availability, manage_memory and search_memory (long-term facts
and preferences), remember_person_note and recall_person (profiles of the
people you correspond with).

Search memory, people and docs before answering. Store durable facts you learn,
and save notes about the sender when you learn something lasting about them.

Instructions:
{instructions}"""


class Router(BaseModel):
    """Route an incoming email."""

    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"]


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


def build_agent(store):
    llm = init_chat_model("openai:gpt-4o-mini")
    router_llm = llm.with_structured_output(Router)
    tools = [
        write_email,
        schedule_meeting,
        check_calendar_availability,
        make_search_docs_tool(store),
        *make_people_tools(store),
        *memory_tools(),
    ]

    def agent_prompt(state, config, store):
        user_id = config["configurable"]["langgraph_user_id"]
        system = AGENT_PROMPT.format(
            full_name=PROFILE["full_name"],
            instructions=get_procedure(store, user_id, "agent_instructions"),
        )
        return [{"role": "system", "content": system}] + state["messages"]

    response_agent = create_react_agent(llm, tools, prompt=agent_prompt, store=store)

    def triage(state: State, config, store) -> Command[Literal["response_agent", "__end__"]]:
        user_id = config["configurable"]["langgraph_user_id"]
        email = state["email_input"]
        examples = store.search((APP, user_id, "examples"), query=str({"email": email}))
        system = TRIAGE_PROMPT.format(
            full_name=PROFILE["full_name"],
            name=PROFILE["name"],
            background=PROFILE["background"],
            ignore=get_procedure(store, user_id, "triage_ignore"),
            notify=get_procedure(store, user_id, "triage_notify"),
            respond=get_procedure(store, user_id, "triage_respond"),
            examples=format_examples(examples),
        )
        user = (
            f"From: {email['author']}\nTo: {email['to']}\n"
            f"Subject: {email['subject']}\n\n{email['email_thread']}"
        )
        result = router_llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        print(f"Triage: {result.classification.upper()} ({result.reasoning})")
        if result.classification == "respond":
            return Command(
                goto="response_agent",
                update={"messages": [{"role": "user", "content": f"Respond to this email:\n{user}"}]},
            )
        return Command(goto=END)

    graph = StateGraph(State)
    graph.add_node("triage", triage)
    graph.add_node("response_agent", response_agent)
    graph.add_edge(START, "triage")
    # The checkpointer provides short-term memory, keyed by thread_id.
    return graph.compile(store=store, checkpointer=MemorySaver())
