from langchain_core.tools import tool

# Mock tools: nothing is actually sent or scheduled.


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'\n---\n{content}"


@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    return (
        f"Meeting '{subject}' ({duration_minutes} min) scheduled for "
        f"{preferred_day} with {len(attendees)} attendees"
    )


@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"
