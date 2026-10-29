"""Single-agent baseline and multi-agent version. Both use Ollama via ADK's LiteLLM connector.

`root_agent` (multi-agent) is what `adk run helpdesk` / `adk web` picks up.
"""
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from . import config as C
from .tools import (search_documents, lookup_timetable, count_courses,
                    lookup_instructor_email, draft_email, log_unanswered)


def _model():
    return LiteLlm(model=C.LLM_MODEL)


REFUSAL = ("I don't have information about that in the academic documents available to me. "
           "I've logged your question so a human can follow up.")

CITE_RULE = (
    "Answer ONLY from tool results. Never use outside knowledge or guess. "
    "After each fact, cite its source in square brackets, e.g. [UG_Ordinance.pdf, page 12] or "
    "[timetable.csv]. "
    f"If tools return nothing relevant, call log_unanswered(question, reason) and reply exactly: '{REFUSAL}'"
)

policy_agent = Agent(
    name="policy_agent", model=_model(),
    description="Answers questions on academic rules, regulations, attendance, exams, grading, "
                "degree requirements, credits, curriculum structure, academic calendar dates, "
                "office orders, notifications and registration procedures.",
    instruction="You are the policy and curriculum expert for IIT Jodhpur. Use search_documents "
                "(rephrase the query if the first search is weak). " + CITE_RULE,
    tools=[search_documents, log_unanswered],
)

schedule_agent = Agent(
    name="schedule_agent", model=_model(),
    description="Answers questions on course slots, timetable, rooms, instructors of a course, "
                "and counts of courses floated by a department/level this semester.",
    instruction="You are the timetable expert. Use lookup_timetable for a specific course and "
                "count_courses for counting questions (level is a number like 400). " + CITE_RULE,
    tools=[lookup_timetable, count_courses, log_unanswered],
)

email_agent = Agent(
    name="email_agent", model=_model(),
    description="Drafts emails from a student to a course instructor (sick leave, leave, makeup exam).",
    instruction="You draft emails for students. Steps: (1) get the course code (ask the student if "
                "missing); (2) call lookup_instructor_email; (3) call draft_email with the dates/reason "
                "given; (4) show the draft and say it has NOT been sent and must be reviewed. "
                "Do not invent dates, reasons, names or email addresses. If the instructor is not found, "
                "call log_unanswered and say so.",
    tools=[lookup_instructor_email, draft_email, log_unanswered],
)

root_agent = Agent(
    name="helpdesk_orchestrator", model=_model(),
    description="IIT Jodhpur Academic Helpdesk orchestrator.",
    instruction=(
        "You route student questions to exactly one specialist and never answer from your own knowledge.\n"
        "- Rules, regulations, attendance, exams, credits, degree requirements, curriculum, "
        "academic calendar dates, office orders, notifications -> transfer to policy_agent\n"
        "- Course slot, timetable, room, instructor, number of courses floated -> schedule_agent\n"
        "- Write/draft an email to an instructor (leave, sick leave, makeup exam) -> email_agent\n"
        "- ANYTHING else (campus dining, transport, general knowledge, chit-chat) is out of scope: "
        f"call log_unanswered and reply: '{REFUSAL}'"
    ),
    sub_agents=[policy_agent, schedule_agent, email_agent],
    tools=[log_unanswered],
)

# Baseline: one agent, all tools
single_agent = Agent(
    name="helpdesk_single", model=_model(),
    description="Single-agent academic helpdesk.",
    instruction=("You are the IIT Jodhpur academic helpdesk. Use search_documents for rules/curriculum, "
                 "lookup_timetable/count_courses for slots and counts, lookup_instructor_email + "
                 "draft_email for emails (draft only, never sent). " + CITE_RULE +
                 " Questions unrelated to academics at IIT Jodhpur are out of scope: refuse the same way."),
    tools=[search_documents, lookup_timetable, count_courses, lookup_instructor_email,
           draft_email, log_unanswered],
)
