"""Plain Python functions; ADK turns them into tools using the signature + docstring."""
import csv
from datetime import datetime

import pandas as pd

from . import config as C
from .rag import retrieve


def search_documents(query: str) -> dict:
    """Search the official IIT Jodhpur documents (rules, regulations, curriculum, office orders,
    registration notices) for passages relevant to the query.

    Args:
        query: A self-contained search query in English, e.g. "minimum attendance to sit in major exam".

    Returns:
        dict with 'status' ('ok' or 'no_relevant_context') and, if ok, 'passages': a list of
        {'text', 'source', 'loc'}. Always cite 'source' and 'loc' when using a passage.
    """
    r = retrieve(query)
    if not r["results"] or r["best_distance"] > C.MAX_BEST_DISTANCE:
        return {"status": "no_relevant_context", "passages": []}
    return {"status": "ok",
            "passages": [{"text": p["text"], "source": p["source"], "loc": p["loc"]} for p in r["results"]]}


def lookup_timetable(course: str) -> dict:
    """Look up a course in the timetable CSV by course code (e.g. 'CSL7570') or by part of the
    course name (e.g. 'Autonomous Systems'). Returns slot, room, instructor, level, department.

    Args:
        course: Course code or course name fragment.
    """
    df = pd.read_csv(C.TIMETABLE_CSV, dtype=str).fillna("")
    m = df[df["course_code"].str.upper() == course.strip().upper()]
    if m.empty:
        m = df[df["course_name"].str.contains(course.strip(), case=False, regex=False)]
    if m.empty:
        return {"status": "not_found"}
    return {"status": "ok", "source": C.TIMETABLE_CSV.name, "rows": m.to_dict("records")}


def count_courses(department: str, level: int, course_type: str = "") -> dict:
    """Count courses floated this semester by a department at a given level (e.g. 400), optionally
    filtered by type such as 'Elective'. Use for 'how many X-level electives did department Y float'.

    Args:
        department: Department name or code, e.g. 'Humanities' or 'HSS'.
        level: Course level as a number, e.g. 400.
        course_type: Optional type filter, e.g. 'Elective'. Empty string = no filter.
    """
    df = pd.read_csv(C.TIMETABLE_CSV, dtype=str).fillna("")
    m = df[df["department"].str.contains(department, case=False, regex=False)]
    m = m[m["level"].astype(int) == int(level)]
    if course_type:
        m = m[m["type"].str.contains(course_type, case=False, regex=False)]
    return {"status": "ok", "source": C.TIMETABLE_CSV.name, "count": int(len(m)),
            "courses": m[["course_code", "course_name"]].to_dict("records")}


def lookup_instructor_email(course_code: str) -> dict:
    """Find the instructor name and email for a course code (e.g. 'CSL7570').

    Args:
        course_code: The course code.
    """
    tt = pd.read_csv(C.TIMETABLE_CSV, dtype=str).fillna("")
    m = tt[tt["course_code"].str.upper() == course_code.strip().upper()]
    if m.empty:
        return {"status": "not_found"}
    name = m.iloc[0]["instructor"]
    fac = pd.read_csv(C.FACULTY_CSV, dtype=str).fillna("")
    f = fac[fac["name"].str.lower() == name.lower()]
    if f.empty:
        return {"status": "instructor_email_missing", "instructor": name}
    return {"status": "ok", "instructor": name, "email": f.iloc[0]["email"], "source": C.FACULTY_CSV.name}


def draft_email(instructor_name: str, instructor_email: str, course_code: str,
                request_type: str, details: str, student_name: str = "[Your Name]",
                roll_number: str = "[Roll Number]") -> dict:
    """Create an email DRAFT for the student to review. The email is NOT sent.

    Args:
        instructor_name: Instructor's name.
        instructor_email: Instructor's email address.
        course_code: Course code.
        request_type: 'sick leave', 'leave' or 'makeup exam'.
        details: Specifics such as dates, duration, reason, which exam.
        student_name: Student's name if known.
        roll_number: Roll number if known.
    """
    subject = f"{course_code}: Request for {request_type}"
    body = (f"Dear Prof. {instructor_name},\n\n"
            f"I am {student_name} ({roll_number}), enrolled in {course_code}. "
            f"I am writing to request {request_type}. {details}\n\n"
            "I would be grateful for your consideration and will provide supporting documents "
            "if required.\n\nThank you.\n\nSincerely,\n"
            f"{student_name}\n{roll_number}")
    return {"status": "draft", "to": instructor_email, "subject": subject, "body": body}


def log_unanswered(question: str, reason: str) -> dict:
    """Log a question the helpdesk cannot answer from its documents/CSVs so a human can follow up.
    Call this whenever you refuse or cannot find an answer.

    Args:
        question: The user's original question.
        reason: Short reason, e.g. 'out of scope' or 'no relevant context'.
    """
    new = not C.UNANSWERED_LOG.exists()
    with open(C.UNANSWERED_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "question", "reason"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), question, reason])
    return {"status": "logged"}
