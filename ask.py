"""Ask the helpdesk your own questions (default LLM: Gemini, needs GEMINI_API_KEY in .env).

Usage:
    python ask.py                      # interactive chat
    python ask.py "What is the attendance rule?"
    python ask.py --mode single "..."  # single-agent baseline
    LLM_PROVIDER=ollama python ask.py  # use local Ollama instead
"""
import argparse
import asyncio
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from helpdesk import config as C
from helpdesk.agents import root_agent, single_agent


async def chat(agent, questions):
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="helpdesk_ask", session_service=svc)
    sid = uuid.uuid4().hex
    await svc.create_session(app_name="helpdesk_ask", user_id="user", session_id=sid)

    async def one(q):
        msg = types.Content(role="user", parts=[types.Part(text=q)])
        async for ev in runner.run_async(user_id="user", session_id=sid, new_message=msg):
            if ev.is_final_response() and ev.content and ev.content.parts:
                text = "".join(p.text or "" for p in ev.content.parts).strip()
                if text:
                    print(f"\n[{ev.author}] {text}\n")

    if questions:
        for q in questions:
            await one(q)
        return
    print(f"Model: {C.LLM_MODEL}  (type 'exit' to quit)")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q:
            try:
                await one(q)
            except Exception as e:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "multi"], default="multi")
    ap.add_argument("question", nargs="*", help="omit for interactive mode")
    a = ap.parse_args()
    asyncio.run(chat(root_agent if a.mode == "multi" else single_agent, [" ".join(a.question)] if a.question else []))
