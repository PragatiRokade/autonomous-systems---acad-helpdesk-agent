"""FastAPI backend. Run from repo root: uvicorn backend.main:app --port 8000"""
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from helpdesk import config as C
from helpdesk.agents import root_agent, single_agent

APP = "helpdesk_api"
app = FastAPI(title="Academic Helpdesk API")
svc = InMemorySessionService()
runners = {
    "multi": Runner(agent=root_agent, app_name=APP, session_service=svc),
    "single": Runner(agent=single_agent, app_name=APP, session_service=svc),
}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: str = "multi"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    agent: str = ""
    tools: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok", "model": C.LLM_MODEL}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.mode not in runners:
        raise HTTPException(400, "mode must be 'multi' or 'single'")
    sid = req.session_id or uuid.uuid4().hex
    if not await svc.get_session(app_name=APP, user_id="web", session_id=sid):
        await svc.create_session(app_name=APP, user_id="web", session_id=sid)
    msg = types.Content(role="user", parts=[types.Part(text=req.message)])
    answer, agent, tools = "", "", []
    try:
        async for ev in runners[req.mode].run_async(user_id="web", session_id=sid, new_message=msg):
            tools += [fc.name for fc in ev.get_function_calls() or []]
            if ev.is_final_response() and ev.content and ev.content.parts:
                text = "".join(p.text or "" for p in ev.content.parts).strip()
                if text:
                    answer, agent = text, ev.author
    except Exception as e:
        raise HTTPException(500, str(e))
    return ChatResponse(answer=answer, session_id=sid, agent=agent, tools=tools)
