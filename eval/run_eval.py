"""Run every question N times against the single- or multi-agent system and save results.

Usage:
    python -m eval.run_eval --mode multi  --runs 3 --out eval/results_multi_llama.csv
    LLM_MODEL=ollama_chat/qwen3:8b python -m eval.run_eval --mode single --runs 3 --out eval/results_single_qwen.csv

Auto-checks are heuristics; still read the answers and score accuracy/groundedness by hand
for the report (add columns 'accuracy_manual' and 'grounded_manual').
"""
import argparse
import asyncio
import csv
import re
import uuid
from collections import defaultdict
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from helpdesk.agents import root_agent, single_agent

REFUSAL_MARK = "don't have information"
CITE_RE = re.compile(r"\[[^\]]+\]")


async def ask(agent, question: str) -> dict:
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="helpdesk_eval", session_service=svc)
    sid = uuid.uuid4().hex
    await svc.create_session(app_name="helpdesk_eval", user_id="eval", session_id=sid)
    msg = types.Content(role="user", parts=[types.Part(text=question)])
    answer, tools, agents = "", [], []
    async for ev in runner.run_async(user_id="eval", session_id=sid, new_message=msg):
        for fc in ev.get_function_calls() or []:
            tools.append(fc.name)
        if ev.is_final_response() and ev.content and ev.content.parts:
            answer = "".join(p.text or "" for p in ev.content.parts)
            agents.append(ev.author)
    return {"answer": answer.strip(), "tools": tools, "final_agent": agents[-1] if agents else ""}


async def main(mode: str, runs: int, out: str):
    agent = root_agent if mode == "multi" else single_agent
    qs = list(csv.DictReader(open(Path(__file__).parent / "questions.csv", encoding="utf-8-sig")))
    rows = []
    for q in qs:
        for r in range(1, runs + 1):
            try:
                res = await ask(agent, q["question"])
            except Exception as e:  # keep going on model/tool errors
                res = {"answer": f"ERROR: {e}", "tools": [], "final_agent": ""}
            refused = REFUSAL_MARK in res["answer"].lower() or "log_unanswered" in res["tools"]
            cited = bool(CITE_RE.search(res["answer"]))
            exp = q["expected_behavior"]
            if exp == "refuse":
                auto_ok = refused
            elif exp in ("answer", "email_draft"):
                auto_ok = (not refused) and (cited or exp == "email_draft")
            else:
                auto_ok = True
            rows.append({**q, "run": r, "answer": res["answer"], "tools_called": "|".join(res["tools"]),
                         "final_agent": res["final_agent"], "refused": refused, "has_citation": cited,
                         "auto_ok": auto_ok})
            print(f"Q{q['id']} run{r}: refused={refused} cited={cited} tools={res['tools']}")

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary + instability across runs
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r)
    print("\n== summary ==")
    print("auto_ok rate:", sum(r["auto_ok"] for r in rows) / len(rows))
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        print(f"  {cat}: {sum(r['auto_ok'] for r in sub) / len(sub):.2f}")
    print("questions whose behaviour changed across runs:")
    for qid, rs in by_q.items():
        if len({(r["refused"], r["has_citation"]) for r in rs}) > 1 or len({r["answer"] for r in rs}) > 1:
            print(f"  Q{qid}: {rs[0]['question'][:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "multi"], default="multi")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="eval/results.csv")
    a = ap.parse_args()
    asyncio.run(main(a.mode, a.runs, a.out))
