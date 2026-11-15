"""Gradio frontend. Start backend first, then: python frontend/app.py"""
import os
import uuid

import gradio as gr
import requests

API = os.getenv("BACKEND_URL", "http://localhost:8000")


def respond(message, history, mode, session_id):
    try:
        r = requests.post(f"{API}/chat", timeout=300,
                          json={"message": message, "session_id": session_id, "mode": mode})
        r.raise_for_status()
        d = r.json()
        return f"{d['answer']}\n\n_agent: {d['agent']} | tools: {', '.join(d['tools']) or 'none'}_"
    except Exception as e:
        return f"Error contacting backend: {e}"


with gr.Blocks(title="IIT Jodhpur Academic Helpdesk") as demo:
    gr.Markdown("# IIT Jodhpur Academic Helpdesk")
    mode = gr.Radio(["multi", "single"], value="multi", label="Agent mode")
    sid = gr.State(lambda: uuid.uuid4().hex)
    gr.ChatInterface(respond, additional_inputs=[mode, sid], type="messages")

if __name__ == "__main__":
    demo.launch(server_port=int(os.getenv("PORT", "7860")))
