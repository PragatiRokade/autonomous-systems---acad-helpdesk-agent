# IIT Jodhpur Academic Helpdesk (multi-agent RAG)

Google ADK + Ollama (via LiteLLM) + ChromaDB + BM25, all local (16 GB RAM).

## Architecture
- `helpdesk_orchestrator` routes to:
  - `policy_agent`   -> `search_documents` (hybrid dense+BM25 RAG, cited)
  - `schedule_agent` -> `lookup_timetable`, `count_courses` (CSV)
  - `email_agent`    -> `lookup_instructor_email`, `draft_email` (draft only, never sent)
- Every agent can call `log_unanswered`, which appends to `data/unanswered_questions.csv`.
- `single_agent` (all tools, one agent) is the baseline for comparison.

## Setup
```bash
ollama pull llama3.1:8b
ollama pull qwen3:8b            # for the LLM-swap experiment
ollama pull nomic-embed-text
pip install -r requirements.txt
```
1. Put your 20-30 documents (PDF/HTML/TXT/MD) in `data/docs/`. File names become citations,
   so name them meaningfully (e.g. `UG_Ordinance_2024.pdf`).
2. Replace `data/timetable.csv` and `data/faculty.csv` with real (dummy-personal-data) files.
   Required timetable columns: course_code, course_name, slot, instructor, room, level, department, type, semester.
   Instructor names must match `faculty.csv` `name`.
3. `python -m helpdesk.ingest`  (set `USE_CONTEXTUAL_RETRIEVAL=1` for LLM-generated chunk context; slow)
4. Chat: `adk web` (run from the project root, select `helpdesk`) or `adk run helpdesk`.

## Tuning refusals
`MAX_BEST_DISTANCE` in `helpdesk/config.py` is the cosine-distance cut-off. Print `best_distance` for your
in-scope vs out-of-scope questions and choose a value that separates them.

## Evaluation
Edit `eval/questions.csv` (fill the `<...>` placeholders with your real file names), then:
```bash
python -m eval.run_eval --mode single --runs 3 --out eval/results_single_llama.csv
python -m eval.run_eval --mode multi  --runs 3 --out eval/results_multi_llama.csv
LLM_MODEL=ollama_chat/qwen3:8b python -m eval.run_eval --mode multi --runs 3 --out eval/results_multi_qwen.csv
```
Score accuracy and groundedness by hand in the CSV; the script prints refusal/citation heuristics
and the questions whose output changed between runs.
