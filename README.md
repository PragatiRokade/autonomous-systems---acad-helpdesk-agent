# IIT Jodhpur Academic Helpdesk (Multi-Agent RAG)

**Live demo (Gradio):** https://accepting-subsection-avon-quiet.trycloudflare.com/

**Team:** Sirin Changulani (B23CM1038), Pragati Rokade (B23CM1055), Sonam Sikarwar (B23CM1060), Vandita Gupta (B23CM1061)

A multi-agent helpdesk built with Google ADK that answers questions on academic regulations, curriculum, academic calendar, course slots and course counts, and drafts (but never sends) leave and make-up-exam e-mails to instructors. Every substantive answer cites its source document in square brackets. Questions outside the document collection are refused and written to `data/unanswered_questions.csv` for a human to follow up.

The default configuration uses Gemini (`gemini-3.5-flash`) as the LLM. Retrieval runs locally on CPU, so it fits on a 16 GB laptop. A fully local configuration (Ollama with Llama 3.1 8B or Qwen3 8B) is also supported and was used as the baseline in the evaluation.

## Architecture

```
helpdesk_orchestrator
 |-- policy_agent    -> search_documents (hybrid dense + BM25 retrieval, cited)
 |-- schedule_agent  -> lookup_timetable, count_courses (courses.csv, slots.csv)
 `-- email_agent     -> lookup_instructor_email, draft_email (faculty.csv; draft only)
```

- The orchestrator routes each query to a specialist and refuses out-of-scope queries.
- Every agent can call `log_unanswered`.
- `helpdesk_single` (one agent with all tools) is the single-agent baseline for the ablation.

### Retrieval

- **Chunking:** paragraph-aware, at most 900 characters with 150-character overlap; repeated headers and footers removed.
- **Dense:** `BAAI/bge-small-en-v1.5` (384-d) through `fastembed` (ONNX, local).
- **Sparse:** BM25 Okapi over the same chunks.
- **Fusion:** Reciprocal Rank Fusion, `RRF(d) = sum over methods of 1 / (60 + rank(d))`.
- **Refusal cutoff:** if the best dense distance is above `MAX_BEST_DISTANCE` (`helpdesk/config.py`), the tool returns `no_relevant_context` and the agent declines.
- **Corpus:** `data/docs/raw/` (PDF, HTML), `data/docs/ocr/` (OCR text for scanned circulars such as Circulars 51 and 52) and `data/docs/csv/` (`courses.csv`, `slots.csv`, `faculty.csv`). Duplicates are removed by file hash. Personal data is dummy.

## Project layout

```
helpdesk/   config.py, ingest.py, rag.py, tools.py, agents.py, agent.py, __init__.py
backend/    FastAPI app (main.py)
frontend/   Gradio app (app.py)
eval/       questions.csv, run_eval.py, analyze.py, evaluate_ragas.py, check_distances.py
data/       docs/{raw,ocr,csv}/, unanswered_questions.csv
ask.py      command-line question interface
check_setup.py   checks that the embedding model and LLM respond
```

Both `__init__.py` files must have the `.py` extension.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # set GEMINI_API_KEY (https://aistudio.google.com/apikey)
python check_setup.py
```

The file name of each document is its citation, so keep names descriptive, for example `02_Circular51_Attendance_Policy_Amendment_2024.txt`. Scanned PDFs have no text layer; put their OCR text in `data/docs/ocr/`.

Build the index (re-run whenever documents are added, changed or renamed):

```bash
python -m helpdesk.ingest
```

### Choosing the LLM

| Configuration | How |
| --- | --- |
| Gemini (default) | `GEMINI_API_KEY` in `.env`; model set by `GEMINI_MODEL` |
| Ollama (local) | `LLM_PROVIDER=ollama`, then `ollama pull llama3.1:8b` (or `qwen3:8b`) |

For the Ollama baseline, the Qwen3 swap is run with `LLM_MODEL=ollama_chat/qwen3:8b`.

## Running

Command line:

```bash
python ask.py                                    # interactive
python ask.py "What is the minimum attendance?"
```

ADK developer UI (from the project root, pick `helpdesk`):

```bash
adk web
```

Web app (FastAPI + Gradio):

```bash
uvicorn backend.main:app --port 8000     # terminal 1
python frontend/app.py                   # terminal 2 -> http://localhost:7860
```

Set `BACKEND_URL` if the backend runs on another host.

## Evaluation

`eval/questions.csv` contains 30 prompts across policy, schedule, email, out-of-scope and tricky/adversarial categories, with columns `expected_behavior`, `expected_source`, `expected_page`, `expected_answer_notes` and `corpus_verified`.

**1. Run each configuration three times** (to measure run-to-run instability):

```bash
python -m eval.run_eval --mode single --runs 3 --out eval/results_single_llama.csv
python -m eval.run_eval --mode multi  --runs 3 --out eval/results_multi_llama.csv
python -m eval.run_eval --mode multi  --runs 3 --out eval/results_multi_gemini.csv   # Gemini (default LLM)
```

Each result file has 90 rows (30 questions x 3 runs) with the answer, tools called, final agent, `refused`, `has_citation` and a heuristic `auto_ok`. Delete `data/unanswered_questions.csv` before the first run for a clean log.

**2. Summarise:**

```bash
python -m eval.analyze eval/results_single_llama.csv eval/results_multi_llama.csv eval/results_multi_gemini.csv
```

This reports the pass rate per category, citation completeness, correct-refusal and false-refusal rates, the agent that gave each final answer, and the questions whose behaviour changed between runs.

**3. Ragas metrics** (faithfulness, answer relevancy, answer correctness, context precision, context recall):

```bash
python eval/evaluate_ragas.py
```

**4. Tune refusals:** `python -m eval.check_distances` prints `best_distance` for answerable versus refusal questions; set `MAX_BEST_DISTANCE` between the two groups and re-run.

## Results summary

| Metric | Baseline (Llama 3.1 8B + Nomic) | Gemini 3.5 Flash + BGE-small |
| --- | :---: | :---: |
| Overall auto-check pass rate | 87.8% | 97.8% |
| Citation completeness | 84.2% | 99.1% |
| Out-of-scope refusal | 100% | 100% |
| Run-to-run variation (3 runs) | 22.2% | 0.0% |
| Ragas faithfulness | n/a | 0.962 |

See the report for the full category breakdown, Ragas results and case studies.
