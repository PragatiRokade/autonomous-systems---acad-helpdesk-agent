# IIT Jodhpur Academic Helpdesk (multi-agent RAG)

Google ADK + Ollama (via LiteLLM) + ChromaDB + BM25. Everything runs locally on a 16 GB laptop.

The agent answers questions on academic rules, curriculum, calendars, course slots and counts, and drafts
leave / make-up-exam emails to instructors. Answers cite their source document. Anything out of scope is
refused and written to `data/unanswered_questions.csv` for a human to follow up.

## Architecture

```
helpdesk_orchestrator
 |-- policy_agent    -> search_documents (hybrid dense + BM25 RAG, cited)
 |-- schedule_agent  -> lookup_timetable, count_courses (timetable.csv)
 `-- email_agent     -> lookup_instructor_email, draft_email (draft only, never sent)
```

- Every agent can call `log_unanswered`.
- `single_agent` (one agent, all tools) is the baseline used for comparison in the evaluation.
- Retrieval: ChromaDB (cosine, `nomic-embed-text`) + BM25, merged with Reciprocal Rank Fusion. If the best
  dense distance is above `MAX_BEST_DISTANCE` (`helpdesk/config.py`) the tool returns `no_relevant_context`.

## Project layout

```
helpdesk/   config.py, ingest.py, rag.py, tools.py, agents.py, agent.py, __init__.py
eval/       questions.csv, run_eval.py, analyze.py, check_distances.py
data/       docs/ (PDFs), timetable.csv, faculty.csv, chroma/, chunks.json
new.py      copies/cleans documents into data/docs (dedupe, rename)
```

Both `__init__.py` files must have the `.py` extension.

## Setup

```bash
ollama pull llama3.1:8b
ollama pull qwen3:8b            # for the LLM-swap experiment
ollama pull nomic-embed-text
pip install -r requirements.txt
python check_setup.py           # confirms the embedding model and LLM respond
```

1. Put the 20-30 documents in `data/docs/`. The file name is the citation, so rename opaque files, e.g.
   `bioreg-638769690932905139.pdf` -> `2_Academic_Regulations_Final_03_09_2019.pdf`.
2. Scanned PDFs have no text and are skipped by ingestion. OCR them first.
3. `data/timetable.csv` columns: course_code, course_name, slot, instructor, room, level, department, type,
   semester. Instructor names must match `faculty.csv` `name`. Personal data is dummy.
4. Build the index (re-run whenever documents change or are renamed):
   ```bash
   python -m helpdesk.ingest
   ```
5. Chat: `adk web` (run from the project root, pick `helpdesk`) or `adk run helpdesk`.

## Tuning refusals

```bash
python -m eval.check_distances
```

Look at `best_distance` for the `answer` rows versus the `refuse`/`out_of_scope` rows and set
`MAX_BEST_DISTANCE` (env var or `config.py`) between the two groups. Re-run after changing it.

## Evaluation

`eval/questions.csv` has 30 prompts: policy, schedule, email, out_of_scope, tricky. Columns:
`expected_behavior` (answer / answer_or_refuse / email_draft / refuse), `expected_source`, `expected_page`,
`expected_answer_notes`, `corpus_verified`.

### 1. Run (3 runs each, so instability can be measured)

Make sure Ollama is running first. Use the same questions for every run.

```bash
python -m eval.run_eval --mode single --runs 3 --out eval/results_single_llama.csv
python -m eval.run_eval --mode multi  --runs 3 --out eval/results_multi_llama.csv
```

LLM swap (Qwen3 8B):

```bash
# Linux / macOS / Git Bash
LLM_MODEL=ollama_chat/qwen3:8b python -m eval.run_eval --mode multi --runs 3 --out eval/results_multi_qwen.csv
# Windows PowerShell
$env:LLM_MODEL="ollama_chat/qwen3:8b"; python -m eval.run_eval --mode multi --runs 3 --out eval/results_multi_qwen.csv
# Windows cmd
set LLM_MODEL=ollama_chat/qwen3:8b && python -m eval.run_eval --mode multi --runs 3 --out eval/results_multi_qwen.csv
```

Each run writes the answer, tools called, final agent, `refused`, `has_citation` and a heuristic `auto_ok`.
Each result file has 90 rows (30 questions x 3 runs). Delete `data/unanswered_questions.csv` before the first run
if you want a clean log, and keep a copy after the last run.

### 2. Score by hand

Open each results CSV in Excel/Sheets and add two columns, filling 1 or 0 for every row:

- `accuracy_manual`
  - answer questions: 1 if the facts match `expected_answer_notes` and nothing is invented
  - refuse questions (23, 26-30): 1 if it refused, did not guess, and did not draft or answer
  - `answer_or_refuse` (2, 13): 1 if it refused, or gave a number that is really in a cited document
  - email questions: 1 if addressed to the right email, says it is not sent, no invented dates or reasons
- `grounded_manual` (answerable rows only; leave blank for refusals)
  - 1 if the cited file name and page actually contain the stated fact, 0 if the citation is missing, wrong or
    points to a file that does not support it

Save the file (keep it as CSV).

### 3. Summarise

```bash
python -m eval.analyze eval/results_single_llama.csv eval/results_multi_llama.csv eval/results_multi_qwen.csv
```

It prints, per file: heuristic pass rate, accuracy per category, overall accuracy, groundedness, correct
refusal rate, false-refusal rate, which agent gave the final answer, and the questions whose behaviour changed
between runs. Copy these numbers into the report (section 5), and paste the old and new answers for each unstable
question.

## Known data limitations

- `timetable.csv` and `faculty.csv` are synthetic (dummy `.example` emails).
- Questions 2 and 13 need `BTech_Regulations_Latest_21July2018.pdf`; without it, refusing is the correct answer.
- Two-column curriculum tables (questions 8, 17, 18) may be extracted out of order; check them by hand.
- Several regulation versions (2018, 2019, 2022 PG) coexist and can disagree; check which one an answer cites.
