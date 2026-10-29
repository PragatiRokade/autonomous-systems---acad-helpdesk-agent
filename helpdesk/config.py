import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
TIMETABLE_CSV = ROOT / "data" / "timetable.csv"
FACULTY_CSV = ROOT / "data" / "faculty.csv"
CHROMA_DIR = ROOT / "data" / "chroma"
CHUNKS_JSON = ROOT / "data" / "chunks.json"
UNANSWERED_LOG = ROOT / "data" / "unanswered_questions.csv"

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_HOST)  # used by LiteLLM
# Swap the LLM here for the "replace the LLM" experiment, e.g. ollama_chat/qwen3:8b
LLM_MODEL = os.getenv("LLM_MODEL", "ollama_chat/llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Retrieval
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K = 5
# Cosine distance of the best dense hit above which we say "nothing relevant".
# TUNE this on your own data using the in-scope vs out-of-scope questions.
MAX_BEST_DISTANCE = float(os.getenv("MAX_BEST_DISTANCE", "0.55"))
USE_CONTEXTUAL_RETRIEVAL = os.getenv("USE_CONTEXTUAL_RETRIEVAL", "0") == "1"
