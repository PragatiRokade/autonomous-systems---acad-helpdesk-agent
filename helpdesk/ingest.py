"""Parse every file in data/docs, chunk it, embed with Ollama, store in ChromaDB.

Run once (and again whenever documents change):  python -m helpdesk.ingest
"""
import json
import re
import chromadb
import fitz  # PyMuPDF
import ollama
import requests
from bs4 import BeautifulSoup
from loguru import logger

from . import config as C

_ollama = ollama.Client(host=C.OLLAMA_HOST)


def embed(texts: list[str]) -> list[list[float]]:
    return _ollama.embed(model=C.EMBED_MODEL, input=texts)["embeddings"]


def _split(text: str) -> list[str]:
    """Paragraph-aware splitter with overlap; skips empty/tiny pieces."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p) > 20]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > C.CHUNK_CHARS:
            out.append(cur)
            cur = cur[-C.CHUNK_OVERLAP:] + " " + p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out


def parse_file(path) -> list[dict]:
    """Return [{'text', 'source', 'loc'}]. 'source' is the file name -> used in citations."""
    items = []
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        for i, page in enumerate(doc, start=1):
            for piece in _split(page.get_text()):
                items.append({"text": piece, "source": path.name, "loc": f"page {i}"})
    elif path.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
        for piece in _split(soup.get_text("\n")):
            items.append({"text": piece, "source": path.name, "loc": "html"})
    elif path.suffix.lower() in {".txt", ".md"}:
        for piece in _split(path.read_text(errors="ignore")):
            items.append({"text": piece, "source": path.name, "loc": "text"})
    else:
        logger.warning(f"skipping unsupported file {path.name}")
    return items


def add_context(chunk: dict, doc_title: str) -> str:
    """Optional contextual retrieval using the local LLM (slow on a laptop)."""
    prompt = (
        f"Document: {doc_title} ({chunk['loc']})\nChunk:\n{chunk['text']}\n\n"
        "In one short sentence, say what part of the document this chunk is about."
    )
    r = _ollama.chat(model=C.LLM_MODEL.split("/", 1)[-1],
                     messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()


def main():
    chunks = []
    for path in sorted(C.DOCS_DIR.iterdir()):
        for it in parse_file(path):
            it["id"] = f"c{len(chunks)}"           # ONE id scheme used everywhere (dense + BM25)
            it["embed_text"] = it["text"]
            if C.USE_CONTEXTUAL_RETRIEVAL:
                it["embed_text"] = add_context(it, path.name) + "\n" + it["text"]
            chunks.append(it)
    logger.info(f"{len(chunks)} chunks from {len(list(C.DOCS_DIR.iterdir()))} files")

    client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
    try:
        client.delete_collection("helpdesk")
    except Exception:
        pass
    col = client.create_collection("helpdesk", metadata={"hnsw:space": "cosine"})

    B = 32
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=embed([c["embed_text"] for c in batch]),
            documents=[c["text"] for c in batch],           # original text, used for answers/citations
            metadatas=[{"source": c["source"], "loc": c["loc"]} for c in batch],
        )
        logger.debug(f"embedded {i + len(batch)}/{len(chunks)}")

    C.CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False))
    logger.info("ingestion done")


if __name__ == "__main__":
    main()
