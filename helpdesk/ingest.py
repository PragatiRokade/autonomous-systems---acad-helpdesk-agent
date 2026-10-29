"""Parse every file in data/docs, chunk it, embed with Ollama, store in ChromaDB.

Run once (and again whenever documents change):  python -m helpdesk.ingest
"""
import hashlib
import json
import re

import chromadb
import ollama
from bs4 import BeautifulSoup
from loguru import logger

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

from . import config as C

_ollama = ollama.Client(host=C.OLLAMA_HOST)

# noise found in the scraped iitj.ac.in pages
_WEB_NOISE = "###147852369$$$_RedirectToLoginPage_%%%963258741!!!"
_WEB_FOOTER = re.compile(r"Important links\s+CCCD @ IITJ.*", re.S)


def embed(texts: list[str]) -> list[list[float]]:
    return _ollama.embed(model=C.EMBED_MODEL, input=texts)["embeddings"]


def _hard_split(p: str) -> list[str]:
    """Break one very long paragraph (e.g. PDFs with no blank lines) into <= CHUNK_CHARS pieces."""
    if len(p) <= int(C.CHUNK_CHARS * 1.5):
        return [p]
    out, cur = [], []
    size = 0
    for w in p.split(" "):
        if cur and size + len(w) + 1 > C.CHUNK_CHARS:
            out.append(" ".join(cur))
            cur = cur[-25:]                       # ~25 words of overlap
            size = sum(len(x) + 1 for x in cur)
        cur.append(w)
        size += len(w) + 1
    if cur:
        out.append(" ".join(cur))
    return out


def _split(text: str) -> list[str]:
    """Paragraph-aware splitter with overlap; skips empty/tiny pieces."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [piece for p in paras if len(p) > 20 for piece in _hard_split(p)]
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
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(path)
        for i, page in enumerate(doc, start=1):
            for piece in _split(page.get_text()):
                items.append({"text": piece, "source": path.name, "loc": f"page {i}"})
        if not items:
            logger.warning(f"{path.name}: NO extractable text (scanned PDF?) - skipped")
    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for piece in _split(soup.get_text("\n")):
            items.append({"text": piece, "source": path.name, "loc": "html"})
    elif suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"SOURCE_URL:\s*(\S+)", raw)
        loc = m.group(1) if m else "text"          # scraped pages: cite the original URL
        raw = _WEB_FOOTER.sub("", raw.replace(_WEB_NOISE, " "))
        for piece in _split(raw):
            items.append({"text": piece, "source": path.name, "loc": loc})
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
    chunks, seen_text, seen_files = [], set(), set()
    files = [p for p in sorted(C.DOCS_DIR.iterdir()) if p.is_file() and not p.name.startswith(".")]
    for path in files:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen_files:
            logger.warning(f"{path.name}: identical to an earlier file - skipped")
            continue
        seen_files.add(digest)
        for it in parse_file(path):
            key = hashlib.md5(re.sub(r"\W+", "", it["text"].lower()).encode()).hexdigest()
            if key in seen_text:                    # identical chunk already stored
                continue
            seen_text.add(key)
            it["id"] = f"c{len(chunks)}"            # ONE id scheme used everywhere (dense + BM25)
            it["embed_text"] = it["text"]
            if C.USE_CONTEXTUAL_RETRIEVAL:
                it["embed_text"] = add_context(it, path.name) + "\n" + it["text"]
            chunks.append(it)
    logger.info(f"{len(chunks)} chunks from {len(seen_files)} unique files")
    if not chunks:
        raise SystemExit("No chunks produced - put documents in data/docs/")

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

    C.CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    logger.info("ingestion done")


if __name__ == "__main__":
    main()