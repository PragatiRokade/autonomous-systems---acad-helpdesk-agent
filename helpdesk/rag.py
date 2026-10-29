"""Hybrid retrieval: dense (Chroma) + BM25, fused with Reciprocal Rank Fusion."""
import json
import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from . import config as C
from .ingest import embed


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


@lru_cache(maxsize=1)
def _load():
    chunks = json.loads(C.CHUNKS_JSON.read_text())
    bm25 = BM25Okapi([_tok(c["embed_text"]) for c in chunks])
    col = chromadb.PersistentClient(path=str(C.CHROMA_DIR)).get_collection("helpdesk")
    return chunks, bm25, col


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, top_k: int = C.TOP_K, pool: int = 15) -> dict:
    chunks, bm25, col = _load()
    by_id = {c["id"]: c for c in chunks}

    dense = col.query(query_embeddings=embed([query]), n_results=pool)
    dense_ids = dense["ids"][0]
    best_distance = dense["distances"][0][0] if dense["distances"][0] else 1.0

    bm = bm25.get_scores(_tok(query))
    sparse_ids = [chunks[i]["id"] for i in sorted(range(len(bm)), key=lambda i: bm[i], reverse=True)[:pool]
                  if bm[i] > 0]

    fused = rrf([dense_ids, sparse_ids])[:top_k]        # keeps fused ORDER
    results = [{
        "chunk_id": cid,
        "text": by_id[cid]["text"],
        "source": by_id[cid]["source"],
        "loc": by_id[cid]["loc"],
    } for cid, _ in fused]
    return {"best_distance": best_distance, "results": results}
