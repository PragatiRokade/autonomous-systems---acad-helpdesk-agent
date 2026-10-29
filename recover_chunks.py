import json
import chromadb
from helpdesk import config as C

col = chromadb.PersistentClient(path=str(C.CHROMA_DIR)).get_collection("helpdesk")
data = col.get(include=["documents", "metadatas"])

chunks = [
    {"id": i, "text": d, "embed_text": d, "source": m["source"], "loc": m["loc"]}
    for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
]
chunks.sort(key=lambda c: int(c["id"][1:]))   # c0, c1, c2 ... in original order

C.CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
print("Recovered", len(chunks), "chunks")