import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
chroma_client = chromadb.Client()
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(chunk):
# embeddings generation function
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding
#vector store 
def vector_store(collection, embed_vector, chunk, chunk_no):
    collection.add(
        embeddings=[embed_vector],
        documents=[chunk["text"]],
        metadatas=[chunk["metadata"]],
        ids=[f"id{chunk_no}"]
    )
def process_and_store(chunk, idx,collection):
    # embedding creation
    embedding = get_embeddings(chunk["text"])
    # 2️⃣ storein chromdb
    vector_store(collection, embedding, chunk, idx)
    logger.debug(f"✅ stored embedding for chunk {idx}")
    return idx

def store_all_chunks(chunks,collection, max_workers=8):
    """
    chunks: list of dicts with at least 'text' and 'id'
    max_workers: number of threads to spin up
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit one job per chunk
        futures = {
            pool.submit(process_and_store, chunk, i,collection): (chunk, i)
            for i, chunk in enumerate(chunks, start=1)
        }
        for fut in as_completed(futures):
            chunk, idx = futures[fut]
            try:
                fut.result()  # will re‑raise any exception inside process_and_store
            except Exception as e:
                logger.error(f"❌ failed for chunk {idx}: {e}")