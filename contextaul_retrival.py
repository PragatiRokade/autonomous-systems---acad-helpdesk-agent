
import openai
from openai import OpenAI
from loguru import logger
from src.prompts import prompt_template
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

openai.api_key = os.getenv("OPENAI_API_KEY")#api key 
client = OpenAI()

def generate_context(chunk):
    prompt = prompt_template.format(chunk=chunk)#context generation fior each chunk
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

def all_chunks_context(chunks, max_workers=8):
    # dispatch all jobs to the thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as pool:#keeping context generation for chunk in batches
        futures = {pool.submit(generate_context, c): c for c in chunks}# creating context for each chunk
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                ctx = fut.result()
                c["text"] = ctx + c["text"]
                logger.debug(f"contextual retrieval done for chunk {c.get('id','?')}")
            except Exception as e:
                logger.error(f"chunk {c.get('id','?')} failed: {e}")

