from src.parsing import parse_all
from src.contextaul_retrival import all_chunks_context
from src.embeddings import get_embeddings
from src.vector_store import store_all_chunks
import chromadb
from src.rank_calculation import get_final_results_rrf
from src.sparse_rank import tf_idf
import openai
from openai import OpenAI
from loguru import logger
from src.prompts import*
chroma_client = chromadb.Client()
import os, asyncio


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="LSTMNotes")#chromadb collection 

pdf_path = "LSTM.pdf"
url = "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
chunk_list = parse_all(pdf_path, url)#getting list of chunks
all_chunks_context(chunk_list,8)#contextaul retrival

if collection.count() == 0:
    logger.info("Collection is empty. Storing all chunks...")
    store_all_chunks(chunk_list,collection,max_workers=8)#call only when empty
else:
    logger.info("Collection already contains data. Skipping storage.")
def run_rag(query: str, top_k: int = 5) -> str:
    embedding_query = get_embeddings(query)
    results_dense = collection.query(query_embeddings=[embedding_query], n_results=top_k)#dense rank
    #print(results_dense)
    topk_sparse_indices = tf_idf(chunk_list, query, top_k)#sparse rank
    topk_chunks,l = get_final_results_rrf(results_dense, chunk_list, topk_sparse_indices, k=5)#final rank using rank fusion
    passages = []
    for idx, ch in enumerate(topk_chunks, start=1):
        passages.append(
            f"Passage {idx} :\n"
            "```text\n"
            f"{ch['text']}\n"
            "```"
        )
    joined_passages = "\n\n".join(passages)


    #user_prompt= "the given Context"+ topk_chunks +"\n"+" The is Query"+query
    user_prompt = f"""
    You are a Question and answer  technical assistant.
    When answering: 
    • If any passage contains the answer, then answer it using the provided context only.
    • Otherwise if passage are totally irrelavant to query , say “No explicit answer found in the passages.”

    QUESTION:
    {query}

    RETRIEVED PASSAGES:
    {joined_passages}

    INSTRUCTIONS:
    – Look at each Passage in order.
    – Summarize or quote the relevant lines.
    – Do not hallucinate or invent new content.

    ANSWER:
    """
    response = client.chat.completions.create(
        model="gpt-4",  
        messages=[
           
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5 
    )
    logger.debug(response.choices[0].message.content)
    print(l)
    return response.choices[0].message.content,l
    