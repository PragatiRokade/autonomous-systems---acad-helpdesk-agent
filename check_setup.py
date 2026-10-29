import ollama

r = ollama.embed(model="nomic-embed-text", input=["hello world"])
print("Embedding length:", len(r["embeddings"][0]))

r = ollama.chat(model="llama3.1:8b",
                messages=[{"role": "user", "content": "Reply with the single word OK"}])
print("LLM says:", r["message"]["content"])