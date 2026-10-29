prompt_template = (
    "Given this chunk from a document:\n\n"
    "{chunk}\n\n"
    "Generate a brief summary or surrounding context (50–100 tokens) that helps situate it within the full document."
)
system_prompt = (
    "You are a knowledgeable assistant. "
    "Answer the user's question using the provided context. "
    "If context is totally irrelavant to the query then say query related context not in Document "
)
prompt2="a"
# prompt = f"""
# You are a precise technical assistant.  
# When responding, you MUST use the exact text from any retrieved passage that answers the question.  
# If the retrieved passage contains the answer, quote or paraphrase it directly; otherwise say “No explicit answer found.”

# QUESTION:
# {question}

# RETRIEVED PASSAGES (ranked):
# 1. \"\"\"{top_chunks[0]['text']}\"\"\"  
# 2. \"\"\"{top_chunks[1]['text']}\"\"\"  
# 3. \"\"\"{top_chunks[2]['text']}\"\"\"

# INSTRUCTIONS:
# - Look at each passage in order.
# - If a passage answers the question, **summarize or quote** that section verbatim.
# - Do not hallucinate or defer to “document does not provide”; only say that if none of the passages contain an answer.
# - At the end, list which chunk(s) you used.

# ANSWER:
# """