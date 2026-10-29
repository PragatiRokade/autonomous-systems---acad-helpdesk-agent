from openai import OpenAI
client = OpenAI()
def get_embeddings(chunk):
# embeddings generation function
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding