from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def tf_idf(chunk_list,query_text,k):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunk_list])

    # transform the query
    query_vec = vectorizer.transform([query_text])
    scores_sparse = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # top retrived chunks
    topk_sparse_indices = scores_sparse.argsort()[::-1][:k]
    return topk_sparse_indices
