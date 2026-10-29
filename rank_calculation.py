def get_rank_dict(id_list):
    #Given a ranked list of IDs, return a dict mapping ID to rank.
    return {doc_id: rank for rank, doc_id in enumerate(id_list)}

def reciprocal_rank_fusion(rank_dicts, k=60):
    #Apply Reciprocal Rank Fusion to combine rank dictionaries.
    rrf_scores = {}
    for rank_dict in rank_dicts:
        for doc_id, rank in rank_dict.items():
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def get_final_results_rrf(result_dense, chunk_list, topk_sparse_indices, k=5):
    
    dense_ids = result_dense["ids"][0]  
    sparse_ids = [chunk_list[i]["id"] for i in topk_sparse_indices]

    dense_rank = get_rank_dict(dense_ids)
    sparse_rank = get_rank_dict(sparse_ids)

    fused = reciprocal_rank_fusion([dense_rank, sparse_rank], k=60)

    topk_fused_ids = [doc_id for doc_id, _ in fused[:k]]
    final_chunks=[]
    l=[]
               # your existing list of [id, loc]

# collect each chunk as its own item
    for chunk in chunk_list:
        if chunk["id"] in topk_fused_ids:
            final_chunks.append({
                "id":   chunk["id"],
                "text": chunk["text"],
                
            })
            l.append([chunk["id"], chunk["metadata"]["loc"]])

    

    return final_chunks,l
