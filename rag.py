import json
import os

import faiss
import torch
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer


def retrieve_top_k_embedding(index, query_text, k=10):
    """Function to retrieve top k document chunk indices for a given query vector.
    The query is embedded using the same SentenceTransformer model used during indexing,
    and the nearest neighbors are found using FAISS.
    """
    # Load the same SentenceTransformer model used during indexing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    query_embedding = model.encode(
        [query_text], normalize_embeddings=True, device=device
    )
    distances, indices = index.search(query_embedding, k)
    print("retrieved indices:", distances, indices)
    return indices[0]


def rerank_documents(
    query,
    chunks,
    max_hits: int = 10,
    dynamic_k_threshold: float | None = None,
    min_hits: int = 5,
    rerank_model="ms-marco-MiniLM-L-12-v2",
):
    """Re-rank the chunks using a cross-encoder model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        f"cross-encoder/{rerank_model}", device=device
    )  # ms-marco-MiniLM-L-6-v2 ms-marco-electra-base LeviatanAIResearch/cross-encoder-bert-base-fr-v1
    model_inputs = [[query, chunk] for chunk in chunks]
    scores = model.predict(model_inputs)
    # Sort the scores in decreasing order
    results = [(chunk, score) for chunk, score in zip(chunks, scores)]
    results = sorted(results, key=lambda x: x[1], reverse=True)
    # Filter out the results based on the dynamic k threshold
    if dynamic_k_threshold is not None:
        filtered = [result for result in results if result[1] >= dynamic_k_threshold]
        # If the number of filtered results is less than min_hits, use the original sorted results
        results = filtered if len(filtered) >= min_hits else results[:min_hits]
    return results[:max_hits]


def build_rag_prompt(query: str, docs: list[str]) -> str:
    """
    Builds a prompt for RAG-style querying with ranked documents.

    Args:
        query: The user query.
        docs: List of retrieved documents (strings), ranked by similarity.

    Returns:
        A tuple of instructions and full prompt string for the LLM.
    """
    # You can truncate or select top-k docs here if needed
    doc_section = "\n\n".join(
        [f"[Document {i + 1}]\n{doc.strip()}" for i, doc in enumerate(docs)]
    )

    full_prompt = f"Given the question\n[QUESTION]\n{query}\nUse ONLY the following documents to answer the question.\n\n{doc_section}\n\nAnswer:"

    instructions = f"A conversation between User and Assistant. User will provide a question and a collection of documents that likely contain the answer to the question but were preprocessed removing stopwords and lematized (so they are hard to read and should be interpreted smartly). Assistant will use the documents to answer the question. If the documents don't contain the answer to the question, Assistant will reply 'The search did not return any useful result'. You are a helpful Assistant."

    return instructions, full_prompt


def answer_query(
    index,
    query_text,
    api_key,
    k_retrieve=100,
    min_hits=5,
    max_hits=10,
    dynamic_k_threshold=None,
    model="gpt-4o",
    rerank_model="ms-marco-MiniLM-L-12-v2",
):
    top_k = retrieve_top_k_embedding(index=index, query_text=query_text, k=k_retrieve)
    top_chunks = []
    for vector_id in top_k:
        # IDs were stored as string keys in the metadata store
        str_id = str(vector_id)
        metadata = metadata_store.get(str_id, None)
        if metadata is not None:
            top_chunks.append(metadata["content"])
        else:
            print(f"Vector ID {vector_id}: No metadata found.")

    reranked_docs = [
        x
        for x, _ in rerank_documents(
            query=query_text,
            chunks=top_chunks,
            max_hits=max_hits,
            dynamic_k_threshold=dynamic_k_threshold,
            min_hits=min_hits,
            rerank_model=rerank_model,
        )
    ]

    # Generate the instructions and full prompt
    instructions, full_prompt = build_rag_prompt(query_text, reranked_docs)
    print(f"\n\n\n instructions:\n{instructions}")
    print(f"\n\n\nFull prompt:\n{full_prompt[:2500]}")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=full_prompt,
    )

    return response.output_text


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")

    query_text = "what does a spectrum engineer provide?"

    folder_path = "/home/cerrion/DATATHON/data/normalized_data"

    # Load the index and metadata store from disk
    index = faiss.read_index(
        "/home/cerrion/DATATHON/datathon25/index_all-MiniLM-L6-v2_test.faiss"
    )
    print("Index loaded successfully. Faiss index shape:", index.ntotal)

    with open(
        "/home/cerrion/DATATHON/datathon25/metadata_store_all-MiniLM-L6-v2_test.json",
        "r",
    ) as f:
        metadata_store = json.load(f)

    print(answer_query(index, query_text, api_key))
