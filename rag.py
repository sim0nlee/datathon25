import os
import logging 
import json 

import faiss
import torch
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer


API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

INDEX = faiss.read_index(
    "index_all-MiniLM-L6-v2_test.faiss"
)
print("Index loaded successfully. Faiss index shape:", INDEX.ntotal)

with open(
        "metadata_store_all-MiniLM-L6-v2_test.json", "r"
) as f:
    METADATA_STORE = json.load(f)


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
    # print("retrieved indices:", distances, indices)
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


INSTRUCTIONS = """
You are a helpful assistant answering user questions based an external database.
When answering:
1. First, decide if you need more information from the database.
2. If you do, call the `search_knowledge_base` function by rewriting the user’s question into a concise, focused search query.
   - Example: For “What are the safety concerns with lithium-ion batteries?”, generate: "lithium-ion battery safety concerns"
3. Once results are returned, use the retrieved documents to answer the question.
   - If the documents contain a clear answer, write a helpful, complete response based on them.
   - If the documents do not contain the answer, respond with: **"The search did not return any useful result."**
4. If the question can be answered without retrieval, answer it directly.
The documents you receive may be lemmatized and have stopwords removed, so you should interpret them smartly and reconstruct meaning where possible.
Always aim to be helpful, factual, and clear.
"""

def answer(
    user_prompt,
    history,
    api_key=API_KEY,
    metadata_store=METADATA_STORE,
    index=INDEX,
    k_retrieve=100,
    min_hits=5,
    max_hits=10,
    dynamic_k_threshold=0.8,
    model="gpt-4o",
    rerank_model="ms-marco-MiniLM-L-12-v2",
):
    # Define tool for database search
    # The LLM will internally decide:
    # 1. Whether to invoke the tool (do RAG)
    # 2. What the search query should be
    tools = [
        {
            "type": "function",
            "function": {
                "name": "do_rag",
                "description": "Search a database to retrieve relevant documents. Generate a smart and concise search query from the user's question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The generated search query based on the user's question."
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        }
    ]
    # expected tool call object:
    # {
    #   "tool_calls": [
    #     {
    #       "name": "do_rag",
    #       "arguments": {
    #         "query": "lithium-ion battery safety concerns"
    #       }
    #     }
    #   ]
    # }
    # Step 1: Let LLM decide whether to invoke RAG
    response = client.chat.completions.create(
        model=model,
        messages=history + [{"role": "user", "content": user_prompt}],
        tools=tools,
        tool_choice="auto"
    )
    tool_calls = response.choices[0].message.tool_calls # get the first tool call
    if tool_calls:
        print("[LLM tool call]")
        print(tool_calls)
        # Step 2: LLM invoked `search_database` → Do RAG
        print("[LLM requested database search — executing RAG]")
        # Extract the search query from the tool call or, if missing, use the original user input.
        query = json.loads(tool_calls[0].function.arguments).get("query", user_prompt)
        print(f"[RAG query (LLM-generated)] {query}")
        # Perform retrieval
        top_k = retrieve_top_k_embedding(index=index, query_text=query, k=k_retrieve)
        # Load top chunks using metadata
        top_chunks = []
        for vector_id in top_k:
            metadata = metadata_store.get(str(vector_id), None)
            if metadata and "content" in metadata:
                top_chunks.append(metadata["content"])
        if not top_chunks:
            return "The search did not return any useful result."
        # Rerank results
        reranked_docs = [
            x for x, _ in rerank_documents(
                query=query,
                chunks=top_chunks,
                max_hits=max_hits,
                dynamic_k_threshold=dynamic_k_threshold,
                min_hits=min_hits,
                rerank_model=rerank_model
            )
        ]
        if not reranked_docs:
            return "The search did not return any useful results."
        user_prompt = (
            f"[QUESTION]\n{user_prompt}\n\n"
            f"Use ONLY the following documents to answer the question:\n\n"
            +
            "\n\n".join(
                [f"Document {i+1}: {doc}" for i, doc in enumerate(reranked_docs)]
            )
        )
        print(f"[RAG user prompt] {user_prompt}")
    else:
        # LLM decided no search was needed — just answer directly
        print("[LLM did not request knowledge base search — direct response]")
    # Step 3: Generate final answer
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": INSTRUCTIONS}] + history + [{"role": "user", "content": user_prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # api_key = os.getenv("OPENAI_API_KEY")

    # query_text = "what does a spectrum engineer provide?"

    # folder_path = "/home/cerrion/DATATHON/data/normalized_data"

    # Load the index and metadata store from disK

    # print(answer_query(index, query_text, api_key))
    pass