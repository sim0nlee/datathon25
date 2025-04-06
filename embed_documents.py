import hashlib
import json
import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_id(filename, page_url, chunk_index):
    # Create a unique string from the metadata
    unique_str = f"{filename}-{page_url}-{chunk_index}"
    # Use MD5 to generate a hash and then take part of it to fit in 64 bits
    hash_digest = hashlib.md5(unique_str.encode("utf-8")).hexdigest()[:16]
    return int(hash_digest, 16) % (1 << 63)


def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []


def chunk_text(text, window_size):
    """Chunks the text into smaller pieces of a specified window size with overlapping of half the window."""
    chunks = []
    step = window_size // 2  # half the window size
    for i in range(0, len(text), step):
        chunk = text[i : i + window_size]
        chunks.append(chunk)
    return chunks


def insert_doc_in_index(filename, model, index, metadata_store):
    """Process a single normalized JSON file: for each web page, chunk
    and embed using sentence transformers. Then insert into the FAISS index."""

    # Only process JSON files
    if not filename.endswith(".json"):
        return

    file_path = os.path.join(input_folder, filename)
    doc = load_documents(file_path)

    text_by_page = doc.get("text_by_page_url", {})
    # copy over parents metadata
    # ['url', 'timestamp', 'text_by_page_url', 'doc_id']
    doc_metadata = {k: v for k, v in doc.items() if k != "text_by_page_url"}

    embeddings_list = []
    ids_list = []
    metadatas_list = []

    for page_url, text in text_by_page.items():
        # Chunk text and batch encode
        chunks = chunk_text(text, WINDOW_SIZE)
        # skip empty chunks after preprocessing due to removing of non-ascii
        if not chunks:
            continue
        embeddings = model.encode(
            chunks, show_progress_bar=False, device=device, normalize_embeddings=True
        )

        # Generate IDs and corresponding metadata for each chunk
        for i, chunk in enumerate(chunks):
            vector_id = generate_id(filename, page_url, i)
            ids_list.append(vector_id)
            metadata = {
                "json_path": filename,
                "text_by_page_url": page_url,
                "chunk_index": i,
                "content": chunk,
            }
            metadata.update(doc_metadata)
            metadatas_list.append(metadata)

        embeddings_list.append(embeddings)

    if embeddings_list:
        # Concatenate all embeddings from the document
        all_embeddings = np.vstack(embeddings_list)
        all_ids = np.array(ids_list, dtype="int64")
        # Insert in batch using add_with_ids
        index.add_with_ids(all_embeddings, all_ids)
        # Store metadata for later retrieval (convert IDs to string keys)
        for vector_id, metadata in zip(all_ids, metadatas_list):
            metadata_store[str(vector_id)] = metadata


if __name__ == "__main__":
    WINDOW_SIZE = 1000

    # use the clean data
    input_folder = "/home/cerrion/DATATHON/data/normalized_data"
    files_in_folder = os.listdir(input_folder)
    print(f"Found {len(files_in_folder)} files in {input_folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # external metadata_store
    metadata_store = {}

    # use relatively high links per vector, more accurate
    index = faiss.IndexHNSWFlat(
        model.get_sentence_embedding_dimension(), 64, faiss.METRIC_INNER_PRODUCT
    )
    # index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())

    # change the index parameters to be more accurate
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100

    # Wrap with an ID map to allow custom IDs
    index = faiss.IndexIDMap(index)

    # Process and index all JSON files in the input folder
    for filename in tqdm(files_in_folder, desc="Indexing files"):
        insert_doc_in_index(filename, model, index, metadata_store)

    # save index and metadata_store to disk
    faiss.write_index(index, "index_all-MiniLM-L6-v2.faiss")

    with open("metadata_store_all-MiniLM-L6-v2.json", "w") as f:
        json.dump(metadata_store, f)
