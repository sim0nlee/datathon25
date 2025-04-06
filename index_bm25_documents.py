import hashlib
import json
import os

from tqdm import tqdm
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import NUMERIC, STORED, TEXT, Schema
from whoosh.index import create_in


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


def create_whoosh_index(index_dir, schema):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    return create_in(index_dir, schema)


if __name__ == "__main__":
    # Set the window size for text chunking
    WINDOW_SIZE = 2000

    # Folder containing the JSON documents
    input_folder = "/home/cerrion/DATATHON/data/normalized_data"
    files_in_folder = os.listdir(input_folder)
    print(f"Found {len(files_in_folder)} files in {input_folder}")

    # Define the Whoosh schema
    # The 'content' field is analyzed using the StandardAnalyzer and will be searched using BM25 ranking
    schema = Schema(
        id=NUMERIC(stored=True, unique=True, bits=64),
        filename=STORED,
        page_url=STORED,
        chunk_index=NUMERIC(stored=True),
        content=TEXT(stored=True, analyzer=StandardAnalyzer()),
        # Store additional metadata (converted to JSON) if needed
        metadata=STORED,
    )

    index_dir = "whoosh_index"
    ix = create_whoosh_index(index_dir, schema)
    writer = ix.writer()

    for filename in tqdm(files_in_folder, desc="Indexing files"):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(input_folder, filename)
        doc = load_documents(file_path)

        text_by_page = doc.get("text_by_page_url", {})
        # Retain any other metadata from the document (except the text_by_page_url)
        doc_metadata = {k: v for k, v in doc.items() if k != "text_by_page_url"}

        for page_url, text in text_by_page.items():
            # Chunk the page text
            chunks = chunk_text(text, WINDOW_SIZE)
            if not chunks:
                continue

            for i, chunk in enumerate(chunks):
                doc_id = generate_id(filename, page_url, i)
                writer.add_document(
                    id=doc_id,
                    filename=filename,
                    page_url=page_url,
                    chunk_index=i,
                    content=chunk,
                    metadata=json.dumps(doc_metadata),
                )

    writer.commit()
    print("Indexing complete. You can now perform BM25 search queries using Whoosh.")
