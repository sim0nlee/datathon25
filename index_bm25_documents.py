import json
import os

from tqdm import tqdm
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import NUMERIC, STORED, TEXT, Schema
from whoosh.index import create_in


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
    """Chunks the text into smaller pieces of a specified window size with overlapping of 1/4 the window."""
    chunks = []
    step = int(window_size * 3 / 4)
    for i in range(0, len(text), step):
        chunk = text[i : i + window_size]
        chunks.append(chunk)
    return chunks[:6]


def create_whoosh_index(index_dir, schema):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    return create_in(index_dir, schema)


if __name__ == "__main__":
    # Set the window size for text chunking
    WINDOW_SIZE = 4000

    # Folder containing the JSON documents
    input_folder = "/home/orderfox/DATATHON/data/normalized_data"
    files_in_folder = os.listdir(input_folder)
    print(f"Found {len(files_in_folder)} files in {input_folder}")

    # Define the Whoosh schema
    # The 'content' field is analyzed using the StandardAnalyzer and will be searched using BM25 ranking
    schema = Schema(
        json_path=STORED,
        url=STORED,
        page_url=STORED,
        chunk_index=NUMERIC(stored=True),
        content=TEXT(stored=True, analyzer=StandardAnalyzer()),
    )

    index_dir = "whoosh_index_4000_first6"
    ix = create_whoosh_index(index_dir, schema)
    writer = ix.writer()

    for filename in tqdm(files_in_folder, desc="Indexing files"):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(input_folder, filename)
        doc = load_documents(file_path)

        text_by_page = doc.get("text_by_page_url", {})
        url = doc.get("url", "")

        for page_url, text in text_by_page.items():
            # Chunk the page text
            chunks = chunk_text(text, WINDOW_SIZE)
            if not chunks:
                continue

            for i, chunk in enumerate(chunks):
                writer.add_document(
                    json_path=filename,
                    url=url,
                    page_url=page_url,
                    chunk_index=i,
                    content=chunk,
                )

    writer.commit()
    print("Indexing complete. You can now perform BM25 search queries using Whoosh.")
