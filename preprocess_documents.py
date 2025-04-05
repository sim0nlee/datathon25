import json
import multiprocessing as mp
import os

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tqdm import tqdm

nltk.download("stopwords")
nltk.download("wordnet")


def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []


def normalize_text(text: str):
    """Input doc and return clean list of tokens for BM25 indexing"""

    if len(text) > 1000000:  # Skip long texts due to nltk limitations
        return text
    doc = understandable_languages(text.lower())  # Tokenize and normalize case
    tokens = []
    for token in doc:
        if token.is_alpha and token.text.isascii():  # keep only alphabetic tokens
            word = token.text
            if word not in stoplist:
                lemma = wnl.lemmatize(word)
                tokens.append(lemma)
    return " ".join(tokens)


def process_file(filename):
    """Process a single JSON file: normalize each page's text and save to disk."""

    # Only process JSON files
    if not filename.endswith(".json"):
        return

    file_path = os.path.join(folder_path, filename)
    doc = load_documents(file_path)

    # Create a new document that copies all original keys
    output_doc = {}
    for key in doc:
        if key != "text_by_page_url":
            output_doc[key] = doc[key]

    # Normalize the text for pages in text_by_page_url
    text_by_page = doc.get("text_by_page_url", {})
    normalized_pages = {}
    for page_url, text in text_by_page.items():
        # Skip non-content entries
        if "css" in text or "json" in text:
            continue
        normalized_text = normalize_text(text)
        normalized_pages[page_url] = normalized_text

    # Replace the original text_by_page_url with the normalized content
    output_doc["text_by_page_url"] = normalized_pages

    output_path = os.path.join(output_folder, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, ensure_ascii=False, indent=2)

    return output_path


if __name__ == "__main__":
    # Initialize NLTK resources
    # !python -m spacy download xx_ent_wiki_sm
    stoplist = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    understandable_languages = spacy.load("xx_ent_wiki_sm")

    folder_path = "/home/cerrion/DATATHON/data/hackathon_data"
    output_folder = "/home/cerrion/DATATHON/data/normalized_data"
    os.makedirs(output_folder, exist_ok=True)

    files_in_folder = os.listdir(folder_path)
    print(f"Found {len(files_in_folder)} files in {folder_path}")

    # Example usage of normalize_text function
    print(
        normalize_text(
            "This is valid. <div> var \nx = 5; </div> Je suis content. <code>.css bella Italia"
        )
    )

    # Use multiprocessing Pool to process files in parallel
    with mp.Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(process_file, files_in_folder),
            total=len(files_in_folder),
        ):
            pass
