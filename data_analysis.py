import json
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

folder_path = "../data/hackathon_data"
files_in_folder = os.listdir(folder_path)
print(f"Found {len(files_in_folder)} files in folder: ")


def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []


for filename in files_in_folder:
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        page = load_documents(file_path)
        break

print("Page keys", page.keys())

num_subpages_per_page = []
char_counts_per_subpage = []

long_subpages = []  # Pages longer than 100k characters
samples_subpages = []  # Random sample of pages 0 < len <= 5000

LONG_TEXT_THRESHOLD = 100_000
RANDOM_SAMPLE_THRESHOLD = 5_000
MAX_SAVED = 10

random_subpage_candidates = []

for filename in tqdm(files_in_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        page = load_documents(file_path)

        subpages = page.get(
            "text_by_page_url", {}
        )  # Get all the subpages in a dictionary of the form {url: text}

        num_subpages = len(subpages)  # Number of subpages
        num_subpages_per_page.append(num_subpages)

        for subpage_url, text in subpages.items():
            length = len(text)  # Length of subpage text
            char_counts_per_subpage.append(length)

            # Save long texts
            if length > LONG_TEXT_THRESHOLD and len(long_subpages) < MAX_SAVED:
                long_subpages.append(
                    {
                        "source_file": filename,
                        "page_url": subpage_url,
                        "char_length": length,
                        "text": text[:1000],
                    }
                )

            # Collect candidates for random sampling
            if length <= RANDOM_SAMPLE_THRESHOLD:
                random_subpage_candidates.append(
                    {
                        "source_file": filename,
                        "page_url": subpage_url,
                        "char_length": length,
                        "text": text[:1000],
                    }
                )

# Sample randomly from eligible candidates
samples_subpages = random.sample(
    random_subpage_candidates, min(len(random_subpage_candidates), MAX_SAVED)
)

# Output paths
base_output_path = os.path.abspath(os.path.join(folder_path, ".."))
long_subpages_output_path = os.path.join(base_output_path, "subpages_over_100k.json")
short_subpages_output_path = os.path.join(base_output_path, "subpages_empty.json")
random_subpages_output_path = os.path.join(
    base_output_path, "random_subpages_under_5k.json"
)

# Save long subpages
with open(long_subpages_output_path, "w", encoding="utf-8") as f:
    json.dump(long_subpages, f, ensure_ascii=False, indent=2)

# Save random subpages
with open(random_subpages_output_path, "w", encoding="utf-8") as f:
    json.dump(samples_subpages, f, ensure_ascii=False, indent=2)

char_counts_per_subpage_np = np.array(char_counts_per_subpage)
log_lengths = np.log10(char_counts_per_subpage_np + 1)

# Final output
print("Summary Statistics:")
print(f"Mean number of subpages per page: {np.array(num_subpages_per_page).mean()}")
print(f"Total number of text blocks processed: {len(char_counts_per_subpage)}")
print("Mean character count:", np.mean(char_counts_per_subpage_np))
print(f"Example character counts per text block: {char_counts_per_subpage[:10]}")
print(
    f"Saved {len(long_subpages)} long subpages blocks to: {long_subpages_output_path}"
)
print(
    f"Saved {len(samples_subpages)} random subpage blocks (0 < len <= 5k) to: {random_subpages_output_path}"
)

# Plotting the distribution of subpage lengths and number of subpages per page
plt.figure(figsize=(10, 5))
plt.hist(log_lengths, bins=100, color="skyblue", edgecolor="black")
plt.xlabel("log10(Subpage length in characters)")
plt.ylabel("Frequency")
plt.title("Distribution of Subpage Lengths (Character Count, log-scale)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, "subpage_length_distribution.png"))

items_array = np.array(num_subpages_per_page)
log_items = np.log10(items_array + 1)

plt.figure(figsize=(10, 5))
plt.hist(log_items, bins=50, color="salmon", edgecolor="black")
plt.xlabel("log10(Number of subpages per page)")
plt.ylabel("Frequency")
plt.title("Distribution of subpages per page (log-scale)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, "subpages_per_page_distribution.png"))


# now filter out the subpages that are not relevant
num_subpages_per_page_filtered = []
char_counts_per_subpage_filtered = []

for filename in tqdm(files_in_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        page = load_documents(file_path)

        subpages = page.get(
            "text_by_page_url", {}
        )  # Get all the subpages in a dictionary of the form {url: text}

        count = 0
        for subpage_url, text in subpages.items():
            if re.search(r"([^A-Za-z0-9])css", subpage_url) or re.search(
                r"([^A-Za-z0-9])json", subpage_url
            ):
                continue
            elif text == "":
                continue

            length = len(text)  # Length of subpage text
            char_counts_per_subpage_filtered.append(length)
            count += 1
        num_subpages_per_page_filtered.append(count)


char_counts_per_subpage_filtered_np = np.array(char_counts_per_subpage_filtered)
log_lengths_filtered = np.log10(char_counts_per_subpage_filtered_np + 1)

# Final output
print("Summary Statistics:")
print(
    "Mean character count after filtering:",
    np.mean(char_counts_per_subpage_filtered_np),
)
print(
    f"Mean number of subpages per page after filtering: {np.array(num_subpages_per_page_filtered).mean()}"
)
print(f"Total number of text blocks processed: {len(char_counts_per_subpage_filtered)}")


plt.figure(figsize=(10, 5))
plt.hist(log_lengths_filtered, bins=100, color="skyblue", edgecolor="black")
plt.xlabel("log10(Text length in characters)")
plt.ylabel("Frequency")
plt.title("Distribution of Text Lengths (Character Count, log-scale)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_output_path, "text_length_distribution_filtered.png"))


items_array_filtered = np.array(num_subpages_per_page_filtered)
log_items = np.log10(items_array_filtered + 1)

plt.figure(figsize=(10, 5))
plt.hist(log_items, bins=50, color="salmon", edgecolor="black")
plt.xlabel("log10(Number of pages per document)")
plt.ylabel("Frequency")
plt.title("Distribution of Pages per Document (log-scale)")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(base_output_path, "pages_per_document_distribution_filtered.png")
)
