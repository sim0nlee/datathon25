# Designing a RAG Agent on Raw HTML Data 
## Team : Rage Against the Machine Learning

In this report we describe each step of our workflow in designing the agent.

### Data analysis

Before diving into any form of processing, we plotted the logarithms of the subpage lenghts in characters, noting that they follow a log-normal distribution with mean ~ 18316:

![alt text](subpage_length_distribution.png).

Based on this observation, we decided to examine the subpages with lengths above 100.000 (corresponding to ~3σ in the log-space), finding out that they mostly consist of CSS and JSON files with no information content:

```
{
    "source_file": "cqs.com.json",
    "page_url": "https://www.cqs.com/assets/gstring/main-v2.css?v=3",
    "char_length": 320380,
    "text": "/********** START CSS **********/\r\n/*colors*/\r\n/*margins and paddings*/\r\n/*border*/\r\n/*m........
}
```

We also identified many empty subpages, i.e., subpages of length 0. 


### Data preprocessing

1. Filtered out `.json` and `.css` files from the dataset
2. Removed non-unicode characters
3. Removed non-alfanumeric characters
4. Removed stopwords 
5. Lowercased 
6. Lemmatized 

Performing this pre-processing steps also greatly improves the size of the final dataset.

### Subpage Chunk Embedding

We used `sentence-transformers/all-MiniLM-L6-v2` to embed document chunks of size 1000 in vectors of dimension 384. There is a 50% overlap between chunks to ensure some form of semantic continuity.

With 22M parameters, the model is fast and memory-efficient, suitable for large-scale indexing with moderate accuracy. To ensure cosine similarity aligns with inner product similarity, we let the model normalize the embeddings.

### Indexing

We use Faiss to index our database with an`IndexHNSWFlat` wrapped in `IndexIDMap`. We decided to set `M=64` bi-directional links per node to favor accuracy.

- `efConstruction=200`: Construction time accuracy control.
    - Higher = better quality graph, slower to build.
- `efSearch=100`: Search-time trade-off.
    - Higher = more accurate results, slower queries.


### Retrieval

We combine the index queries with the `BM25` ranking function for 





