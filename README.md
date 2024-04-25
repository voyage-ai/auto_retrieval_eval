# Auto Labelling for Text Retrieval Evaluation
This GitHub repository provides code for constructing query-document pairs for text retrieval evaluation. Given a set of queries and documents, the queries can include questions, titles, abstracts, or summaries, while the documents may contain relevant information about the queries. This code helps in building a dataset with GPT-4 labeled query-document pairs for evaluating text retrieval.

## Preparing data
Store the queries and documents in the `data/{task_name}` directory:

`query.jsonl` should include the query id and query text.
```json
"00000": {"text": "This is the text of the first query."},
"00001": {"text": "This is the text of the second query."},
```

`corpus.jsonl` should contain the document id and document text.
```json
"000000000": {"text": "This is the text of the first document."},
"000000001": {"text": "This is the text of the second document."},
```

## Setup configurations

Configure settings in `paths.py`: 
```python
# Task name
task_name = 'legal_summarization_release'

# List of embedding models
embedding_models = ['voyage-2', 'voyage-code-2', 'text-embedding-3-small', 'text-embedding-3-large', 'embed-english-v3.0', 'mistral', 'googlecloud_textembedding-gecko@latest']

# Number of top retrieved documents selected for each query
topk = 20

# Model used for generating labels
generative_model = 'gpt-4-0125-preview'
```

Store your API keys in the `api_key` folder. 

Based on the embedding and generative models you are using, save the keys into the corresponding files.

    api_keys/voyage_key.txt
    api_keys/openai_key.txt
    api_keys/mistral_key.txt
    api_keys/cohere_key.txt
    api_keys/google_vertex.txt

## Generate query-document pairs

To generate paired labels, first run the embedding models and select the top-k documents for each query from each embedding model.

```bash
python generate_pairs.py --task_type generate_pairs
```

Combine the top-k documents from various embedding models:

```bash
python generate_pairs.py --task_type merge_pairs
```

## Labeling query-document pairs using GPT4

For each query-document pair, use GPT-4 to determine if they are a good match. The document and query are assessed as a pair based on criteria that are divided into four levels: reject (label 1), borderline reject (label 2), borderline accept (label 3), and accept (label 4).

```bash
python main.py --task_type generate_score
```

Extract labels ranging from 1 to 4, which represent reject, borderline reject, borderline accept, and accept, from the GPT-4 output.

```bash
python main.py --task_type parse_score
```

Select valid pairs (those labeled as 4) and build query-document datasets for text retrieval evaluation. The pairs are saved in the `data/{task_name}/pairs.jsonl` folder.

```bash
python main.py --task_type build_dataset
```