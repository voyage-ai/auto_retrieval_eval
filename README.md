# Auto Labelling 

### Prepare data
Place queries and documents in the `data/{task_name}` folder:

`query.jsonl` contains the query id and query text.

`corpus.jsonl` contains the document id and document text.

### Setup configurations

Setup configurations in `paths.py`.

Save your API keys in the `api_key` folder.

### Generate pairs

Run embedding models and select top-k documents for each query:

```python generate_pairs.py --task_type generate_pairs```

Merge the top-k documents from different embedding models:

```python generate_pairs.py --task_type merge_pairs```

### Label pairs using GPT4

Label pairs using GPT4:

```python main.py --task_type generate_score```

Parse GPT4 output:

```python main.py --task_type parse_score```

Select valid pairs and build datasets:

```python main.py --task_type build_dataset```