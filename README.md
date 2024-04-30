# Auto Labelling for Text Retrieval Evaluation
This repository offers a codebase for generating query-document pairs to assess text retrieval performance. It processes a collection of queries and documents where queries may consist of questions, titles, abstracts, or summaries, and the documents are likely to hold pertinent details concerning these queries. The provided code facilitates the creation of a dataset labeled with GPT-4, specifically designed for evaluating text retrieval capabilities.

## Installation

We recommend using Conda for the installation.

```bash
conda create -n auto_labeling_env python=3.10
conda activate auto_labeling_env
pip install -r requirements.txt
```

## Prepare data
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

## API Key Configuration

To configure API keys using environment variables, please store them in a `.env` file located in the root directory of your project.

## Generate query-document pairs

To generate paired labels, first run the embedding models and select the top-k documents for each query from each embedding model. Generated files are saved under the folder `./data/task-name/meta_data`.

```bash
python generate_pairs.py --task-name example_task --embedding-models voyage-large-2,text-embedding-3-large --topk 20
```

## Label query-document pairs using GPT4

For each query-document pair, use GPT-4 to determine if they are a good match. The document and query are assessed as a pair based on criteria that are divided into four levels: reject (label 1), borderline reject (label 2), borderline accept (label 3), and accept (label 4). We will elect valid pairs (those labeled as 4) and build query-document datasets for text retrieval evaluation. The final pairs are saved in `./data/task-name/pairs.jsonl`.

```bash
python label_pairs.py --task-name example_task --topk 20 --generative-model gpt-4-0125-preview
```