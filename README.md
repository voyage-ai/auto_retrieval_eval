# Automatic Construction of Text Retrieval Evaluation Set
Given a collection of unpaired queries and documents, this repository offers a simple and cost-efficient method for constructing an evaluation dataset for retrieval quality. It automatically labels the gold document for each query using GPT-4. Instead of employing GPT-4 to review the entire corpus for each query—which incurs costs linear to the size of the corpus and is often prohibitive—we utilize a variety of embedding models to pre-filter the corpus. This creates a smaller set of candidate documents, within which GPT-4 identifies the gold documents.

For a corpus of size $n$, the original approach would require $n$ GPT calls to label a single query. Our method reduces this number to approximately 100 calls. In typical scenarios, the cost of labeling each query is thus reduced to less than $1.

## Installation

We recommend using Conda for the installation.

```bash
conda create -n auto_retrieval_eval_env python=3.10
conda activate auto_retrieval_eval_env
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

We need to invoke APIs for embedding models and generative language models. To configure the API keys using environment variables, please store them in `.env` file located in the root directory of your project.

## Prefilter query-document pairs

For each query, use a set of embedding models to create a set of pre-filtered candidate documents. Generated candidates are saved under the folder `./data/task-name/meta_data`.

```bash
python prefilter_pairs.py --task-name example_task --embedding-models voyage-large-2,text-embedding-3-large --topk 20
```

## Label prefiltered pairs using GPT4

For each prefiltered pair, use GPT-4 to determine if they constitute a relevant match. The document and query are assessed as a pair based on criteria that are divided into four levels: reject (label 1), borderline reject (label 2), borderline accept (label 3), and accept (label 4). We will elect valid pairs (those labeled as 4) and build query-document datasets for text retrieval evaluation. The final pairs are saved in `./data/task-name/pairs.jsonl`.

```bash
python label_pairs.py --task-name example_task --topk 20 --generative-model gpt-4-0125-preview
```