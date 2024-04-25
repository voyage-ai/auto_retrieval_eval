import os

task_name = 'legal_summarization_release'
embedding_models = ['voyage-2', 'voyage-code-2', 'text-embedding-3-small', 'text-embedding-3-large', 'embed-english-v3.0', 'mistral', 'googlecloud_textembedding-gecko@latest']
topk = 20
generative_model = 'gpt-4-0125-preview'

api_key_path = 'api_keys'
data_path = f'data/{task_name}'
meta_data_path = f'{data_path}/meta_data'
embedding_path = f'{meta_data_path}/embeddings'
retrieval_path = f'{meta_data_path}/retrieval'
merged_retrieval_path = f'{meta_data_path}/retrieval_merge'
filtered_pair_path = f'{meta_data_path}/pairs'

os.makedirs(embedding_path, exist_ok=True)
os.makedirs(retrieval_path, exist_ok=True)
os.makedirs(merged_retrieval_path, exist_ok=True)
os.makedirs(filtered_pair_path, exist_ok=True)

merged_retrieval_data_path = f'{merged_retrieval_path}/pairs_{len(embedding_models)}embedding_models_top{topk}.jsonl'
merged_pair_labels_path = f'{merged_retrieval_path}/pairs_{len(embedding_models)}embedding_models_top{topk}_gpt4.jsonl'