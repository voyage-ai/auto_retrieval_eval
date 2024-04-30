import random
import os
import numpy as np
from config import Config
from utils import parse_arguments, read_json, read_json_lines, save_json_lines, create_directories
from embedding.embedding_fn import generate_embedding

def generate_pairs(data_path, embedding_models, embedding_path, retrieval_path, topk):
    queries = read_json(f'{data_path}/query.jsonl')
    corpus = read_json(f'{data_path}/corpus.jsonl')
    queries_id = list(queries.keys())
    queries_list = [v['text'] for v in queries.values()]
    documents_id = list(corpus.keys())
    documents = [v['text'] for v in corpus.values()]
    
    for embedding_model_name in embedding_models:
        model_embedding_path = f'{embedding_path}/{embedding_model_name}_embedding.pickle'
        model_retrieval_path = f'{retrieval_path}/pairs_{embedding_model_name}_top{topk}.jsonl'

        query_embeddings, doc_embeddings = generate_embedding(model_embedding_path, embedding_model_name, queries_list, documents)
        scores = np.dot(np.array(query_embeddings), np.array(doc_embeddings).T)
        top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :topk]
        pairs_for_labelling = [[query_index, doc_index] for query_index in range(len(top_indices)) for doc_index in top_indices[query_index]]

        pairs = []
        for ind, pair_index in enumerate(pairs_for_labelling):
            query = queries_list[pair_index[0]]
            query_id = queries_id[pair_index[0]]
            
            doc = documents[pair_index[1]]
            doc_id = documents_id[pair_index[1]]
            pairs.append({'query_id':query_id, 'doc_id': doc_id, 'query':query, 'doc':doc})
            
        print('there are %d pairs' % len(pairs))
        if not os.path.exists(model_retrieval_path):
            print('save selected pairs to %s' % model_retrieval_path)
            save_json_lines(model_retrieval_path, pairs)

def merge_pairs(embedding_models, retrieval_path, merged_retrieval_data_path, topk):
    print('---------------------------------------------------------------')
    print(f'Starting merge of pairs, saving to {merged_retrieval_data_path}')

    unique_pairs = {}
    for embedding_model_name in embedding_models:
        model_retrieval_path = f'{retrieval_path}/pairs_{embedding_model_name}_top{topk}.jsonl'
        pairs = read_json_lines(model_retrieval_path)
        for pair in pairs:
            pair_key = (pair['query_id'], pair['doc_id'])  
            if pair_key not in unique_pairs:
                unique_pairs[pair_key] = pair

    pairs_list = list(unique_pairs.values())

    print(f'There are {len(pairs_list)} unique pairs')
    save_json_lines(merged_retrieval_data_path, pairs_list)
        

def main():
    args = parse_arguments()
    config = Config(args)
    create_directories(config)
    generate_pairs(config.data_path, config.embedding_models, config.embedding_path, config.retrieval_path, config.topk)
    merge_pairs(config.embedding_models, config.retrieval_path, config.merged_retrieval_data_path, config.topk)
            
if __name__ == "__main__":
    main()

