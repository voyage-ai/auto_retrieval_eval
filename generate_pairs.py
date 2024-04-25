import argparse
import random
import numpy as np
from paths import *
from utils import *
from embedding.embedding_fn import generate_embedding

parser = argparse.ArgumentParser(description="Generate embeddings and select top-k pairs")
parser.add_argument("--task_type", type=str, default="", choices=['generate_pairs', 'merge_pairs'])
args = parser.parse_args()

def read_queries_docs(data_path):
    queries = read_json(f'{data_path}/query.jsonl')
    corpus = read_json(f'{data_path}/corpus.jsonl')
    print('there are %d queries' % len(queries))
    print('there are %d docs' % len(corpus))
    return queries, corpus

def generate_pairs(data_path, embedding_models, embedding_path, retrieval_path, topk):
    queries, corpus = read_queries_docs(data_path)
    
    queries_id = [k for k,v in queries.items()]
    queries_list = [v for k,v in queries.items()]
    documents_id = [k for k,v in corpus.items()]
    documents = [v['text'] for k,v in corpus.items()]
    
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
    if os.path.exists(merged_retrieval_data_path):
        print('---------------------------------------------------------------')
        print('load merged pairs from %s' % merged_retrieval_data_path)
    else:
        print('---------------------------------------------------------------')
        print('save merged pairs to %s' % merged_retrieval_data_path)
        pairs_new = []
        for embedding_model_name in embedding_models:
            model_retrieval_path = f'{retrieval_path}/pairs_{embedding_model_name}_top{topk}.jsonl'
            pairs = read_json_lines(model_retrieval_path)
            for pair in pairs:
                if pair not in pairs_new:
                    pairs_new.append(pair)
                else:
                    repeated_pairs = [tem for tem in pairs_new if pair['query']==tem['query'] and pair['doc']==tem['doc']]
                    assert len(repeated_pairs)==1
        print('there are %d pairs' % len(pairs_new))

        random.shuffle(pairs_new) ## shuffle pairs
        
        pairs_new_query_dict = {}
        for pair in pairs_new:
            if pair['query_id'] not in pairs_new_query_dict:
                pairs_new_query_dict[pair['query_id']] = [pair]
            else:
                pairs_new_query_dict[pair['query_id']].append(pair)
        print('there are %d queries' % len(pairs_new_query_dict))

        pair_test = []
        for query_id, _ in pairs_new_query_dict.items():
            pair_test += [pair for pair in pairs_new if pair['query_id']==query_id]

        save_json_lines(merged_retrieval_data_path, pair_test)
        

def main():
    if args.task_type == 'generate_pairs':
        generate_pairs(data_path, embedding_models, embedding_path, retrieval_path, topk)
    elif args.task_type == 'merge_pairs':    
        merge_pairs(embedding_models, retrieval_path, merged_retrieval_data_path, topk)
            
if __name__ == "__main__":
    main()

