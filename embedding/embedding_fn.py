import os
from utils import read_pickle, save_pickle
from embedding.voyage_fn import VoyageEmbedding
from embedding.openai_fn import OpenAIEmbedding
from embedding.cohere_fn import CohereEmbedding
from embedding.mistral_fn import MistralEmbedding
from embedding.google_fn import GoogleEmbedding

def generate_embedding(embedding_path, embedding_model_name, queries, documents):
    if os.path.exists(embedding_path):
        print('load embedding from %s' % embedding_path)
        embedding_result = read_pickle(embedding_path)
        query_embeddings = embedding_result['query_embeddings']
        doc_embeddings = embedding_result['doc_embeddings']
    else:
        print('generate embeddings and save them to %s' % embedding_path)
        if 'voyage' in embedding_path:
            embedding_model = VoyageEmbedding(embedding_model_name, batch_size=8)
            query_embeddings = embedding_model.embed_documents(queries, input_type='query')
            doc_embeddings = embedding_model.embed_documents(documents, input_type='document')
        elif 'text-embedding' in embedding_path:
            embedding_model = OpenAIEmbedding(embedding_model_name, batch_size=8)
            query_embeddings = embedding_model.embed_documents(queries)
            doc_embeddings = embedding_model.embed_documents(documents)
        elif 'embed-english-v3.0' in embedding_path:
            embedding_model = CohereEmbedding(embedding_model_name, batch_size=8)
            query_embeddings = embedding_model.embed_documents(queries, input_type="search_query")
            doc_embeddings = embedding_model.embed_documents(documents, input_type="search_document")
        elif 'mistral' in embedding_path:
            embedding_model = MistralEmbedding(embedding_model_name, batch_size=2)
            query_embeddings = embedding_model.embed_documents(queries)
            doc_embeddings = embedding_model.embed_documents(documents)
        elif 'google' in embedding_path:
            embedding_model_name = embedding_model_name.split('_')[1]
            embedding_model = GoogleEmbedding(embedding_model_name, batch_size=1)
            query_embeddings = embedding_model.embed_documents(queries)
            doc_embeddings = embedding_model.embed_documents(documents)
        else:
            print('no such embedding model')
            
        embedding_result = {'query_embeddings': query_embeddings, 'doc_embeddings': doc_embeddings}
        save_pickle(embedding_path, embedding_result)
    return query_embeddings, doc_embeddings
