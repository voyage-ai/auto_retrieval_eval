from typing import List
from tqdm import tqdm
import pdb
import tiktoken
from setup_api import openai_api

class OpenAIEmbedding():
    def __init__(self, model_name=None, batch_size=64):
        self.model_name = model_name
        self.client = openai_api()

        if self.model_name=='text-embedding-ada-002':
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        else:
            self.tokenizer = tiktoken.get_encoding('cl100k_base')

        self.batch_size = batch_size
        self.max_token_len = 8191

    def embed_documents(self, texts: List[str], input_type = None, decode = True):
        batch_list = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        res_list = []
        for batch in tqdm(batch_list):
            all_tokens = []
            used_indices = []
            for j, txt in enumerate(batch):
                if not(txt):
                    print("Detected empty item, which is not allowed by the OpenAI API - Replacing with empty space")
                    txt = " "

                tokens = self.tokenizer.encode(txt, disallowed_special=())
                token_len = len(tokens)
                if token_len > self.max_token_len:
                    tokens = tokens[:self.max_token_len]
                # For some characters the API raises weird errors, e.g. input=[[126]]
                if decode:
                    tokens = self.tokenizer.decode(tokens)
                all_tokens.append(tokens)
                used_indices.append(j)

            out = [[]] * len(batch)
            if all_tokens:
                try:
                    response = self.client.embeddings.create(input = all_tokens, model=self.model_name)
                except:
                    pdb.set_trace()

                assert len(response.data) == len(all_tokens), f"Sent {len(all_tokens)}, got {len(response.data)}"
                for data in response.data:
                    idx = data.index
                    # OpenAI seems to return them ordered, but to be save use the index and insert
                    idx = used_indices[idx]
                    embedding = data.embedding
                    # print(len(embedding))
                    out[idx] = embedding
            res_list.extend(out)
        
        return res_list
