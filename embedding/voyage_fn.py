from typing import List
from tqdm import tqdm
import time
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('stabilityai/StableBeluga2', cache_fir = './', use_fast = False)
from setup_api import voyage_api

class VoyageEmbedding():
    def __init__(self, model_name='voyage-02', batch_size=64):
        self.model_name = model_name
        self.client = voyage_api()
        self.batch_size = batch_size
        self.max_token_len = 4096

    def embed_documents(self, texts: List[str], input_type = None):
        batch_list = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        res_list = []
        for batch in tqdm(batch_list):
            batch_chunk = []
            for j, txt in enumerate(batch):
                tokens = tokenizer.encode(txt)
                token_len = len(tokens)
                if token_len > self.max_token_len:
                    tokens = tokens[:self.max_token_len]
                    txt = tokenizer.decode(tokens)
                batch_chunk.append(txt)
            
            success = False
            while 1:
                try:
                    batch_res = self.client.embed(batch_chunk, model=self.model_name, input_type=input_type, truncation=True)
                    success = True
                except:
                    time.sleep(10)

                if success:
                    break
                    
            embeddings = batch_res.embeddings
            res_list.extend(embeddings)
        return res_list
