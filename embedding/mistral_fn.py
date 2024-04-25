from typing import List, Optional
from tqdm import tqdm
import time
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('stabilityai/StableBeluga2', cache_fir = './', use_fast = False)
from setup_api import mistral_api

class MistralEmbedding():
    def __init__(self, model_name=None, batch_size=64):
        if model_name == 'mistral':
            self.model_name = "mistral-embed"
            
        self.mistral = mistral_api()
        
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
                    batch_res = self.mistral.embeddings(model=self.model_name, input=batch_chunk)
                    success = True
                except:
                    time.sleep(10)

                if success:
                    break

            embeddings = [tem.embedding for tem in batch_res.data]
            res_list.extend(embeddings)

        return res_list
