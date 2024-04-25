from typing import List, Optional
from tqdm import tqdm
import time

from vertexai.language_models import TextEmbeddingModel

from setup_api import google_vertex_api

class GoogleEmbedding():
    def __init__(self, model_name=None, batch_size=64):
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str], input_type = None):
        batch_list = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        res_list = []
        for batch in tqdm(batch_list):
            success = False
            while 1:
                try:
                    batch_res = self.model.get_embeddings(batch)
                    success = True
                except:
                    time.sleep(10)

                if success:
                    break

            embeddings = [tem.values for tem in batch_res]
            res_list.extend(embeddings)

        return res_list
