import torch
from torch.nn import functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_query(self, query: str, normalize=True) -> List[float]:
        pass
    @abstractmethod
    def embed_passage(self, passage, min_overlap=0.2, normalize=True) -> List[List[float]]:
        pass

class EmbeddingModelMiniLML6(EmbeddingModel):
    def __init__(self):
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.max_len = 512 - 2
        
    def embed_query(self, query: str, normalize=True) -> List[float]:
        inp = self.tokenizer(query, return_tensors='pt')
        embedding = self._mean_pooling(self.model(**inp), inp['attention_mask'])
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.reshape(-1).tolist()

    def embed_passage(self, passage, min_overlap=0.2, normalize=True) -> List[List[float]]:
        chunks = self._chunk_passage(passage, min_overlap)
        embedded_chunks = []
        for chunk in chunks:
            input_ids = torch.tensor(chunk).unsqueeze(0)
            token_type_ids = torch.zeros_like(input_ids)
            attention_mask = torch.ones_like(input_ids)
            inp = {
                "input_ids":      input_ids, 
                "token_type_ids": token_type_ids, 
                "attention_mask": attention_mask
            }
            with torch.no_grad():
                output = self.model(**inp)
            embedding = self._mean_pooling(output, inp['attention_mask'])
            if normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
            embedded_chunks.append(embedding.reshape(-1).tolist())
        return embedded_chunks

    def _mean_pooling(self, model_output, attention_mask) -> torch.tensor:
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _chunk_passage(self, passage, min_overlap) -> List[List[float]]:
        tokenized_passage = self.tokenizer(passage)['input_ids'][1:-1] #slicing cuts of [CLS] and [SEP] token at beginning and end
        if len(tokenized_passage) > self.max_len: 
            chunks = []
            start = 0
            while True:
                chunk = [101] + tokenized_passage[start:start+self.max_len] + [102]
                chunks.append(chunk)
                if (start + self.max_len) > len(tokenized_passage):
                    break
                start += (self.max_len - int(self.max_len * min_overlap))
        else: 
            chunks = [[101] + tokenized_passage + [102]]
        return chunks
