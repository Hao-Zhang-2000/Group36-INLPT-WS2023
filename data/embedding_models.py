import torch
from torch.nn import functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from transformers import AutoModel, AutoTokenizer

class EmbeddingModel(ABC):
    """
    Abstract base class for the embedding models we use. Classes that inherit from this need to implement
    embed_query, which directly embeds a users query without any preprocessing and embed_passage, which does a few preprocessing steps, most notable chunking.
    """
    @abstractmethod
    def embed_query(self, query: str, normalize=True) -> List[float]:
        pass
    @abstractmethod
    def embed_passage(self, passage, min_overlap=0.2, normalize=True) -> List[List[float]]:
        pass

class EmbeddingModelMiniLML6(EmbeddingModel):
    def __init__(self):
        # model we use for embedding queries and passages
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # tokenizer for the embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # max length of the embedding model, -2 because of the need for [CLS] and [SEP] tokens (Bert based model)
        self.max_len = 512 - 2
        
    def embed_query(self, query: str, normalize=True) -> List[float]:
        # tokenize query 
        inp = self.tokenizer(query, return_tensors='pt')
        # embed tokenized query 
        embedding = self._mean_pooling(self.model(**inp), inp['attention_mask'])
        # normalize query
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        # cast embedding to list and return
        return embedding.reshape(-1).tolist()

    def embed_passage(self, passage, min_overlap=0.2, normalize=True) -> List[List[float]]:
        # split passage into tokenized chunks
        chunks = self._chunk_passage(passage, min_overlap)
        embedded_chunks = []
        for chunk in chunks:
            # create input to embedding model
            input_ids = torch.tensor(chunk).unsqueeze(0)
            token_type_ids = torch.zeros_like(input_ids)
            attention_mask = torch.ones_like(input_ids)
            inp = {
                "input_ids":      input_ids, 
                "token_type_ids": token_type_ids, 
                "attention_mask": attention_mask
            }
            # run embedding model and pool outputs 
            with torch.no_grad():
                output = self.model(**inp)
            embedding = self._mean_pooling(output, inp['attention_mask'])
            # normalize embedding
            if normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
            embedded_chunks.append(embedding.reshape(-1).tolist())
        return embedded_chunks

    def _mean_pooling(self, model_output, attention_mask) -> torch.tensor:
        """
        This function is directly taken from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        It's needed for this specific embedding model to produce relevant embeddings
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _chunk_passage(self, passage, min_overlap) -> List[List[float]]:
        """
        This function takes a passage and turns it into tokenized chunks with a max length of self.model.max_length + 2
        """
        # tokenize passage
        tokenized_passage = self.tokenizer(passage)['input_ids'][1:-1] #slicing cuts of [CLS] and [SEP] token at beginning and end
        if len(tokenized_passage) > self.max_len:
            chunks = []
            start = 0
            while True:
                # form a chunk from start to start + self.max_len and add [CLS] at the start and [SEP] at the end
                chunk = [101] + tokenized_passage[start:start+self.max_len] + [102]
                chunks.append(chunk)
                if (start + self.max_len) > len(tokenized_passage):
                    break
                # move start forward such that the minimum amount of overlap is obeyed
                start += (self.max_len - int(self.max_len * min_overlap))
        else: 
            chunks = [[101] + tokenized_passage + [102]]
        return chunks
