import torch
from torch.nn import functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from embedding_models import EmbeddingModel

class DataBase(ABC):
    @abstractmethod
    def retrieve_by_query(self, query: str, top: int=5) -> List[str]:
        pass

class NumpyDataBase(DataBase):
    def __init__(self, data: List[Dict], embedding_model: EmbeddingModel):
        for sample in data:
            assert len(sample['embedding']) == len(data[0]['embedding']), "embeddings don't all have the same dimension"

        self.embeddings = np.stack([s['embedding'] for s in data])
        self.metadata = [s['metadata'] for s in data]
        self.embedding_model = embedding_model
        
    def retrieve_by_query(self, query: str, top: int=5) -> List[str]:
        embedded_query = np.array(self.embedding_model.embed_query(query, normalize=True))
        cosines = self.embeddings @ embedded_query
        indices = cosines.argpartition(-top)[-top:]
        return [self.metadata[i] for i in indices] 


class RedisDataBase(DataBase):
    def __init__(self):
        raise NotImplementedError
    @staticmethod
    def run_setup(database_url: str, data: List[Dict]) -> None:
        pass

class PineconeDataBase(DataBase):
    def __init__(self):
        raise NotImplementedError
    @staticmethod
    def run_setup(pinecone_api_key: str, database_id: str, data: List[Dict]) -> None:
        pass



