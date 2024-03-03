import sys
sys.path.append("../")

import torch
from torch.nn import functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from pinecone import Pinecone
from tqdm import tqdm
#from embedding_models import EmbeddingModel

class DataBase(ABC):
    """
    abstract base class for our vector databases. 
    classes that inherit from this only need to implement retrieve_by_query so that they are 
    drop in replacements for each other
    """
    @abstractmethod
    def retrieve_by_query(self, query: str, top: int=5) -> List[Dict]:
        pass

class NumpyDataBase(DataBase):
    """
    Numpy database that implements cosine similarity search by hand (assuming that the embedding vectors are normalized)
    On this scale (~20000 vectors) numpy is consistently quicker than cloud databases such as pinecone
    and not hard to implement.
    """
    def __init__(self, data: List[Dict], embedding_model):
        for sample in data:
            assert len(sample['embedding']) == len(data[0]['embedding']), "embeddings don't all have the same dimension"
        # stack all embedding vectors for easy cosine similarity calculation
        self.embeddings = np.stack([s['embedding'] for s in data])
        # store metadata of vectors in its own array for easy access
        self.metadata = [s['metadata'] for s in data]
        # embedding model used to embed user query
        self.embedding_model = embedding_model
        
    def retrieve_by_query(self, query: str, top: int=5) -> List[Dict]:
        # embed user query, normalize the vector and cast to numpy for cosine similartiy calculation
        embedded_query = np.array(self.embedding_model.embed_query(query, normalize=True))
        # since the vectors are normalized, the dot product already represents the cosine sim on it's own
        cosines = self.embeddings @ embedded_query
        # argpartition is like a sort based on argmax
        indices = cosines.argpartition(-top)[-top:]
        # find the metadata of the top indices and return it
        return [self.metadata[i] for i in indices] 


class PineconeDataBase(DataBase):
    """
    Pinecone database is slower and less convenient for single user use than numpy, but 
    here we use it so that not everybody has to run embed_and_store on their machine.
    Other limitations:
    + Pinecone doesn't allow arbitrary sizes for metadata (we ran into that issue because some abstracts are too long), numpy doesn't have this limitation
    
    """
    def __init__(self, embedding_model): 
        # embedding model used to embed user query
        self.embedding_model = embedding_model
        # pinecone instance with hardcoded api key for convenience
        pc = Pinecone(api_key="5268fcc9-a63a-4f19-b603-6ec31b79ed10")
        # the pinecone index we need access to 
        self.index = pc.Index("pubmedabstracts")
        
    def run_setup(self, data: List[Dict]) -> None:
        """
        This function is supposed to be called only once with the data.json that is produced by
        embed_and_store.py. It will populate the index and set everything up.
        """
        for idx, sample in enumerate(tqdm(data)):
            #print(sample['metadata'])
            self.index.upsert(
                vectors=[{
                        "id": f"vec:{idx}", 
                        "values": sample['embedding'], 
                        "metadata":  {
                            # truncating abstracts because pinecone doesn't allow arbitrary sizes for metadata
                            "abstract": sample['metadata']["abstract"][:15000] if sample['metadata']["abstract"] is not None else "",
                            "title": sample['metadata']['title'] if sample['metadata']['title'] is not None else "",
                            "article_id": sample['metadata']['article_id'][0][0] if sample['metadata']['article_id'][0][0] is not None else ""
                        }
                }],
                namespace= "default"
            )

    def retrieve_by_query(self, query: str, top: int=5) -> List[Dict]:
        embedded_query = self.embedding_model.embed_query(query, normalize=True)
        results = self.index.query(
            namespace="default",
            vector=embedded_query,
            top_k=top,
            include_values=True,
            include_metadata=True
        )['matches']
        return [{
            "abstract": r['metadata']['abstract'],
            "title": r['metadata']['title'],
            "article_id": [[r['metadata']['article_id']]]
        } for r in results]
    


class RedisDataBase(DataBase):
    def __init__(self):
        raise NotImplementedError
    @staticmethod
    def run_setup(database_url: str, data: List[Dict]) -> None:
        pass

