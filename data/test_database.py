from tqdm import tqdm
import numpy as np
import json
import torch
from embedding_models import EmbeddingModelMiniLML6
from databases import NumpyDataBase

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, default="Are there any Artificial Neural Networks that can detect or predict cancer?")
    query = parser.parse_args().query
    
    with open("data.json", "r") as f:
        data = json.load(f)

    embedding_model = EmbeddingModelMiniLML6()
    database = NumpyDataBase(data=data, embedding_model=embedding_model)
    print(f"Retrieving results for query:\n{query}\n\nResults:\n")
    results = database.retrieve_by_query(query)

    for result in results:
        print(result['abstract'], end="\n\n")

