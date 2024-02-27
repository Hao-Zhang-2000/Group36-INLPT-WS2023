from tqdm import tqdm
import numpy as np
import json
import torch
from embedding_models import EmbeddingModelMiniLML6

model = EmbeddingModelMiniLML6()

with open("data_raw.json", 'r') as f: 
    raw_data = json.load(f)
    metadata_list = []
    for i in range(len(raw_data['abstracts'])):
        metadata_list.append({
            "abstract": raw_data['abstracts'][i],
            "authors": raw_data['authior'][i],
            "title": raw_data['title'][i],
            "article_id": raw_data['article_id'][i]
        })

print("Found", len(metadata_list), "samples in 'data_raw.json'")
print("Embedding chunks..")

data = []
for sample in tqdm(metadata_list):
    embedded_chunks = model.embed_passage(sample['abstract'])
    for chunk in embedded_chunks:
        data.append({
            "embedding": chunk,
            "metadata": sample
        })
        
print("Produced", len(data), "embedded chunks together with their metadata")
print("Saving to data.json..")

with open("data.json", "w+") as f:
    f.write(json.dumps(data, indent=2))

print("Finished!")