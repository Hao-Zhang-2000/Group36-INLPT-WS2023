import numpy as np
import torch
from torch.utils.data import Dataset
import json
from utils import format_conversation

class RAGDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        system_message,
        message_templates,
        pad_length = 4096,
    ):
        self.tokenizer = tokenizer
        self.system_message = system_message
        self.message_templates = message_templates
        self.pad_length = pad_length
        
        with open("training_data/interactions.json", "r") as f:
            self.samples = json.load(f)
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conversation = [{
            "role": "system", 
            "content": self.system_message
        }] + self.samples[idx]['conversation-history']
        
        _, inputs = format_conversation(conversation, self.message_templates, self.tokenizer, pad_length=self.pad_length)
        return inputs









