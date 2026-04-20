import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset
import json

class GeneralizedDataset(Dataset):
    def __init__(self, seq_len: int, split: str = "train"):
        self.seq_len = seq_len
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # 1. Load Tiny Shakespeare dataset
        hf_data = load_dataset("tiny_shakespeare", split=split)
        
        print(f"Tokenizing {split} split...")
        token_list = []
        
        eos_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

        # Tiny Shakespeare contains very few rows (often just one massive string per split), 
        # so this loop runs virtually instantly.
        for text in hf_data["text"]:
            if len(text.strip()) > 0:
                token_list.extend(tokenizer.encode(text))
                token_list.append(eos_id)
        
        self.tokens = torch.tensor(token_list, dtype=torch.long)
        print(f"Total tokens for {split}: {len(self.tokens)}")
        
    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        chunk = self.tokens[start_idx : start_idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:]
        }
            
def create_dataloader(split: str, shuffle: bool = True):
    with open('data.json', 'r') as file:
        data = json.load(file)
    dataset = GeneralizedDataset(data['seq_len'], split)
    
    return DataLoader(dataset, batch_size=data['batch_size'], shuffle=shuffle)