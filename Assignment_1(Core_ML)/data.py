import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset

class GeneralizedDataset(Dataset):
    def __init__(self, split: str, seq_len: int):
        self.seq_len = seq_len
        
        tokenizer = tiktoken.get_encoding("gpt2")
        
        hf_data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        
        raw_text = "".join(hf_data["text"])
        
        self.tokens = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
        print(f"Loaded {len(self.tokens)}")
    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        chunk = self.tokens[start_idx : start_idx + self.seq_len + 1]
        
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:]
        }
            
def create_dataloader(cfg: DictConfig, split: str, shuffle: bool = True):
    dataset = GeneralizedDataset(split, cfg.seq_len)
    return DataLoader(dataset, cfg.batch_size, shuffle=shuffle)