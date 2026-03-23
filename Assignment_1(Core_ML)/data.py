import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset

Run for different context lengths


class GeneralizedDataset(Dataset):
    def __init__(self, seq_len: int, split: str = "train"):
        self.seq_len = seq_len
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # 1. Load dataset
        hf_data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        
        print(f"Tokenizing {split} split...")
        token_list = []
        
        eos_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

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
            
def create_dataloader(cfg: DictConfig, split: str, shuffle: bool = True):
    dataset = GeneralizedDataset(cfg.seq_len, split)
    return DataLoader(dataset, cfg.batch_size, shuffle=shuffle)