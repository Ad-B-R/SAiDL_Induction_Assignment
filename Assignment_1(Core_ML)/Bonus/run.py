import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
import math
from data import create_dataloader
import wandb
import hydra
import time
import AFT_full, AFT_local, AFT_simple, AFT_conv
import os
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt

print(f" CUDA: {'cuda' if torch.cuda.is_available() else 'cpu'}")

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        seq_len,
        num_layers,
        d_ff,
        n_heads,
        dropout,
        attention_type="full",
        window_size=32
    ):
        super().__init__()
        if attention_type=="full":
            attn_cls = AFT_full.AFTFull
            model = AFT_full

        elif attention_type=="local":
            attn_cls = AFT_local.AFTLocal
            model = AFT_local

        elif attention_type=="simple":
            attn_cls = AFT_simple.AFTSimple
            model = AFT_simple
        
        elif attention_type=="conv":
            attn_cls = AFT_conv.AFTConv
            model = AFT_conv

        else:
            raise ValueError(attention_type)
        
        self.embedding = model.InputEmbedding(
            d_model=d_model,
            vocab_size=vocab_size
        )

        self.pos_embed = model.PositionalEncoding(
            d_model=d_model,
            seq_len=seq_len,
            dropout=dropout
        )


        layers = nn.ModuleList([
            model.AttentionBlock(
                features=d_model,

                self_attention_block=attn_cls(
                    d_model=d_model,
                    seq_len=seq_len,
                    window_size=window_size,
                    dropout=dropout,
                    n_heads=n_heads,

                ),

                feed_forward_block=model.FeedForwardNetwork(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout
                ),

                dropout=dropout
            )

            for _ in range(num_layers)
        ])

        self.transformer = model.StandardAttention(
            features=d_model,
            layers=layers
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.pos_embed(x)

        x = self.transformer(x, mask=None)

        return self.lm_head(x)
        
config_dir = os.path.dirname(os.path.abspath(__file__))

@torch.no_grad()
def run_validation(model, dataloader, loss_fn, vocab_size, device, eval_iters=None):
    model.eval()
    total_loss = 0.0
    eval_full = 0
    for i, batch in enumerate(dataloader):
        eval_full+=1
        if eval_iters is not None and i >= eval_iters:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            loss = loss_fn(logits, y)

            
        total_loss += loss.item()
        
    model.train()
    avg_loss = total_loss / eval_iters if eval_iters != None else total_loss/eval_full
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity

@hydra.main(version_base=None, config_path=config_dir, config_name="config")
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_seq_len = cfg.seq_len
    base_batch_size = cfg.batch_size

    attention_types = ["conv"
                    , "local", "simple", "full"
                       ]
    multipliers = [2] # seq_len multi
    
    for attn in attention_types:
        for mult in multipliers:
            cfg.attention = attn
            cfg.seq_len = base_seq_len * mult
            cfg.batch_size = max(1, base_batch_size // mult) 
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            wandb.init(
                project="transformer-master-ablation", 
                name=f"{attn}_seq_{cfg.seq_len}_Standard", 
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True 
            )
            
            train_loader = create_dataloader(cfg, split="train", shuffle=True)
            val_loader = create_dataloader(cfg, split="validation", shuffle=False)
            model = GPT(vocab_size= cfg.vocab_size, d_model=cfg.d_model, window_size=cfg.window_size, 
                        seq_len=cfg.seq_len, n_heads=cfg.h, num_layers=cfg.num_layers, 
                        d_ff=cfg.d_ff, dropout=cfg.dropout, attention_type=attn).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            loss_fn = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler("cuda")
            
            global_step = 0
            
            for epoch in range(cfg.epochs):
                epoch_start_time = time.time()
                epoch_loss = torch.tensor(0.0, device=device)
                tokens_in_epoch = 0
                model.train()
                
                start_time = time.time() 
                tokens_processed = 0      

                for idx, batch in enumerate(train_loader):
                    x = batch["input_ids"].to(device)
                    y = batch["labels"].to(device)
                    
                    optimizer.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = loss_fn(logits.view(-1, cfg.vocab_size), y.view(-1))
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += loss.detach()
                    tokens_in_epoch += (x.shape[0] * x.shape[1])
                    
                    tokens_processed += (x.shape[0] * x.shape[1])
                    global_step += 1
                    if idx % 20 == 0 and idx != 0 and idx%250!=0: 
                        elapsed_time = time.time() - start_time
                        throughput = tokens_processed / elapsed_time
                        
                        wandb.log({
                            "train_loss": loss.item(),
                            "throughput_tkn/s": throughput,
                            "epoch": epoch + (idx / len(train_loader))
                        }, step=global_step)
                        
                        print(f"Epoch {epoch+1} | Batch {idx} | Train Loss: {loss.item():.4f} | Throughput: {throughput:.0f} tkn/s")
                        
                        start_time = time.time()
                        tokens_processed = 0
                    
                    if idx % 250 == 0 and idx != 0:
                        elapsed_time = time.time() - start_time
                        throughput = tokens_processed / elapsed_time
                        
                        wandb.log({
                            "train_loss": loss.item(),
                            "throughput_tkn/s": throughput,
                            "epoch": epoch + (idx / len(train_loader))
                        }, step=global_step)
                        
                        print(f"Epoch {epoch+1} | Batch {idx} | Train Loss: {loss.item():.4f} | Throughput: {throughput:.0f} tkn/s")
                        val_loss, val_perplexity = run_validation(model, val_loader, loss_fn, cfg.vocab_size, device, eval_iters=50)
                        
                        wandb.log({
                            "val_loss": val_loss,
                            "val_perplexity": val_perplexity
                        }, step=global_step)
                        
                        print(f"--- MIDWAY EVAL | Batch {idx} | Val Loss: {val_loss:.4f} | Val Perp: {val_perplexity:.4f} ---")

                        start_time = time.time()
                        tokens_processed = 0

                epoch_duration = time.time() - epoch_start_time
                avg_train_loss = epoch_loss.item() / len(train_loader)
                epoch_throughput = tokens_in_epoch / epoch_duration
                
                val_loss, val_perplexity = run_validation(model, val_loader, loss_fn, cfg.vocab_size, device)
                
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

                wandb.log({
                    "epoch": epoch + 1,
                    "final_train_loss": avg_train_loss,
                    "final_val_loss": val_loss,
                    "final_val_perplexity": val_perplexity,
                    "peak_gpu_mem_gb": peak_mem_gb,
                    "train_time_epoch_sec": epoch_duration,
                    "avg_throughput_tkn_s": epoch_throughput
                }, step=global_step)

                print(f"DONE {attn} L{cfg.seq_len} | Loss: {val_loss:.4f} | Mem: {peak_mem_gb:.2f}GB | Speed: {epoch_throughput:.0f} tkn/s")
            wandb.finish()
if __name__=="__main__":
    train()