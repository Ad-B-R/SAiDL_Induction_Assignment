import torch
import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import math
from data import create_dataloader
from attention import baseline, Sliding_Window, GQA, Softmax
from positional import sine_cosine, rope, rpe, alibi
import time


# calculate the throughput
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fnn1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fnn2(self.dropout(torch.relu(self.fnn1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features:int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x-mean)/(std + self.eps) + self.beta

class ResidualConnections(nn.Module):
    def __init__(self, features:int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class AttentionBlock(nn.Module):
    def __init__(self, features, self_attention_block: nn.Module, feed_forward_block: FeedForwardNetwork,
                 dropout: float, ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_foward_network = feed_forward_block
        self.dropout = dropout
        self.residual_connections = nn.ModuleList([ResidualConnections(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Residual_connections stores residual blocks, 
        # hence following means ResidualConnections.forward(x, sublayer)
        x = self.residual_connections[0](x, 
            lambda x: self.self_attention_block(x,src_mask))
        # residual_connections__Call__() expects only one input function, hence we do this
        x = self.residual_connections[1](x, self.feed_foward_network)
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.embedding = InputEmbedding(cfg.d_model, cfg.vocab_size)
        w = None
        h_GQA = None
        if True:
            self.pos_encoding = sine_cosine.PositionalEncoding(cfg.d_model, cfg.seq_len, cfg.dropout)
            

        if True:
            attention_math = GQA.GroupedQueryAttention
            attention_body = GQA.StandardAttention

            h_GQA = cfg.h_GQA
        elif False: 
    #   if cfg.attention.type == "sliding_window":
            attention_math = Sliding_Window.SlidingWindowAttention
            attention_body = Sliding_Window.StandardAttention
            w = cfg.window_size
        else:
            attention_math = baseline.MultiHeadAttention
            attention_body = baseline.StandardAttention

        layers = nn.ModuleList([
            AttentionBlock(
                features=cfg.d_model,
                self_attention_block=attention_math(cfg.d_model, cfg.h, cfg.dropout, h_GQA = h_GQA),
                feed_forward_block=FeedForwardNetwork(cfg.d_model, cfg.d_ff, cfg.dropout),
                dropout=cfg.dropout
            ) for _ in range(cfg.num_layers)
        ])
        self.transformer = attention_body(cfg.d_model, layers, window=w)
        
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        x = self.transformer(x, mask=None) 
        
        # Output shape: (batch_size, seq_len, vocab_size)
        return self.lm_head(x)

def run_validation(model, dataloader, device):
    perplexity = 0.0
    return perplexity

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="transformer-wikitext",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    train_loader = create_dataloader(cfg, split="train", shuffle=True)
    val_loader = create_dataloader(cfg, split="validation", shuffle=False)
    
    model = DecoderOnlyTransformer(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    print("Starting Training Loop...")
    for epoch in range(cfg.epochs):
        model.train()
        
        start_time = time.time()
        tokens_processed = 0
        
        for idx, batch in enumerate(train_loader):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            logits = logits.view(-1, cfg.vocab_size)
            y = y.view(-1)
            
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            tokens_processed += (x.shape[0] * x.shape[1])
            
            if idx % 20 == 0: 
                elapsed_time = time.time() - start_time
                throughput = tokens_processed / elapsed_time
                
                wandb.log({
                    "train_loss": loss.item(),
                    "throughput_tok_sec": throughput 
                })
                
                print(f"Epoch {epoch+1} | Batch {idx} | Train Loss: {loss.item():.4f} | Throughput: {throughput:.0f} tok/s")
                
                start_time = time.time()
                tokens_processed = 0
            
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                
                logits = model(x)
                
                logits = logits.view(-1, cfg.vocab_size)
                y = y.view(-1)
                
                loss = loss_fn(logits, y)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Perplexity: {perplexity:.4f}")
        wandb.log({
            "val_loss": avg_val_loss,
            "val_perplexity": perplexity,
            "epoch": epoch + 1
        })
        
    wandb.finish()

if __name__ == "__main__":
    train()