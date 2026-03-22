import torch
import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import math
from data import create_dataloader

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(self.seq_len, d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* 
                            (-math.log(10000)/self.d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    @torch.no_grad()
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :] # for different broadcasting purposes
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        assert d_model%h == 0,  "d_model not divisible by h"

        self.d_k = d_model//h
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, seq_len, d_k) -> (batch, seq_len, seq_len)
        attention_scores = ((query @ key.transpose(-2,-1))/math.sqrt(d_k))
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # batch, h, seq_len, seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores


    def forward(self, x, mask):
        # batch, seq len, d_model
        query = self.Wq(x)
        value = self.Wv(x)
        key = self.Wk(x)
        # batch, seq_len, h, d_k -> batch, h, seq_len, d_k
        query = query.view(query.shape[0], -1, self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], -1, self.h, self.d_k).transpose(1,2)
        # key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).permute([0,2,1,3])
        value = value.view(value.shape[0], -1, self.h, self.d_k).transpose(1,2)

        # (batch, h, seq_len, d_k) 
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2)
        x = x.contiguous().view(x.shape[0], -1, self.h*self.d_k)

        return self.Wo(x)

class ResidualConnections(nn.Module):
    def __init__(self, features:int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class AttentionBlock(nn.Module):
    def __init__(self, features, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork,
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
    
class StandardAttention(nn.Module):
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)
    
    def forward(self, x, mask=None):
        if mask is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(batch_size, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, seq_len: int, h: int, d_ff: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, seq_len, dropout)
        
        layers = nn.ModuleList([
            AttentionBlock(
                features=d_model,
                self_attention_block=MultiHeadAttention(d_model, h, dropout),
                feed_forward_block=FeedForwardNetwork(d_model, d_ff, dropout),
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.transformer = StandardAttention(d_model, layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
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
    
    model = DecoderOnlyTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        seq_len=cfg.seq_len,
        h=cfg.h,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    print("Starting Training Loop...")
    for epoch in range(cfg.epochs):
        model.train()
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
            if idx%20==0:
                wandb.log({"train_loss": loss.item()})
                print(f"Epoch {epoch+1} | Batch {idx} | Train Loss: {loss.item():.4f}")
            
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