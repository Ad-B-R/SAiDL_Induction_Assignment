import torch
import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import math
from data import create_dataloader
from attention import baseline, Sliding_Window, GQA, Softmax
from positional import rpe
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
        
        # Debugging
        print(OmegaConf.to_yaml(cfg))

        self.embedding = InputEmbedding(cfg.d_model, cfg.vocab_size)
        
        # Set to none if they dont exist
        w = None
        h_GQA = None

        attention_type = cfg.attention.get("type", "Standard")
        rpe_pe = rpe.RelativePositionBias

        if attention_type=="GQA":
            attention_math = GQA.GroupedQueryAttention
            attention_body = GQA.StandardAttention
            h_GQA = getattr(cfg.attention, "h_GQA", 2)

            print(f"GQA no of Group heads: {h_GQA}")

        elif attention_type=="Sliding": 
            attention_math = Sliding_Window.SlidingWindowAttention
            attention_body = Sliding_Window.StandardAttention
            w = getattr(cfg.attention, "window_size", 64)

            print(f"Window Size: {w}")
        elif attention_type=="SoftmaxFree":
            attention_math = Softmax.MultiHeadAttention
            attention_body = Softmax.StandardAttention

            print("Softmax-free Attention being used")
        else:
            attention_math = baseline.MultiHeadAttention
            attention_body = baseline.StandardAttention

            print("Standard Attention being used")
        rpe_fn = rpe_pe(cfg.h, cfg.seq_len)
        layers = nn.ModuleList([
            AttentionBlock(
                features=cfg.d_model,
                self_attention_block=attention_math(cfg.d_model, cfg.h, cfg.dropout, h_GQA = h_GQA, 
                                                    rpe_fn=rpe_fn, use_rpe=True),
                feed_forward_block=FeedForwardNetwork(cfg.d_model, cfg.d_ff, cfg.dropout),
                dropout=cfg.dropout
            ) for _ in range(cfg.num_layers)
        ])
        self.transformer = attention_body(cfg.d_model, layers, window=w)
        
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
    def forward(self, x):
        x = self.embedding(x)
        
        x = self.transformer(x, mask=None) 
        
        # Output shape: (batch_size, seq_len, vocab_size)
        return self.lm_head(x)

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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_seq_len = cfg.seq_len
    base_batch_size = cfg.batch_size

    attention_types = ["Standard","GQA", "SoftmaxFree", "Sliding"]
    multipliers = [1, 2, 3, 4]
    multipliers = [2]
    
    for attn in attention_types:
        for mult in multipliers:
            cfg.attention.type = attn
            cfg.seq_len = base_seq_len * mult
            cfg.batch_size = max(1, base_batch_size // mult) 
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            wandb.init(
                project="transformer-master-ablation", 
                name=f"{attn}_seq_{cfg.seq_len}_RPE", 
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True 
            )
            
            train_loader = create_dataloader(cfg, split="train", shuffle=True)
            val_loader = create_dataloader(cfg, split="validation", shuffle=False)
            model = DecoderOnlyTransformer(cfg).to(device)
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

            if cfg.seq_len == 512:
                extrap_lengths = [512, 1024, 2048]
                
                for ext_len in extrap_lengths:
                    print(f"Testing Extrapolation on L={ext_len}...")
                    
                    cfg.seq_len = ext_len
                    cfg.batch_size = max(1, base_batch_size // (ext_len // 256)) 
                    
                    extrap_loader = create_dataloader(cfg, split="validation", shuffle=False)
                    
                    try:
                        extrap_loss, extrap_perp = run_validation(model, extrap_loader, loss_fn, cfg.vocab_size, device)
                        
                        wandb.log({
                            f"extrap_val_perplexity_L{ext_len}": extrap_perp
                        })
                        print(f"Extrap L={ext_len} | Perplexity: {extrap_perp:.2f}")
                        
                    except Exception as e:
                        print(f"Extrap L={ext_len} FAILED: {e}")
                        wandb.log({f"extrap_val_perplexity_L{ext_len}": float('inf')})
                
                cfg.seq_len = 512
            wandb.finish()
            
if __name__ == "__main__":
    train()