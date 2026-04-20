import huggingface_hub

if not hasattr(huggingface_hub, "split_torch_state_dict_into_shards"):
    def dummy(*args, **kwargs):
        return None
    huggingface_hub.split_torch_state_dict_into_shards = dummy

from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("distilgpt2")
model.to(device)
model.eval()

tokenizer = model.tokenizer
hook_point = "blocks.2.hook_resid_pre"

class TopKSAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, k_percent=0.10):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = int(hidden_dim * k_percent)

    def forward(self, x):
        encoded = self.encoder(x)
        topk_values, topk_indices = torch.topk(encoded, self.k, dim=-1)
        sparse_encoded = torch.zeros_like(encoded)
        sparse_encoded.scatter_(-1, topk_indices, topk_values)
        reconstructed = self.decoder(sparse_encoded)
        return reconstructed, sparse_encoded

os.makedirs("./sae_checkpoints_64", exist_ok=True)

sae_512 = TopKSAE(input_dim=768, hidden_dim=512).to(device)
sae_1024 = TopKSAE(input_dim=768, hidden_dim=1024).to(device)

opt_512 = optim.Adam(sae_512.parameters(), lr=1e-4)
opt_1024 = optim.Adam(sae_1024.parameters(), lr=1e-4)

criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')

dataset = load_dataset("openwebtext", split="train", streaming=True)

def get_activation_batches(dataset, batch_size=128, seq_len=128):
    batch_texts = []
    token_buffer = []
    for example in dataset:
        tokens = tokenizer(example["text"])["input_ids"]
        token_buffer.extend(tokens)
        
        while len(token_buffer) >= seq_len:
            chunk = token_buffer[:seq_len]
            token_buffer = token_buffer[seq_len:]
            batch_texts.append(torch.tensor(chunk).unsqueeze(0)) 

            if len(batch_texts) == batch_size:
                yield torch.cat(batch_texts, dim=0)
                batch_texts = []

dataset_loader = get_activation_batches(dataset, batch_size=64) # Increased batch size

for step, batch_loaded in enumerate(dataset_loader):
    batch_loaded = batch_loaded.to(device)

    # A. Run the heavy LLM pass ONCE
    with torch.no_grad():
        _, cache = model.run_with_cache(batch_loaded, names_filter=[hook_point])
        layer_3_acts = cache[hook_point].detach()
    
    flat_acts = layer_3_acts.view(-1, 768).to(torch.float32)
    mean = flat_acts.mean(dim=0, keepdim=True)
    std = flat_acts.std(dim=0, keepdim=True)
    normalized_acts = (flat_acts - mean) / (std + 1e-5)
    
    shuffle_indices = torch.randperm(normalized_acts.size(0), device=device)
    shuffled_acts = normalized_acts[shuffle_indices]

    opt_512.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        recon_512, _ = sae_512(shuffled_acts)
        loss_512 = criterion(recon_512, shuffled_acts)
    scaler.scale(loss_512).backward()
    scaler.step(opt_512)

    # D. Update Model 2 (1024)
    opt_1024.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        recon_1024, _ = sae_1024(shuffled_acts)
        loss_1024 = criterion(recon_1024, shuffled_acts)
    scaler.scale(loss_1024).backward()
    scaler.step(opt_1024)
    
    scaler.update()

    # Logging
    if step % 50 == 0:
        print(f"Step {step} | Loss 512: {loss_512.item():.4f} | Loss 1024: {loss_1024.item():.4f}")

    if step > 0 and step % 5000 == 0:
        torch.save(sae_512.state_dict(), f"./sae_checkpoints_64/sae_m512_step{step}.pt")
        torch.save(sae_1024.state_dict(), f"./sae_checkpoints_64/sae_m1024_step{step}.pt")
        tqdm.write(f"--> Checkpoints saved at step {step}")

    if step >= 100000: 
        break

# Final Save
torch.save(sae_512.state_dict(), "./sae_m512_final_tl_100k.pt")
torch.save(sae_1024.state_dict(), "./sae_m1024_final_tl_100k.pt")
print("Training Complete!")