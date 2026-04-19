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

# Hook point (layer 3 input)
hook_point = "blocks.2.hook_resid_pre"

dataset = load_dataset("openwebtext", split="train", streaming=True)


def get_activation_batches(dataset, batch_size=1024, seq_len=128):
    batch_texts = []
    token_buffer = []

    for example in dataset:
        tokens = tokenizer(example["text"], return_tensors="pt")["input_ids"][0]
        token_buffer.extend(tokens.tolist())

        while len(token_buffer) >= seq_len:
            chunk = token_buffer[:seq_len:1]
            token_buffer = token_buffer[seq_len::1]

            batch_texts.append(torch.tensor(chunk).unsqueeze(0)) 

            if len(batch_texts) == batch_size:
                yield torch.cat(batch_texts, dim=0)
                batch_texts = []

dataset_loader = get_activation_batches(dataset, batch_size=64)

class TopKSAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, k_percent=0.10):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = int(hidden_dim * k_percent)

    def forward(self, x):
        # Expand
        encoded = self.encoder(x)
        
        # Enforce Top-K Sparsity
        topk_values, topk_indices = torch.topk(encoded, self.k, dim=-1)
        
        sparse_encoded = torch.zeros_like(encoded)
        sparse_encoded.scatter_(-1, topk_indices, topk_values)
        
        reconstructed = self.decoder(sparse_encoded)
        return reconstructed, sparse_encoded

# model(dummy_input)
m_bottleneck = 512
os.makedirs("./sae_checkpoints_64", exist_ok=True)

sae_model = TopKSAE(input_dim=768, hidden_dim=m_bottleneck).to(device)
sae_optimizer = optim.Adam(sae_model.parameters(), lr=1e-4)
sae_criterion = nn.MSELoss()
sae_optimizer.zero_grad()

scaler = torch.amp.GradScaler('cuda')

for step, batch_loaded in enumerate(dataset_loader):
    batch_loaded = batch_loaded.to(device)
    hook_point = "blocks.2.hook_resid_pre"

    with torch.no_grad():
        _, cache = model.run_with_cache(batch_loaded, names_filter=[hook_point])
        layer_3_acts = cache[hook_point].detach().half()
    flat_acts = layer_3_acts.view(-1, 768).to(torch.float32) # Move back to float32 for SAE math stability
    
    mean = flat_acts.mean(dim=0, keepdim=True)
    std = flat_acts.std(dim=0, keepdim=True)
    normalized_acts = (flat_acts - mean) / (std + 1e-5)
    
    shuffle_indices = torch.randperm(normalized_acts.size(0), device=device)
    shuffled_acts = normalized_acts[shuffle_indices]

    sae_optimizer.zero_grad(set_to_none=True)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        reconstructed_acts, sparse_features = sae_model(shuffled_acts)
        loss = sae_criterion(reconstructed_acts, shuffled_acts)
    
    scaler.scale(loss).backward()
    scaler.step(sae_optimizer)
    scaler.update()

    if step % 10 == 0:
        print(f"Step {step} | SAE Loss (MSE): {loss.item():.4f}")

    # if step >= 200: 
    #     print("Debug pass complete!")
    #     break

    if step > 0 and step % 5000 == 0:
        ckpt_path = f"./sae_checkpoints_64/sae_m{m_bottleneck}_step{step}_64.pt"
        torch.save(sae_model.state_dict(), ckpt_path)
        tqdm.write(f"--> Checkpoint saved: {ckpt_path}")

    # Stop exactly at 100,000 as requested
    if step >= 100000: 
        print("\nTraining Complete!")
        break

# Final Save
final_path = f"./sae_m{m_bottleneck}_final_100k.pt"
torch.save(sae_model.state_dict(), final_path)
print(f"Final model saved locally at: {final_path}")

torch.save(sae_model.state_dict(), "distilgpt2_layer3_sae_100k.pt")