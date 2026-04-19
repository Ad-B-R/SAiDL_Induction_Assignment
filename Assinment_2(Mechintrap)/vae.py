from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()
model.to(device)

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
activations_cache = {} 

class VAE(nn.Module):
    def __init__(self, input_dim=768, latent_dim=1024):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 2 * latent_dim)  # mu + logvar
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decoder(z)
        return recon, z, mu, logvar
    
def hook_fn(module, input, output):
    activations_cache["layer_3"] = output[0].detach().half() 

def vae_loss(x, recon, mu, logvar, beta=0.01):
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl

hook_handle = model.transformer.h[2].register_forward_hook(hook_fn)

# model(dummy_input)
m_bottleneck = 512*2
os.makedirs("./sae_checkpoints_64", exist_ok=True)

vae_model = VAE(input_dim=768, hidden_dim=m_bottleneck).to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-4)
vae_criterion = nn.MSELoss()
vae_optimizer.zero_grad()

scaler = torch.amp.GradScaler('cuda')

for step, batch_loaded in enumerate(dataset_loader):
    batch_loaded = batch_loaded.to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(batch_loaded)

    layer_3_acts = activations_cache["layer_3"]
    activations_cache.clear() # Prevent memory leaks
    
    del out
    flat_acts = layer_3_acts.view(-1, 768).to(torch.float32) # Move back to float32 for SAE math stability
    
    mean = flat_acts.mean(dim=0, keepdim=True)
    std = flat_acts.std(dim=0, keepdim=True)
    normalized_acts = (flat_acts - mean) / (std + 1e-5)
    
    shuffle_indices = torch.randperm(normalized_acts.size(0), device=device)
    shuffled_acts = normalized_acts[shuffle_indices]

    vae_optimizer.zero_grad(set_to_none=True)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        reconstructed_acts, z, mu, logvar = vae_model(shuffled_acts)
        loss = vae_loss(shuffled_acts, reconstructed_acts, mu, logvar, beta=0.01)    
    scaler.scale(loss).backward()
    scaler.step(vae_optimizer)
    scaler.update()

    if step % 10 == 0:
        print(f"Step {step} | SAE Loss (MSE): {loss.item():.4f}")

    # if step >= 200: 
    #     print("Debug pass complete!")
    #     break

    if step > 0 and step % 5000 == 0:
        ckpt_path = f"./sae_checkpoints_64/sae_m{m_bottleneck}_step{step}_64.pt"
        torch.save(vae_model.state_dict(), ckpt_path)
        tqdm.write(f"--> Checkpoint saved: {ckpt_path}")

    # Stop exactly at 100,000 as requested
    if step >= 100000: 
        print("\nTraining Complete!")
        break

# Final Save
final_path = f"./sae_m{m_bottleneck}_final_100k.pt"
torch.save(vae_model.state_dict(), final_path)
print(f"Final model saved locally at: {final_path}")

hook_handle.remove()
torch.save(vae_model.state_dict(), "distilgpt2_layer3_sae_100k.pt")