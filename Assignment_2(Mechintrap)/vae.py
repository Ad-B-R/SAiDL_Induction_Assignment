from datasets import load_dataset
import huggingface_hub    
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

resume_step = 10000
dataset = load_dataset("openwebtext", split="train", streaming=True)
skip_dataset = dataset.skip(resume_step)

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
        if self.training:
            z = self.reparam(mu, logvar)
        else:
            z = mu   
        recon = self.decoder(z)
        return recon, z, mu, logvar
    
def vae_loss(x, recon, mu, logvar, beta=0.01):
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl

# model(dummy_input)
m = [512]
for m_bottleneck in m:
    dataset_loader = get_activation_batches(dataset, batch_size=64)
    activations_cache = {} 
    os.makedirs("./vae_checkpoints_64", exist_ok=True)

    vae_model = VAE(input_dim=768, latent_dim=m_bottleneck).to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-4)
    vae_criterion = nn.MSELoss()
    vae_optimizer.zero_grad()
    if resume_step>0:
        ckpt_path = f"./vae_checkpoints_64/sae_m{m_bottleneck}_step{resume_step}_64.pt"
        vae_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"===== Successfully loaded checkpoint: {ckpt_path} =====")

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

        vae_optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            reconstructed_acts, z, mu, logvar = vae_model(shuffled_acts)
            loss = vae_loss(shuffled_acts, reconstructed_acts, mu, logvar, beta=0.01)    
        scaler.scale(loss).backward()
        scaler.step(vae_optimizer)
        scaler.update()

        if step % 10 == 0:
            print(f"Step {step+resume_step} | SAE Loss (MSE): {loss.item():.4f}")


        if step > 0 and step % 5000 == 0:
            ckpt_path = f"./vae_checkpoints_64/sae_m{m_bottleneck}_step{step+resume_step}_64.pt"
            torch.save(vae_model.state_dict(), ckpt_path)
            tqdm.write(f"--> Checkpoint saved: {ckpt_path}")
        
        # Stop exactly at 100,000 as requested
        if step >= 100000: 
            print("\nTraining Complete!")
            break

    # Final Save
    final_path = f"./vae_m{m_bottleneck}_final_100k.pt"
    torch.save(vae_model.state_dict(), final_path)
    print(f"Final model saved locally at: {final_path}")

    torch.save(vae_model.state_dict(), "distilgpt2_layer3_sae_100k.pt")