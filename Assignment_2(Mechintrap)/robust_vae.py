import huggingface_hub
from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
import random
from tqdm.auto import tqdm
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import wandb
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


model = HookedTransformer.from_pretrained("distilgpt2")
model = model.to(device)
model.eval()

tokenizer = model.tokenizer
k = 0
m = [512]
for m_bottleneck in m:
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
        
    vae_model = VAE(input_dim=768, latent_dim=m_bottleneck).to(device)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    parent_dir = os.path.dirname(current_script_dir)

    weights_folder = os.path.join(parent_dir, "vae_checkpoints_64")

    model_file_name = f"sae_m{m_bottleneck}_step{20000}_64.pt"

    final_weights_path = os.path.join(weights_folder, model_file_name)
    vae_model.load_state_dict(torch.load(final_weights_path, map_location=device))
    vae_model.eval()

    dataset = load_dataset("openwebtext", split="train", streaming=True)
    skip_dataset = dataset.skip(600_000) # skip 400k data so that there is no collision between training and testing data

    hook_point = "blocks.2.hook_resid_pre"

    def get_eval_batches(dataset_stream, num_batches, batch_size=32, seq_len=128):
        batch_texts = []
        token_buffer = []
        batches_yielded = 0

        for example in dataset_stream:
            tokens = tokenizer(example["text"], return_tensors="pt")["input_ids"][0]
            token_buffer.extend(tokens.tolist())

            while len(token_buffer) >= seq_len:
                chunk = token_buffer[:seq_len:1]
                token_buffer = token_buffer[seq_len::1]

                batch_texts.append(torch.tensor(chunk).unsqueeze(0)) 

                if len(batch_texts) == batch_size:
                    yield torch.cat(batch_texts, dim=0).to(device)
                    batch_texts = []
                    batches_yielded += 1
                    if batches_yielded >= num_batches:
                        return # Stop the generator once we hit our eval limit
                        
    def compute_fisher(Z):
        return (Z ** 2).mean(dim=0)

    def quantize_gen(min, max, bit, tensor):
        q_max = int(2**bit - 1)
        s = (max-min)/(q_max + 1e-9)
        shift = tensor - min
        q = torch.round(shift/(s + 1e-9))
        q_clip = torch.clamp(q, 0, q_max)
        de_quantize = (q_clip*s) + min
        return de_quantize
    
    def get_range(x, p=0.999):
        high = torch.quantile(x, p)
        low = torch.quantile(x, 1 - p)
        return low, high
 
    def compute_cka(X, Y):
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

        K = X @ X.T
        L = Y @ Y.T

        hsic = (K * L).sum()
        norm_x = torch.norm(K)
        norm_y = torch.norm(L)

        return (hsic / (norm_x * norm_y + 1e-9)).item()

    def spectral_analysis(Z, Z_hat, k=32):

        Z_c = Z - Z.mean(dim=0, keepdim=True)
        Z_c_hat = Z_hat - Z_hat.mean(dim=0, keepdim=True)

        U, S, V = torch.svd_lowrank(Z_c, q=k)
        U_hat, S_hat, V_hat = torch.svd_lowrank(Z_c_hat, q=k)

        Uk = V[:, :k]
        Uk_hat = V_hat[:, :k]

        cos_thetas = torch.linalg.svdvals(Uk.T @ Uk_hat)
        angles = torch.acos(torch.clamp(cos_thetas, -1, 1)) * (180 / math.pi)

        E = Z - Z_hat
        sds = (torch.norm(E @ Uk)**2 / torch.norm(Z @ Uk)**2).item()

        return {
            "singular_before": S,
            "singular_after": S_hat,
            "angles": angles,
            "sds": sds
        }

    print("\nCalibrating Min/Max ranges on held-out data...")

    calib_batches = list(get_eval_batches(skip_dataset, num_batches=1))
    global_min, global_max = float('inf'), float('-inf')
    feature_min, feature_max = None, None

    eval_batches = list(get_eval_batches(skip_dataset, num_batches=5))

    with torch.no_grad():
        for batch in calib_batches:

            seq_len = batch.size(1)
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            
            # Start with Embeddings
            
            _, cache = model.run_with_cache(batch)
            acts = cache["blocks.2.hook_resid_pre"]
            _, sparse_feats, _, _ = vae_model(acts) 
            batch_min = sparse_feats.amin(dim=(0,1))   # shape [D]
            batch_max = sparse_feats.amax(dim=(0,1))

            global_min = min(global_min, batch_min.min().item())
            global_max = max(global_max, batch_max.max().item())

            if feature_min is None:
                feature_min = batch_min
                feature_max = batch_max
            else:
                feature_min = torch.minimum(feature_min, batch_min)
                feature_max = torch.maximum(feature_max, batch_max)
    feature_min = torch.full_like(feature_min, global_min)
    feature_max = torch.full_like(feature_max, global_max)
    print(f"Calibration Complete | Min: {global_min:.4f}, Max: {global_max:.4f}")
    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)    
  
    
    bit_configs = [16, 8, 4, 2, None]

    modes = ["per_tensor"]
    umap_data = {
        "Baseline": [],
        'per_tensor': {
        "16-bit_per_tensor": [],
        "8-bit_per_tensor": [],
        "4-bit_per_tensor": [],
        "2-bit_per_tensor": []
        }
    }

    results = {}
    def get_quant_hook(mode, bits, vae_model, feature_min, feature_max):
        low  = feature_min.view(1, 1, -1)
        high = feature_max.view(1, 1, -1)
        
        def hook_fn(acts, hook):
            _, Z, _, _ = vae_model(acts)
            
            if bits is None:
                return acts
            
            if mode == "per_tensor":
                Z_flat   = Z.view(-1, Z.size(-1))
                Z_hat_flat = quantize_gen(
                    min=low.view(-1, low.size(-1)),
                    max=high.view(-1, high.size(-1)),
                    bit=bits,
                    tensor=Z_flat
                )
                Z_hat = Z_hat_flat.view_as(Z)
            else:
                raise ValueError(mode)
            
            recon = vae_model.decoder(Z_hat)
            return recon
        
        return hook_fn
    def compute_ppl_with_hook(model, eval_batches, hook_fn):
        total_loss = 0

        with torch.no_grad():
            for batch in eval_batches:
                logits = model.run_with_hooks(
                    batch,
                    fwd_hooks=[(hook_point, hook_fn)]
                )

                loss = model.loss_fn(logits, batch)
                total_loss += loss.item()

        return math.exp(total_loss / len(eval_batches))

    for mode in modes:
        print(f"\nMode: {mode}")

        for bits in bit_configs:
            acts_samples = []
            max_samples = 10
            if bits is not None:
                label = f"{bits}-bit" if bits!=16 else "Baseline_16_bit"
            else:
                label = "Standard_Pass_baseline"

            wandb.init(
            project="vae-quantization",
            name=f"{mode}_{label}_analysis_{m_bottleneck}",
            reinit=True
            )
            print(f"\nEvaluating: {label}")
            Z_all = []
            Z_hat_all = []
            tokens_all = []

            handle = (None)                
            with torch.no_grad():
                idx = 0
                for batch in tqdm(eval_batches, leave=False):

                    _, cache = model.run_with_cache(batch)
                    clean_acts = cache["blocks.2.hook_resid_pre"]
                    B, T, D = clean_acts.shape
                    flat = clean_acts.view(-1, D)

                    _, sparse_feats_flat, _, _ = vae_model(flat)
                    Z_batch = sparse_feats_flat.view(B, T, -1)
                    if len(acts_samples) < max_samples:
                        acts_samples.append(clean_acts[:1].detach())    
                    
                    if bits is not None:
                        low = feature_min.view(1,1,-1)
                        high = feature_max.view(1,1,-1)
                        Z_hat_batch = quantize_gen(
                            min=low,
                            max=high,
                            bit=bits,
                            tensor=Z_batch
                        )

                    else:
                        Z_hat_batch = Z_batch    

                    Z = Z_batch.view(-1, Z_batch.size(-1))
                    Z_hat = Z_hat_batch.view(-1, Z_batch.size(-1))

                    Z_all.append(Z)
                    Z_hat_all.append(Z_hat)
                    tokens_all.append(batch.view(-1))

                    recon_acts = vae_model.decoder(Z_hat_batch)
                    
                    Z_hat_np = (Z_hat.detach().cpu().numpy())
                    if bits is None: umap_data["Baseline"].append(Z_hat_np)
                    else: umap_data[f"{mode}"][f"{bits}-bit_{mode}"].append(Z_hat_np)
            
            
            Z = torch.cat(Z_all)
            Z_hat = torch.cat(Z_hat_all)

            tokens = torch.cat(tokens_all)
            
            hook_fn = get_quant_hook(
                mode=mode,
                bits=bits,
                vae_model=vae_model,
                feature_min=feature_min,
                feature_max=feature_max,
            )

            ppl = compute_ppl_with_hook(model, eval_batches, hook_fn)

            if bits is not None: print(f"{mode} | {bits}-bit → PPL: {ppl:.2f}")
            else: print(f"{mode} | {bits}-bit → PPL: {ppl:.2f}")
            spectral = spectral_analysis(Z, Z_hat)

            fisher = compute_fisher(Z)

            sv_before = spectral["singular_before"]
            sv_after = spectral["singular_after"]

            # fraction of near-zero singular values
            collapse_ratio = (sv_after < 1e-3).float().mean().item()

            # how much variance is lost
            energy_before = (sv_before**2).sum()
            energy_after = (sv_after**2).sum()
            energy_loss = ((energy_before - energy_after) / (energy_before + 1e-9)).item()
            
            cka = compute_cka(Z, Z_hat)
            
            angles = spectral["angles"]
            mean_angle = angles.mean().item()
            max_angle = angles.max().item()

            def sparsity(x):
                return (x == 0).float().mean().item()

            s_before = sparsity(Z)
            s_after = sparsity(Z_hat)

            # how much sparsity changed
            sparsity_delta = s_after - s_before
                
            wandb.log({
                "bit_high": 16,
                "k_value": k,
                "Perplexity": ppl,
                "collapse_ratio": collapse_ratio,
                "energy_loss": energy_loss,
                "sparsity_before": s_before,
                "sparsity_after": s_after,
                "sparsity_delta": sparsity_delta,
                "angle_mean": mean_angle,
                "angle_max": max_angle,
                "CKA": cka,
                "SDS": spectral["sds"]
            })                

            print(f" Summary ({label})")
            print(f"Representation Damage:")
            print(f"SDS: {spectral['sds']:.4f}")
            print(f"Angle: {spectral['angles'].mean().item():.2f}°")
            
            print(f"\nCausal Alignment (Damage vs Perplexity Impact):")

    print("<------------------- ROBUST QUANTIZATION FINISHED --------------------------->")
    print("<------------------- SUCCESS --------------------------->")  