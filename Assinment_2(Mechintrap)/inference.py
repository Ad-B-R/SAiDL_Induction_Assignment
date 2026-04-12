from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import math
import wandb
import matplotlib.pyplot as plt
import umap
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
model.eval()


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

m_bottleneck = 512 
sae_model = TopKSAE(hidden_dim=m_bottleneck).to(device)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_script_dir, "weights")
model_file_name = f"sae_m{m_bottleneck}_final_100k.pt"
final_weights_path = os.path.join(weights_folder, model_file_name)

sae_model.load_state_dict(torch.load(final_weights_path, map_location=device))
sae_model.eval()


dataset = load_dataset("openwebtext", split="train", streaming=True)
skip_dataset = dataset.skip(2_000_000) # skip 2 million data so that there is no collision between training and testing data

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

def quantize_gen(min, max, bit, tensor):
    q_max = 2**bit - 1
    s = (max-min)/(q_max + 1e-9)
    shift = tensor - min
    q = torch.round(shift/(s + 1e-9))
    q_clip = torch.clamp(q, 0, q_max)
    de_quantize = (q_clip*s) + min
    return de_quantize

def SDS(Z, Z_hat, k):
    Z_c = Z - Z.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Z_c, full_matrices=False)
    Uk = Vh[:k].T   # [D, k]

    E = Z - Z_hat

    num = torch.norm(E @ Uk)**2
    den = torch.norm(Z @ Uk)**2

    return (num/den).item()

def CKA(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    dot_product_similarity = torch.norm(X.t() @ Y)**2
    
    normalization_x = torch.norm(X.t() @ X)
    normalization_y = torch.norm(Y.t() @ Y)
    
    cka_score = dot_product_similarity / (normalization_x * normalization_y + 1e-9)
    return cka_score.item()

print("\nCalibrating Min/Max ranges on held-out data...")

calib_batches = list(get_eval_batches(skip_dataset, num_batches=1))
global_min, global_max = float('inf'), float('-inf')
feature_min, feature_max = None, None


with torch.no_grad():
    for batch in calib_batches:

        acts = model.transformer.h[2](model.transformer.h[1](model.transformer.wte(batch))[0])[0]
        _, sparse_feats = sae_model(acts)
        
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

print(f"Calibration Complete | Min: {global_min:.4f}, Max: {global_max:.4f}")

def get_intervention_hook(bits, mode = "per_tensor"):
    def hook(module, input, output):
        orig_acts = output[0]
        _, sparse_feats = sae_model(orig_acts)
        
        if bits is not None:
            if mode=="per_tensor":    
                damaged_feats = quantize_gen(tensor=sparse_feats, bit=bits, min=global_min, max=global_max)
            elif mode == "per_feature":
                damaged_feats = quantize_gen(
                    sparse_feats, bits,
                    feature_min.unsqueeze(0).unsqueeze(0),
                    feature_max.unsqueeze(0).unsqueeze(0)
                )
            reconstructed = sae_model.decoder(damaged_feats)
        else:
            reconstructed = orig_acts  # baseline
            
        return (reconstructed,) + output[1:]
    return hook

bit_configs = [None, 8, 4, 2]
eval_batches = list(get_eval_batches(skip_dataset, num_batches=5))
modes = ["per_tensor", "per_feature"]

for mode in modes:
    print(f"\nMode: {mode}")

    for bits in bit_configs:
        label = f"{bits}-bit" if bits else "Baseline"
        print(f"\nEvaluating: {label}")

        handle = (
            model.transformer.h[2].register_forward_hook(
                get_intervention_hook(bits, mode)
            )
            if bits is not None else None
        )                
        wandb.init(
            project="sae-quantization",
            config={
                "model": "distilgpt2",
                "layer": 3,
                "sae_bottleneck": f"{m_bottleneck}",
                "k_percent": 0.10,
                "seq_len": 128,
                "model_batch": 32,
                "mode": f"{mode}",
                "bits": f"{bits}",
            },
            reinit=True
        )
        total_loss, total_mse, total_sds, total_cka = 0, 0, 0, 0

        with torch.no_grad():
            idx = 0
            for batch in tqdm(eval_batches, leave=False):
                outputs = model(batch, labels=batch)
                total_loss += outputs.loss.item()
                clean_acts = model.transformer.h[2](
                    model.transformer.h[1](model.transformer.wte(batch))[0])[0]

                _, sparse_feats = sae_model(clean_acts)
                Z_batch = sparse_feats.clone()
                if bits is not None:
                    if mode == "per_tensor":
                        Z_hat_batch = quantize_gen(
                            sparse_feats, bits, global_min, global_max
                        )
                    else:
                        Z_hat_batch = quantize_gen(
                            sparse_feats, bits,
                            feature_min.unsqueeze(0).unsqueeze(0),
                            feature_max.unsqueeze(0).unsqueeze(0)
                        )
                else:
                    Z_hat_batch = Z_batch
                Z = Z_batch.view(-1, Z_batch.size(-1))
                Z_hat = Z_hat_batch.view(-1, Z_batch.size(-1))

                recon_acts = sae_model.decoder(Z_hat_batch)
                
                total_mse += nn.MSELoss()(recon_acts, clean_acts).item()
                total_sds += SDS(Z=Z, Z_hat=Z_hat, k=32)
                total_cka += CKA(X=recon_acts.view(-1, 768), Y = clean_acts.view(-1, 768))

                if idx%500==0:
                    wandb.log({
                    "perplexity": math.exp(total_loss/(idx+1)),
                    "mse": total_mse / (idx+1) if bits is not None else 0,
                    "cka": total_cka / (idx+1) if bits is not None else 0,
                    "sds": total_sds / (idx+1) if bits is not None else 0
                    })
                idx+=1
        avg_loss = total_loss / len(eval_batches)
        print(f"Perplexity: {math.exp(avg_loss):.2f}")

        if bits is not None:    
            print(f"MSE: {total_mse / len(eval_batches):.4f}")

        if handle:
            handle.remove()

print("\nGathering 10,000 tokens for UMAP visualization...")
all_clean = []

with torch.no_grad():
    for batch in eval_batches:
        clean = model.transformer.h[2](model.transformer.h[1](model.transformer.wte(batch))[0])[0]
        all_clean.append(clean.view(-1, 768))

X_clean = torch.cat(all_clean, dim=0)[:10000]
viz_data = {"Baseline": X_clean.cpu().numpy()}

with torch.no_grad():
    _, feats = sae_model(X_clean)
    
    for bits in [8, 4, 2]:
        # Applying per_tensor quantization damage
        damaged = quantize_gen(feats, bits, global_min, global_max)
        recon = sae_model.decoder(damaged)
        viz_data[f"{bits}-bit"] = recon.cpu().numpy()

print("Running UMAP Projection (This will take ~60 seconds)...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("UMAP Projection of SAE Quantization Damage", fontsize=16)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
baseline_2d = reducer.fit_transform(viz_data["Baseline"])

for idx, (label, data) in enumerate(viz_data.items()):
    proj = baseline_2d if label == "Baseline" else reducer.transform(data)
    
    axes[idx].scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.5)
    axes[idx].set_title(label)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("quantization_grid_collapse.png", dpi=300)
print("\nSaved visualization to 'quantization_grid_collapse.png'")