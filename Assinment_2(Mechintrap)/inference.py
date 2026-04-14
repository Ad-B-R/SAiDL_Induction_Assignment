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

m_bottleneck = 512

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

sae_model = TopKSAE(hidden_dim=m_bottleneck).to(device)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_script_dir, "weights")
model_file_name = f"sae_m{m_bottleneck}_final_100k.pt"
final_weights_path = os.path.join(weights_folder, model_file_name)

sae_model.load_state_dict(torch.load(final_weights_path, map_location=device))
sae_model.eval()


dataset = load_dataset("openwebtext", split="train", streaming=True)
skip_dataset = dataset.skip(600_000) # skip 400k data so that there is no collision between training and testing data

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
    q_max = int(2**bit - 1)
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

        seq_len = batch.size(1)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Start with Embeddings
        x = model.transformer.wte(batch) + model.transformer.wpe(pos)
        x = model.transformer.drop(x)
        
        # Run up to the input of Layer 3 (Index 2)
        for i in range(2):
            x = model.transformer.h[i](x)[0]
        
        acts = x # This is now the actual input Layer 3 expects
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
                damaged_feats = quantize_gen(min=global_min, 
                                             max=global_max, 
                                             bit=bits,
                                             tensor=sparse_feats) 
            elif mode == "per_feature":
                damaged_feats = quantize_gen(
                    min = feature_min.unsqueeze(0).unsqueeze(0),
                    max = feature_max.unsqueeze(0).unsqueeze(0),
                    bit=bits, tensor=sparse_feats
                )
            reconstructed = sae_model.decoder(damaged_feats)
        else:
            reconstructed = orig_acts  # baseline
            
        return (reconstructed,) + output[1:]
    return hook

bit_configs = [None, 8, 6, 4, 2]
eval_batches = list(get_eval_batches(skip_dataset, num_batches=5))
modes = ["per_tensor", "per_feature"]

umap_data = {
    "Baseline": [],
    'per_tensor': {
    "8-bit_per_tensor": [],
    "6-bit_per_tensor": [],
    "4-bit_per_tensor": [],
    "2-bit_per_tensor": []
    },
    'per_feature': {
    "8-bit_per_feature": [],
    "6-bit_per_feature": [],
    "4-bit_per_feature": [],
    "2-bit_per_feature": []
    }
}

def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

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
            name = f"{mode}_{bits}_32_batch_{m_bottleneck}",
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
                seq_len = batch.size(1)
                pos = torch.arange(seq_len, device=device).unsqueeze(0)
                x = model.transformer.wte(batch) + model.transformer.wpe(pos)
                x = model.transformer.drop(x)
                for i in range(2): 
                    x = model.transformer.h[i](x)[0]
                clean_acts = x
                _, sparse_feats = sae_model(clean_acts)
                Z_batch = sparse_feats.clone()
                if bits is not None:
                    if mode=="per_tensor":    
                        Z_hat_batch = quantize_gen(min=global_min, 
                                             max=global_max, 
                                             bit=bits,
                                             tensor=sparse_feats) 
                    elif mode == "per_feature":
                        Z_hat_batch = quantize_gen(
                        min = feature_min.unsqueeze(0).unsqueeze(0),
                        max = feature_max.unsqueeze(0).unsqueeze(0),
                        bit=bits, tensor=sparse_feats
                )
                else:
                    Z_hat_batch = Z_batch
                Z = Z_batch.view(-1, Z_batch.size(-1))
                Z_hat = Z_hat_batch.view(-1, Z_batch.size(-1))

                recon_acts = sae_model.decoder(Z_hat_batch)
                
                total_mse += nn.MSELoss()(recon_acts, clean_acts).item()
                total_sds += SDS(Z=Z, Z_hat=Z_hat, k=32)
                total_cka += CKA(X=recon_acts.view(-1, 768), Y = clean_acts.view(-1, 768))

                Z_hat_np = (Z_hat.detach().cpu().numpy())
                if bits is None: umap_data["Baseline"].append(Z_hat_np)
                else: umap_data[f"{mode}"][f"{bits}-bit_{mode}"].append(Z_hat_np)

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

wandb.finish()
print("Running UMAP...")
for mode in modes:
    reducer = umap.UMAP(metric='euclidean', min_dist=0.1, random_state=67)
    
    # Grab the baseline from the TOP level of the dictionary
    baseline = np.concatenate(umap_data["Baseline"], axis=0)
    baseline = baseline.reshape(-1, baseline.shape[-1])[:10000]

    # Fit the reducer ONLY to the continuous baseline
    reducer.fit(baseline)
    baseline_2d = reducer.transform(baseline)

    # 5 subplots for [Baseline, 8, 6, 4, 2]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"UMAP Grid Collapse ({mode})", fontsize=16)

    # These are the titles for our subplots
    ordered_labels = ["Baseline", "8-bit", "6-bit", "4-bit", "2-bit"]
    
    for i, label in enumerate(ordered_labels):
        # 1. Fetch the correct data list based on the label
        if label == "Baseline":
            data_list = umap_data["Baseline"]
        else:
            # Reconstruct the exact dictionary key (e.g., "8-bit_per_tensor")
            dict_key = f"{label}_{mode}" 
            if dict_key not in umap_data[mode]: 
                continue # Skip if this bit configuration didn't run
            data_list = umap_data[mode][dict_key]
            
        # 2. Process the data
        data = np.concatenate(data_list, axis=0)
        data = data.reshape(-1, data.shape[-1])[:10000]

        # 3. Transform and Plot
        proj = baseline_2d if label == "Baseline" else reducer.transform(data)

        axes[i].scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.5)
        axes[i].set_title(label)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"umap_fixed_{mode}_{m_bottleneck}.png", dpi=300)
    print(f"Saved: umap_fixed_{mode}.png")