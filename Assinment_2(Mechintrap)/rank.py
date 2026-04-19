from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
model.eval()

m_bottleneck = 512*2

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

def compute_neuron_scores(Z, Z_hat, bins=50):
    # L2 (normalized)
    l2 = torch.norm(Z - Z_hat, dim=0) / Z.size(0)

    # KL divergence per neuron
    D = Z.size(1)
    kl = torch.zeros(D, device=Z.device)

    for j in range(D):
        z = Z[:, j]
        z_hat = Z_hat[:, j]

        min_val = min(z.min().item(), z_hat.min().item())
        max_val = max(z.max().item(), z_hat.max().item())

        if max_val - min_val < 1e-6:
            continue  # avoid degenerate bins

        p = torch.histc(z, bins=bins, min=min_val, max=max_val)
        q = torch.histc(z_hat, bins=bins, min=min_val, max=max_val)

        p = (p + 1e-9) / (p.sum() + 1e-9 * bins)
        q = (q + 1e-9) / (q.sum() + 1e-9 * bins)

        kl[j] = torch.sum(p * torch.log(p / q))

    return l2, kl

def spectral_analysis(Z, Z_hat, k=32):

    Z_c = Z - Z.mean(dim=0, keepdim=True)
    Z_c_hat = Z_hat - Z_hat.mean(dim=0, keepdim=True)

    U, S, Vh = torch.linalg.svd(Z_c, full_matrices=False)
    U_hat, S_hat, Vh_hat = torch.linalg.svd(Z_c_hat, full_matrices=False)

    Uk = Vh[:k].T
    Uk_hat = Vh_hat[:k].T

    cos_thetas = torch.linalg.svdvals(Uk.T @ Uk_hat)
    angles = torch.acos(torch.clamp(cos_thetas, -1, 1)) * (180 / math.pi)

    # SDS
    E = Z - Z_hat
    sds = (torch.norm(E @ Uk)**2 / torch.norm(Z @ Uk)**2).item()

    return {
        "singular_before": S,
        "singular_after": S_hat,
        "angles": angles,
        "sds": sds
    }

def run_subset_ablation(model, sae_model, eval_batches, neurons, device):
    def compute_loss():
        total_loss = 0
        with torch.no_grad():
            for batch in eval_batches:
                outputs = model(batch, labels=batch)
                total_loss += outputs.loss.item()
        return math.exp(total_loss / len(eval_batches))

    baseline_ppl = compute_loss()

    def hook(module, input, output):
        acts = output[0]
        _, feats = sae_model(acts)

        feats[..., neurons] = 0  

        recon = sae_model.decoder(feats)
        return (recon,) + output[1:]

    handle = model.transformer.h[2].register_forward_hook(hook)
    ablated_ppl = compute_loss()
    handle.remove()

    return (ablated_ppl - baseline_ppl)/abs(baseline_ppl+1e-9)

def run_global_ablation(model, sae_model, eval_batches, device, m_bottleneck=512):
    results = {}
    
    def compute_ppl():
        total_loss = 0
        with torch.no_grad():
            for batch in eval_batches:
                outputs = model(batch, labels=batch)
                total_loss += outputs.loss.item()
        return math.exp(total_loss / len(eval_batches))

    baseline_ppl = compute_ppl()
    
    # Sweep all 512 neurons to find the true causal landscape
    for n in tqdm(range(m_bottleneck), desc="Ablating Neurons", leave=False):
        def hook(module, input, output, neuron=n):
            acts = output[0]
            _, feats = sae_model(acts)
            feats[..., neuron] = 0

            ablated_acts = sae_model.decoder(feats)
            return (ablated_acts,) + output[1:]

        handle = model.transformer.h[2].register_forward_hook(hook)
        ablated_ppl = compute_ppl()
        results[n] = (ablated_ppl - baseline_ppl)/abs(baseline_ppl+1e-9)
        handle.remove()

    return results

def compute_strict_alignment(l2_scores, kl_scores, ablation_dict, k_top=20):
    # 1. Convert dict to array spanning 0-511'=
    ablation_array = [ablation_dict[i] for i in range(len(l2_scores))]
    l2_np = l2_scores.cpu().numpy()
    kl_np = kl_scores.cpu().numpy()
    ablation_array = np.array(ablation_array)

    l2_np = (l2_np - l2_np.mean())/(l2_np.std() + 1e-9)
    kl_np = (kl_np - kl_np.mean())/(kl_np.std() + 1e-9)
    ablation_array = (ablation_array - ablation_array.mean())/(ablation_array.std() + 1e-9)

    l2_corr, _ = spearmanr(l2_np, ablation_array)
    kl_corr, _ = spearmanr(kl_np, ablation_array)

    l2_top = set(torch.topk(l2_scores, k_top).indices.tolist())
    kl_top = set(torch.topk(kl_scores, k_top).indices.tolist())
    
    sorted_ablation = sorted(ablation_dict.items(), key=lambda x: x[1], reverse=True)
    impactful_top = set([n for n, _ in sorted_ablation[:k_top]])
    
    return {
        "spearman_l2": float(l2_corr),
        "spearman_kl": float(kl_corr),
        "l2_overlap": len(l2_top & impactful_top) / k_top,
        "kl_overlap": len(kl_top & impactful_top) / k_top
    }

def get_top_tokens(Z, tokens, neurons, k=10):
    results = {}

    for n in neurons:
        vals, idx = torch.topk(Z[:, n], k)
        results[n] = tokenizer.batch_decode(tokens[idx])
    return results

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


bit_configs = [8, 6, 4, 2, None]
modes = ["per_tensor"]
umap_data = {
    "Baseline": [],
    'per_tensor': {
    "8-bit_per_tensor": [],
    "6-bit_per_tensor": [],
    "4-bit_per_tensor": [],
    "2-bit_per_tensor": []
    }
}

results = {}

for mode in modes:
    print(f"\nMode: {mode}")

    for bits in bit_configs:
        label = f"{bits}-bit" if bits else "Baseline"

        wandb.init(
        project="sae-quantization-pt2",
        name=f"{mode}_{label}_analysis",
        reinit=True
        )
        label = f"{bits}-bit" if bits else "Baseline"
        print(f"\nEvaluating: {label}")
        Z_all = []
        Z_hat_all = []
        tokens_all = []

        handle = (None)                
        with torch.no_grad():
            idx = 0
            for batch in tqdm(eval_batches, leave=False):
                outputs = model(batch, labels=batch)
                seq_len = batch.size(1)
                pos = torch.arange(seq_len, device=device).unsqueeze(0)

                x = model.transformer.wte(batch) + model.transformer.wpe(pos)
                x = model.transformer.drop(x)

                for i in range(3):   # include layer 3
                    x = model.transformer.h[i](x)[0]

                clean_acts = x
                _, sparse_feats = sae_model(clean_acts)

                Z_batch = sparse_feats

                if bits is not None:
                    Z_hat_batch = quantize_gen(
                        min=global_min,
                        max=global_max,
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

                recon_acts = sae_model.decoder(Z_hat_batch)
                
                Z_hat_np = (Z_hat.detach().cpu().numpy())
                if bits is None: umap_data["Baseline"].append(Z_hat_np)
                else: umap_data[f"{mode}"][f"{bits}-bit_{mode}"].append(Z_hat_np)

        Z = torch.cat(Z_all)
        Z_hat = torch.cat(Z_hat_all)
        tokens = torch.cat(tokens_all)

        l2, kl = compute_neuron_scores(Z, Z_hat)
        spectral = spectral_analysis(Z, Z_hat)

        wandb.log({
        "l2_mean": l2.mean().item(),
        "l2_max": l2.max().item(),
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "sds": spectral["sds"],
        "angle_mean": spectral["angles"].mean().item()
        })

        wandb.log({
        "l2_hist": wandb.Histogram(l2.cpu().numpy()),
        "kl_hist": wandb.Histogram(kl.cpu().numpy())
        })

        import matplotlib.pyplot as plt

        plt.figure()

        plt.plot(spectral["singular_before"].cpu().numpy(), label="before")
        plt.plot(spectral["singular_after"].cpu().numpy(), label="after")

        plt.legend()
        plt.title(f"Singular Values ({label})")

        plt.savefig(f"svd_{label}_m_{m_bottleneck}.png", dpi=300, bbox_inches="tight")
        plt.close()
                
        k = 20

        top_neurons_kl = torch.topk(kl, k=k).indices.tolist()
        top_neurons_l2 = torch.topk(l2, k=k).indices.tolist()

        top_tokens_kl = get_top_tokens(Z, tokens, top_neurons_kl, k)
        top_tokens_l2 = get_top_tokens(Z, tokens, top_neurons_l2, k)

        print("\nTop KL neurons + tokens:")
        for n, toks in top_tokens_kl.items():
            print(f"Neuron {n}: {toks[:k]}")

        print("\nTop L2 neurons + tokens:")
        for n, toks in top_tokens_l2.items():
            print(f"Neuron {n}: {toks[:k]}")

        global_ablation = run_global_ablation(model, sae_model, eval_batches, device, m_bottleneck=m_bottleneck)

        random_neurons = random.sample(range(len(l2)), k)

        delta_l2 = run_subset_ablation(model, sae_model, eval_batches, top_neurons_l2, device)
        delta_kl = run_subset_ablation(model, sae_model, eval_batches, top_neurons_kl, device)
        delta_rand = run_subset_ablation(model, sae_model, eval_batches, random_neurons, device)
        
        alignment_stats = compute_strict_alignment(l2, kl, global_ablation)

        # 3. Log to WandB
        wandb.log({
            "alignment/spearman_l2": alignment_stats['spearman_l2'],
            "alignment/spearman_kl": alignment_stats['spearman_kl'],
            "alignment/l2_overlap": alignment_stats['l2_overlap'],
            "alignment/kl_overlap": alignment_stats['kl_overlap'],
            "ablation/max_delta_ppl": max(global_ablation.values()),
            "ablation/l2_delta_loss": delta_l2,
            "ablation/kl_delta_loss": delta_kl,
            "ablation/random_delta_loss": delta_rand,
            "ablation/mean_delta_ppl": sum(global_ablation.values()) / len(global_ablation)
        })
        results[bits] = {
            "l2": l2,
            "kl": kl,
            "spectral": spectral,
            
            # The decoded tokens for the top damaged neurons
            'top_neurons_l2': top_tokens_l2,
            'top_neuron_tokens_kl': top_tokens_kl,

            # We now store the entire causal landscape in one dictionary
            "global_ablation": global_ablation,
            
            # This now contains both Spearman correlations and the true Top 10 overlap
            "alignment_stats": alignment_stats 
        }

        print(f" Summary ({label})")
        print(f"Representation Damage:")
        print(f"L2 Mean: {l2.mean().item():.4f}")
        print(f"KL Mean: {kl.mean().item():.4f}")
        print(f"SDS: {spectral['sds']:.4f}")
        print(f"Angle: {spectral['angles'].mean().item():.2f}°")
        
        print(f"\nCausal Alignment (Damage vs Perplexity Impact):")
        print(f"L2 Spearman: {alignment_stats['spearman_l2']:.4f}")
        print(f"KL Spearman: {alignment_stats['spearman_kl']:.4f}")
        
        print(f"Top 10 L2 Overlap: {alignment_stats['l2_overlap'] * 100:.1f}%")
        print(f"Top 10 KL Overlap: {alignment_stats['kl_overlap'] * 100:.1f}%")

print("<------------------- ABLATION FINISHED --------------------------->")
print("<------------------- SUCCESS --------------------------->")